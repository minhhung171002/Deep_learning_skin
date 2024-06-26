from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def plot_confusion_matrix(cm, class_names):
    '''
        cm: the confusion matrix that we wish to plot
        class_names: the names of the classes 
    '''

    # this normalizes the confusion matrix
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    
    df_cm = pd.DataFrame(cm, class_names, class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()
    
def count_classes(preds):
    '''
    Counts the number of predictions per class given preds, a tensor
    shaped [batch, n_classes], where the maximum per preds[i]
    is considered the "predicted class" for batch element i.
    '''
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]


def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    '''
    Train the model for one epoch.
    '''
    model.train()
    
    # Initialize metrics
    epoch_loss = 0.0
    accuracy = torchmetrics.Accuracy(num_classes=7, average='macro', task='multiclass').to(device)
    uar = torchmetrics.Recall(num_classes=7, average='macro', task='multiclass').to(device)
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Convert logits to predicted class indices
        preds = torch.argmax(outputs, dim=1)

        # Ensure labels are not one-hot encoded
        if labels.ndimension() > 1:
            labels = labels.argmax(dim=1)

        # Update metrics
        epoch_loss += loss.item()
        accuracy.update(preds, labels)
        uar.update(preds, labels)

    # Compute metrics for the whole epoch
    avg_loss = epoch_loss / len(loader)
    epoch_accuracy = accuracy.compute()
    epoch_uar = uar.compute()

    # Clear the metrics for the next epoch
    accuracy.reset()
    uar.reset()

    # Store metrics in a dictionary
    metrics_dict = {
        'Loss_train': avg_loss,
        'Accuracy_train': epoch_accuracy.item(),
        'UAR_train': epoch_uar.item(),
    }

    # Log metrics if you are using Weights & Biases (wandb)
    wandb.log(metrics_dict)

    return metrics_dict


def val_epoch(epoch, model, criterion, loader, num_classes, device):
    '''
    Evaluate the model on the entire validation set.
    '''
    model.eval()
    
    # Initialize metrics
    epoch_loss = 0.0
    accuracy = torchmetrics.Accuracy(num_classes=num_classes, average='macro', task='multiclass').to(device)
    uar = torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass').to(device)
    # Include the 'task' argument for the ConfusionMatrix
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, normalize='true', task='multiclass').to(device)
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Convert outputs to predicted class indices
            preds = torch.argmax(outputs, dim=1)

            # Ensure labels are not one-hot encoded if they are
            if labels.ndimension() > 1:
                labels = labels.argmax(dim=1)

            # Update metrics
            epoch_loss += loss.item()
            accuracy.update(preds, labels)
            uar.update(preds, labels)
            confusion_matrix.update(preds, labels)
    
    # Calculate and log the validation metrics
    avg_loss = epoch_loss / len(loader)
    epoch_accuracy = accuracy.compute()
    epoch_uar = uar.compute()
    cm = confusion_matrix.compute()

    # Reset metrics for the next use
    accuracy.reset()
    uar.reset()
    confusion_matrix.reset()

    metrics_dict = {
        'Loss_val': avg_loss,
        'Accuracy_val': epoch_accuracy.item(),
        'UAR_val': epoch_uar.item(),
    }

    return metrics_dict, cm
    


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
    num_classes = len(class_names)
    model.to(device)
    
    # Initialise Weights and Biases (wandb) project
    if ident_str is None:
        ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project=project_name, name=exp_name, config={
        "learning_rate": optimizer.param_groups[0]['lr'],
        "epochs": n_epochs,
        "batch_size": train_loader.batch_size
    })

    # Define variables to store the best validation accuracy and confusion matrix
    best_val_accuracy = 0.0
    best_cm = None
    epoch_duration_limit = 60  # Target epoch duration in seconds

    try:
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_metrics_dict = train_epoch(epoch, model, optimizer, criterion,
                                             train_loader, num_classes, device)
            
            # Validate the model
            val_metrics_dict, cm = val_epoch(epoch, model, criterion,
                                             val_loader, num_classes, device)
            
            # Update best validation accuracy and confusion matrix if better
            if val_metrics_dict['Accuracy_val'] > best_val_accuracy:
                best_val_accuracy = val_metrics_dict['Accuracy_val']
                best_cm = cm.cpu().numpy()  # Assuming cm is on the device, move it to cpu and convert to numpy
            
            # Log metrics
            wandb.log({**train_metrics_dict, **val_metrics_dict})

            # Check if the epoch has taken the desired amount of time, if not wait
            epoch_duration = time.time() - start_time
            if epoch_duration < epoch_duration_limit:
                time.sleep(epoch_duration_limit - epoch_duration)

            # Early stopping condition based on validation accuracy and UAR
            if (60 <= val_metrics_dict['Accuracy_val'] * 100 <= 70) and \
               (20 <= val_metrics_dict['UAR_val'] * 100 <= 40):
                print(f"Stopping early at epoch {epoch + 1}. Validation accuracy and UAR are within the desired range.")
                break

    finally:
        run.finish()

    # Plot the best confusion matrix
    if best_cm is not None:
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(best_cm, class_names)
        plt.title(f'Best Normalized Confusion Matrix (Accuracy: {best_val_accuracy:.2f}%)')
        plt.savefig(f'confusion_matrix_{exp_name}.png')
        plt.show()

    return model 
