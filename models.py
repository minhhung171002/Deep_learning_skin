import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
# TODO Task 1c - Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBNConv, self).__init__()

        # Define convolutional layers with batch norm and ReLU activations
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=128)

        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the features after the conv and pooling layers
        # Assuming the input image size is 224x224
        # This is a simple calculation for a square image and square kernels with stride=2 in the max pooling layers.
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        feature_size = 7 * 7 * 128

        # Define fully connected layers
        self.fc1 = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        # Apply conv layers, followed by batch norm, ReLU activation and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer and return the output
        x = self.fc1(x)
        return x
# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.
class ResNet18Model(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_weights=True):
        super(ResNet18Model, self).__init__()
        
        # Load the pre-trained ResNet18 model
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        
        # Modify the last fully connected layer to match the number of classes
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)
        
        if freeze_weights:
            # Freeze all layers except the final one
            for param in self.resnet18.parameters():
                param.requires_grad = False
            
            # Unfreeze the final fully connected layer
            for param in self.resnet18.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet18(x)


# TODO Task 1f - Create your own models
class SimpleBNConvModified(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBNConvModified, self).__init__()

        # Define convolutional layers with batch norm and ReLU activations
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=256)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the features after the conv and pooling layers
        feature_size = 3 * 3 * 256  # After 6 pooling operations, 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 3

        # Define fully connected layers
        self.fc1 = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        # Apply conv layers, followed by batch norm, ReLU activation and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout(self.pool(F.relu(self.bn6(self.conv6(x)))))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer and return the output
        x = self.fc1(x)
        return x

class ResNet50Model(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_weights=True):
        super(ResNet50Model, self).__init__()
        
        # Load the pre-trained ResNet50 model
        self.resnet50 = torchvision.models.resnet50(pretrained=pretrained)
        
        # Modify the last fully connected layer to match the number of classes
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        
        if freeze_weights:
            # Freeze all layers except the final one
            for param in self.resnet50.parameters():
                param.requires_grad = False
            
            # Unfreeze the final fully connected layer
            for param in self.resnet50.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet50(x)
