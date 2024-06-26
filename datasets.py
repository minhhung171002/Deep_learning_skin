
import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

class LesionDataset(Dataset):
    def __init__(self, img_dir, labels_fname, augment=False):
        self.img_dir = img_dir
        self.data = pd.read_csv(labels_fname)
        self.augment = augment
        self.transform = self.build_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")

        # Handle missing files
        if not os.path.exists(img_path):
            # You could choose to either:
            # - Raise an error
            # raise FileNotFoundError(f"File not found: {img_path}")
            # - Or skip the file by returning None or a placeholder
            print(f"Warning: File not found: {img_path}")
            return None, None
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            label_values = self.data.iloc[index, 1:].values.astype(np.float32)
            label_tensor = torch.tensor(label_values, dtype=torch.float32)

            return img, label_tensor

    def build_transforms(self):
        target_size = (224, 224)
        base_transform = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        if self.augment:
            augmentation_transform = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),   # Reduced the rotation angle
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Reduced jitter values
                transforms.RandomResizedCrop(target_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)), # Adjusted scales and ratios
            ]
            return transforms.Compose(augmentation_transform + base_transform)
        else:
            return transforms.Compose(base_transform)