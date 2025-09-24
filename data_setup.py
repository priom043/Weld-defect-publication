"""
Contains functionality for creating PyTorch DataLoaders for Image classification data
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int,
    transform=transforms.Compose,
    val_dir: str = None,
    num_workers: int = NUM_WORKERS
):
    """
    Creates Training, (Optional Validation), and Testing DataLoaders

    Args:
        train_dir: Path to training directory
        test_dir: Path to test directory
        batch_size: Number of samples per batch in each DataLoader
        transform: torchvision transforms to perform on datasets
        val_dir: (Optional) Path to validation directory
        num_workers: Number of worker processes per DataLoader
    
    Returns:
        If val_dir is provided:
            (train_dataloader, val_dataloader, test_dataloader, class_names)
        Else:
            (train_dataloader, test_dataloader, class_names)
    """
    # Create datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    class_names = train_data.classes

    if val_dir:
        val_data = datasets.ImageFolder(root=val_dir, transform=transform)
        val_dataloader = DataLoader(val_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True)
        return train_dataloader, val_dataloader, test_dataloader, class_names
    else:
        return train_dataloader, test_dataloader, class_names