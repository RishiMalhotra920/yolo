import torchvision.datasets as datasets
import yaml
from typing import cast
import torchvision
from torch.utils.data import DataLoader
import random
from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader
config = yaml.safe_load(open("config.yaml"))


def create_datasets(root_dir: str, transform: torchvision.transforms.Compose) -> tuple[Dataset, Dataset]:
    train_dataset = datasets.ImageNet(root=root_dir,
                                      split='train',
                                      transform=transform,
                                      target_transform=None)

    val_dataset = datasets.ImageNet(root=root_dir,
                                    split='val',
                                    transform=transform,
                                    target_transform=None)
    return train_dataset, val_dataset


def create_dataloaders(root_dir: str, transform: torchvision.transforms.Compose, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:

    train_dataset, val_dataset = create_datasets(root_dir, transform)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_data_loader, val_data_loader
