import os
import random
from typing import cast

import torch
import torchvision
import torchvision.datasets as datasets
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(50),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing()
    ])

    train_dataset, _ = create_datasets(
        config["image_net_data_dir"], data_transform)

    print(train_dataset[0])
