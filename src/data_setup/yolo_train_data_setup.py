import os
import random
from typing import cast

import torch
import torchvision
import torchvision.datasets as datasets
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2

config = yaml.safe_load(open("config.yaml"))


def create_datasets(root_dir: str, transform: transforms_v2.Compose) -> tuple[Dataset, Dataset]:
    train_dataset = datasets.VOCDetection(root=root_dir,
                                          year='2012',
                                          image_set="train",
                                          transform=transform,
                                          download=False,
                                          target_transform=None)

    val_dataset = datasets.VOCDetection(root=root_dir,
                                        year='2012',
                                        image_set="val",
                                        transform=transform,
                                        download=False,
                                        target_transform=None)
    return train_dataset, val_dataset


def create_dataloaders(root_dir: str, transform: transforms_v2.Compose, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:

    train_dataset, val_dataset = create_datasets(root_dir, transform)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_data_loader, val_data_loader


if __name__ == "__main__":

    data_transform = transforms_v2.Compose([
        transforms_v2.Resize((448, 448)),
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.RandomAffine(
            degrees=(0, 0), translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms_v2.ColorJitter(brightness=0.5, contrast=0.5),
        transforms_v2.ToTensor(),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

    train_dataset, _ = create_datasets(
        "/Users/rishimalhotra/projects/cv/image_classification/datasets",
        data_transform
    )

    print("image shape", train_dataset[0][0].shape)
    print("target", train_dataset[0][1])

    a = {'annotation': {'folder': 'VOC2012', 'filename': '2008_000008.jpg', 'source': {'database': 'The VOC2008 Database', 'annotation': 'PASCAL VOC2008', 'image': 'flickr'}, 'size': {'width': '500', 'height': '442', 'depth': '3'}, 'segmented': '0', 'object': [
        {'name': 'horse', 'pose': 'Left', 'truncated': '0', 'occluded': '1', 'bndbox': {'xmin': '53', 'ymin': '87', 'xmax': '471', 'ymax': '420'}, 'difficult': '0'}, {'name': 'person', 'pose': 'Unspecified', 'truncated': '1', 'occluded': '0', 'bndbox': {'xmin': '158', 'ymin': '44', 'xmax': '289', 'ymax': '167'}, 'difficult': '0'}]}}

    # TODO: create a transform out of this next ^
    # and turn this into a tensor using a label transform of shape
    # [0 .. 2]: bbox_1_x, bbox_2_x
    # [3 .. 4]: bbox_1_y, bbox_2_y
    # [5 .. 6]: bbox_1_w, bbox_2_w
    # [7 .. 8]: bbox_1_h, bbox_2_h
    # [9]: bbox_1_confidence
    # [10]: bbox_2_confidence
    # [11 .. 30]: class probabilities

    # you're gonna have a separate transform for visualization anyways so youre good.

    # label_transform =
