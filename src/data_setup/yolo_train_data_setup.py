import os
import random
from typing import Callable, cast

import torch
import torchvision
import torchvision.datasets as datasets
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2

config = yaml.safe_load(open("config.yaml"))

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class_to_index = {cls_name: idx for idx,
                  cls_name in enumerate(VOC_CLASSES)}


def yolo_target_transform(annotation: dict) -> torch.Tensor:
    """
    labels[..., 2] and labels[..., 7] are the width of bbox 1 and bbox 2
    if a grid cell has no labels, then labels[..., 2] and labels[..., 7] will be 0
    if a grid cell has only one label, then labels[..., 2] will be 0
    """
    # a = {'annotation': {'folder': 'VOC2012', 'filename': '2008_000008.jpg', 'source': {'database': 'The VOC2008 Database', 'annotation': 'PASCAL VOC2008', 'image': 'flickr'}, 'size': {'width': '500', 'height': '442', 'depth': '3'}, 'segmented': '0', 'object': [
    # {'name': 'horse', 'pose': 'Left', 'truncated': '0', 'occluded': '1', 'bndbox': {'xmin': '53', 'ymin': '87', 'xmax': '471', 'ymax': '420'}, 'difficult': '0'}, {'name': 'person', 'pose': 'Unspecified', 'truncated': '1', 'occluded': '0', 'bndbox': {'xmin': '158', 'ymin': '44', 'xmax': '289', 'ymax': '167'}, 'difficult': '0'}]}}

    # Convert the annotations to a four tuple of tensors
    objects = annotation["annotation"]["object"]
    image_width = int(annotation["annotation"]["size"]["width"])
    image_height = int(annotation["annotation"]["size"]["height"])

    grid_size = 7

    # grid_cells = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    B = 2
    C = 20
    depth = B * 5 + C
    grid_cells = torch.zeros((grid_size, grid_size, depth))

    grid_cell_to_class = [
        [-1 for _ in range(grid_size)] for _ in range(grid_size)]
    grid_cell_to_num_bboxes = [
        [0 for _ in range(grid_size)] for _ in range(grid_size)]

    for object in objects:
        # normalize the bounding box coordinates
        x_min, x_max = int(object["bndbox"]["xmin"]), int(
            object["bndbox"]["xmax"])
        y_min, y_max = int(object["bndbox"]["ymin"]), int(
            object["bndbox"]["ymax"])

        # get the center of the bounding box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # get grid cell coordinates
        grid_x = int((x_center / image_width) * grid_size)
        grid_y = int((y_center / image_height) * grid_size)

        # get x, y offsets from the grid cell.
        x = (x_center / image_width) * grid_size - grid_x
        y = (y_center / image_height) * grid_size - grid_y

        # get width and height of the bounding box relative to the image
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        class_index = class_to_index[object["name"]]
        confidence = 1.0

        # if class label has not been set for this grid cell, then set it.
        if grid_cell_to_class[grid_y][grid_x] == -1:
            one_hot = torch.zeros(C)
            one_hot[class_index] = 1
            grid_cell_to_class[grid_x][grid_y] = class_index
            grid_cells[grid_x, grid_y, B*5:] = one_hot

        # if the class label is not set or class label is the same as the one already set, then set bounding box coordinates.
        num_bboxes_per_grid_cell = grid_cell_to_num_bboxes[grid_x][grid_y]
        if (grid_cell_to_class[grid_y][grid_x] == -1 or grid_cell_to_class[grid_y][grid_x] == class_index) and num_bboxes_per_grid_cell < B:
            # if class label has been set for this grid cell, then set bounding box coordinates for first 2 boxes if they are of the same class.
            grid_cells[grid_x, grid_y, num_bboxes_per_grid_cell*5:num_bboxes_per_grid_cell *
                       5+5] = torch.tensor([x, y, width, height, confidence])
            grid_cell_to_num_bboxes[grid_x][grid_y] += 1

    return grid_cells


def create_datasets(root_dir: str, transform: transforms_v2.Compose) -> tuple[Dataset, Dataset]:
    train_dataset = datasets.VOCDetection(root=root_dir,
                                          year='2012',
                                          image_set="train",
                                          transform=transform,
                                          download=False,
                                          target_transform=yolo_target_transform)

    val_dataset = datasets.VOCDetection(root=root_dir,
                                        year='2012',
                                        image_set="val",
                                        transform=transform,
                                        download=False,
                                        target_transform=yolo_target_transform)
    return train_dataset, val_dataset


def create_dataloaders(root_dir: str,
                       transform: transforms_v2.Compose,
                       batch_size: int,
                       num_workers: int) -> tuple[DataLoader, DataLoader]:

    train_dataset, val_dataset = create_datasets(
        root_dir, transform)

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

    # you're gonna have a separate transform for visualization anyways so youre good.

    # label_transform =
