import random
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2 as transforms_v2

from src.utils import class_to_index

config = yaml.safe_load(open("config.yaml"))


def categorize_bboxes_into_grid(
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    image_width: int,
    image_height: int,
    grid_size: int = 7,
):
    """

    Args:
        bboxes (torch.Tensor) of shape (N, 4) where N is the number of bounding boxes.
        Each bounding box is represented as [x1, y1, x2, y2]
        labels: (torch.Tensor) of shape (N,) where each element is the label of the corresponding bounding box
        image_width (int): Width of the image
        image_height (int): Height of the image
        grid_size (int): Number of cells in the grid

    Returns:
        torch.Tensor: Tensor of shape (grid_size, grid_size, 30) where each cell contains
        information about the bounding boxes in that cell.


    """

    # Initialize the output tensor
    output = torch.zeros((grid_size, grid_size, 30))

    # Calculate cell width and height
    cell_width = image_width / grid_size
    cell_height = image_height / grid_size

    # Dictionary to store bboxes and labels for each cell
    cell_bboxes: dict[tuple[int, int], list[tuple[torch.Tensor, int]]] = {
        (i, j): [] for i in range(grid_size) for j in range(grid_size)
    }

    # Categorize bboxes into cells
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        cell_x = int(center_x // cell_width)
        cell_y = int(center_y // cell_height)

        # Ensure cell coordinates are within bounds
        cell_x = max(0, min(cell_x, grid_size - 1))
        cell_y = max(0, min(cell_y, grid_size - 1))

        cell_bboxes[(cell_x, cell_y)].append((bbox, int(label.item())))

    # Process each cell
    for (cell_x, cell_y), boxes_and_labels in cell_bboxes.items():
        if not boxes_and_labels:
            continue

        # If there are multiple boxes, handle constraints
        if len(boxes_and_labels) > 1:
            # Group boxes by label
            label_groups: dict[int, Any] = {}
            for box, label in boxes_and_labels:
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append((box, label))

            # If there are multiple labels, choose randomly
            if len(label_groups) > 1:
                chosen_label = random.choice(list(label_groups.keys()))
                boxes_and_labels = label_groups[chosen_label]
            else:
                boxes_and_labels = list(label_groups.values())[0]

            # Take at most two boxes
            boxes_and_labels = boxes_and_labels[:2]

        # Fill the output tensor
        for i, (box, label) in enumerate(boxes_and_labels):
            if i >= 2:  # Ensure we don't exceed two boxes per cell
                break
            x1, y1, x2, y2 = box

            # Convert to cell-relative coordinates
            x = ((x1 + x2) / 2 % cell_width) / cell_width
            y = ((y1 + y2) / 2 % cell_height) / cell_height
            w = (x2 - x1) / image_width
            h = (y2 - y1) / image_height

            # Fill bbox info
            output[cell_x, cell_y, i * 5 : i * 5 + 5] = torch.tensor([x, y, w, h, 1.0])

        # Fill one-hot encoded label
        label_index = boxes_and_labels[0][1]  # Use the label of the first box
        output[cell_x, cell_y, 10 + label_index] = 1.0

    return output


class CustomVOCDetection(VOCDetection):
    def __init__(
        self,
        root,
        year: str,
        image_set: str,
        transform: transforms_v2.Compose,
    ):
        super().__init__(
            root, year="2012", image_set="train", transform=None, target_transform=None
        )
        self.transform = transform

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        image, annotation = super().__getitem__(index)

        # transformed_image,  = self.transform(image)

        objects = annotation["annotation"]["object"]
        image_width = int(annotation["annotation"]["size"]["width"])
        image_height = int(annotation["annotation"]["size"]["height"])

        metadata = {
            "image_id": index,
            "image_width": image_width,
            "image_height": image_height,
        }

        boxes = []

        # create bboxes and images of the format
        # [image, bbox] where bbox is of format [xmin, ymin, w, h]

        labels = []
        for object in objects:
            # normalize the bounding box coordinates
            x_min, x_max = int(object["bndbox"]["xmin"]), int(object["bndbox"]["xmax"])
            y_min, y_max = int(object["bndbox"]["ymin"]), int(object["bndbox"]["ymax"])

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_to_index[object["name"]])

        #  missing type annotations here.
        boxes = tv_tensors.BoundingBoxes(
            data=boxes, format="XYXY", canvas_size=(image_height, image_width)
        )  # type: ignore

        out_image, out_boxes = self.transform(image, boxes)
        # out_label = torch.tensor(labels)
        out_label = categorize_bboxes_into_grid(
            out_boxes, torch.tensor(labels), image_width, image_height
        )

        return out_image, out_label, metadata


def create_datasets(
    root_dir: str, transform: transforms_v2.Compose
) -> tuple[Dataset, Dataset]:
    train_dataset = CustomVOCDetection(
        root=root_dir,
        year="2012",
        image_set="train",
        transform=transform,
        # download=False,
        # target_transform=target_transform,
    )

    val_dataset = CustomVOCDetection(
        root=root_dir,
        year="2012",
        image_set="val",
        transform=transform,
        # download=False,
        # target_transform=target_transform,
    )
    return train_dataset, val_dataset


def create_dataloaders(
    root_dir: str,
    transform: transforms_v2.Compose,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(root_dir, transform)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_data_loader, val_data_loader
