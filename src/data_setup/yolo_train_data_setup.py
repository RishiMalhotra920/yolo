from typing import Callable

import torch
import torchvision.datasets as datasets
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2

from src.utils import class_to_index

config = yaml.safe_load(open("config.yaml"))


def yolo_target_transform(annotation: dict) -> torch.Tensor:
    """
    labels[..., 2] and labels[..., 7] are the width of bbox 1 and bbox 2
    if a grid cell has no labels, then labels[..., 2] and labels[..., 7] will be 0
    if a grid cell has only one label, then labels[..., 2] will be 0

    converts annotation to a tensor of shape (7, 7, 30)
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

    grid_cell_to_class = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
    grid_cell_to_num_bboxes = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    # you actually don't need to scale anything to the 448*448 dimension space.
    for object in objects:
        # normalize the bounding box coordinates
        x_min, x_max = int(object["bndbox"]["xmin"]), int(object["bndbox"]["xmax"])
        y_min, y_max = int(object["bndbox"]["ymin"]), int(object["bndbox"]["ymax"])

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
            grid_cells[grid_x, grid_y, B * 5 :] = one_hot

        # if the class label is not set or class label is the same as the one already set, then set bounding box coordinates.
        num_bboxes_per_grid_cell = grid_cell_to_num_bboxes[grid_x][grid_y]
        if (
            grid_cell_to_class[grid_y][grid_x] == -1
            or grid_cell_to_class[grid_y][grid_x] == class_index
        ) and num_bboxes_per_grid_cell < B:
            # if class label has been set for this grid cell, then set bounding box coordinates for first 2 boxes if they are of the same class.
            grid_cells[
                grid_x,
                grid_y,
                num_bboxes_per_grid_cell * 5 : num_bboxes_per_grid_cell * 5 + 5,
            ] = torch.tensor([x, y, width, height, confidence])
            grid_cell_to_num_bboxes[grid_x][grid_y] += 1

    return grid_cells


import random


class CustomVOCDataset(datasets.VOCDetection):
    def __init__(
        self,
        root: str,
        image_set: str,
    ):
        super().__init__(root, year="2012", image_set=image_set, download=False)

    def get_bboxes_from_annotation(self, annotation: dict) -> torch.Tensor:
        objects = annotation["annotation"]["object"]

        bboxes = []

        for object in objects:
            # normalize the bounding box coordinates
            x_min, x_max = int(object["bndbox"]["xmin"]), int(object["bndbox"]["xmax"])
            y_min, y_max = int(object["bndbox"]["ymin"]), int(object["bndbox"]["ymax"])

            bboxes.append([x_min, y_min, x_max, y_max])

        return torch.tensor(bboxes)

    def remove_bboxes_outside_image_bounds(self, bbox, width, height):
        # condition 1: bbox starts after the image ends
        # condition 2: bbox ends before the image starts
        # condition 3: bbox starts after the image ends
        # condition 4: bbox ends before the image starts
        # if any of these conditions are met, then the bbox is outside the image bounds and should be removed.

        print("-----------------")
        print("image width:", width, "image height:", height, "bbox:", bbox)
        print(
            "condition 1",
            bbox[:, 0] >= width,
            "condition 2",
            bbox[:, 2] <= 0,
            "condition 3",
            bbox[:, 1] >= height,
            "condition 4",
            bbox[:, 3] <= 0,
        )

        mask = (
            (bbox[:, 0] >= width)
            | (bbox[:, 2] <= 0)
            | (bbox[:, 1] >= height)
            | (bbox[:, 3] <= 0)
        )
        return bbox[~mask]

    def random_affine_transform(self, image: torch.Tensor, bboxes: torch.Tensor):
        print("this is image shape:", image.shape)
        _, image_height, image_width = image.shape

        translate_x_pixels = random.randint(
            int(-0.2 * image_width), int(0.2 * image_width)
        )
        translate_y_pixels = random.randint(
            int(-0.2 * image_height), int(0.2 * image_height)
        )
        scale = random.uniform(1.0, 2.0)
        print("affining by:", translate_x_pixels, translate_y_pixels, scale)

        image = transforms_v2.functional.affine(
            image,
            angle=0,
            translate=[translate_x_pixels, translate_y_pixels],
            scale=scale,
            shear=[0, 0],
        )

        side_crop = ((scale - 1) * image_width) / 2
        top_crop = ((scale - 1) * image_height) / 2

        def bbox_affine_x_transform(x: torch.Tensor):
            coord_scaled = x * scale - side_crop
            coord_scaled_and_translated = coord_scaled + translate_x_pixels
            coord_scaled_and_translated_and_clipped = torch.clamp(
                coord_scaled_and_translated, min=0, max=image_width
            )

            return coord_scaled_and_translated_and_clipped

        def bbox_affine_y_transform(y: torch.Tensor):
            coord_scaled = y * scale - top_crop
            coord_scaled_and_translated = coord_scaled + translate_y_pixels
            coord_scaled_and_translated_and_clipped = torch.clamp(
                coord_scaled_and_translated, min=0, max=image_height
            )
            return coord_scaled_and_translated_and_clipped

        bboxes[:, 0] = bbox_affine_x_transform(bboxes[:, 0])
        bboxes[:, 2] = bbox_affine_x_transform(bboxes[:, 2])
        bboxes[:, 1] = bbox_affine_y_transform(bboxes[:, 1])
        bboxes[:, 3] = bbox_affine_y_transform(bboxes[:, 3])

        print("interm bboxes", bboxes)

        # affine maintains so send in original image width and height
        bboxes = self.remove_bboxes_outside_image_bounds(
            bboxes, image_width, image_height
        )

        print("bboxes after removing", bboxes)
        return image, bboxes

    def random_resize_transform(self, image: torch.Tensor, bboxes: torch.Tensor):
        print("image shape after resize:", image.shape)
        _, image_height, image_width = image.shape
        image = transforms_v2.functional.resize(image, [448, 448])

        resize_x_scaling_factor = 448 / image_width
        resize_y_scaling_factor = 448 / image_height

        def bbox_x_resize_transform(x):
            return (x * resize_x_scaling_factor).int()

        def bbox_y_resize_transform(y):
            return (y * resize_y_scaling_factor).int()

        print("====================")
        print("resize_x_scaling_factor:", resize_x_scaling_factor)
        print("resize_y_scaling_factor:", resize_y_scaling_factor)
        print("this is bboxes 1:", bboxes)

        bboxes[:, 0] = bbox_x_resize_transform(bboxes[:, 0])
        bboxes[:, 2] = bbox_x_resize_transform(bboxes[:, 2])
        bboxes[:, 1] = bbox_y_resize_transform(bboxes[:, 1])
        bboxes[:, 3] = bbox_y_resize_transform(bboxes[:, 3])

        print("this is bboxes 2:", bboxes)

        bboxes = self.remove_bboxes_outside_image_bounds(bboxes, 448, 448)

        return image, bboxes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: for the sake of efficiency, you should get rid of the annotation
        # and write your custom loader...
        # it will also help simplify the conversion to the 7x7x30 tensor
        image, annotation = super().__getitem__(index)
        image_width, image_height = image.size

        bboxes = self.get_bboxes_from_annotation(annotation)

        print("this is bboxes 0:", bboxes)
        # random_hue
        image = transforms_v2.functional.to_tensor(image)
        # image, bboxes = self.random_affine_transform(image, bboxes)
        image, bboxes = self.random_resize_transform(image, bboxes)

        # then need to convert the bboxes and labels to a tensor of shape (7, 7, 30)
        # assert bboxes.shape[0]  1 and bboxes.shape[1] == 4
        assert len(bboxes.shape) == 2
        return image, bboxes


def create_datasets(root_dir: str) -> tuple[Dataset, Dataset]:
    train_dataset = CustomVOCDataset(root_dir, image_set="train")
    val_dataset = CustomVOCDataset(root_dir, image_set="val")
    # train_dataset = datasets.VOCDetection(
    #     root=root_dir,
    #     year="2012",
    #     image_set="train",
    #     transform=transform,
    #     download=False,
    #     target_transform=target_transform,
    # )

    # val_dataset = datasets.VOCDetection(
    #     root=root_dir,
    #     year="2012",
    #     image_set="val",
    #     transform=transform,
    #     download=False,
    #     target_transform=target_transform,
    # )
    return train_dataset, val_dataset


def create_dataloaders(
    root_dir: str,
    transform: transforms_v2.Compose,
    batch_size: int,
    num_workers: int,
    target_transform: Callable = yolo_target_transform,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(root_dir)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_data_loader, val_data_loader
