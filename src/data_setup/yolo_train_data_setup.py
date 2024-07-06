from typing import Callable

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2 as transforms_v2

from src.utils import class_to_index

config = yaml.safe_load(open("config.yaml"))


# data should be as the follows:


def yolo_target_transform(
    bounding_boxes: tv_tensors, labels: list[int], image_width: int, image_height: int
) -> torch.Tensor:
    """
    labels[..., 2] and labels[..., 7] are the width of bbox 1 and bbox 2
    if a grid cell has no labels, then labels[..., 2] and labels[..., 7] will be 0
    if a grid cell has only one label, then labels[..., 2] will be 0

    converts annotation to a tensor of shape (7, 7, 30)
    """
    # a = {'annotation': {'folder': 'VOC2012', 'filename': '2008_000008.jpg', 'source': {'database': 'The VOC2008 Database', 'annotation': 'PASCAL VOC2008', 'image': 'flickr'}, 'size': {'width': '500', 'height': '442', 'depth': '3'}, 'segmented': '0', 'object': [
    # {'name': 'horse', 'pose': 'Left', 'truncated': '0', 'occluded': '1', 'bndbox': {'xmin': '53', 'ymin': '87', 'xmax': '471', 'ymax': '420'}, 'difficult': '0'}, {'name': 'person', 'pose': 'Unspecified', 'truncated': '1', 'occluded': '0', 'bndbox': {'xmin': '158', 'ymin': '44', 'xmax': '289', 'ymax': '167'}, 'difficult': '0'}]}}

    # Convert the annotations to a four tuple of tensors
    # objects = annotation["annotation"]["object"]
    # image_width = int(annotation["annotation"]["size"]["width"])
    # image_height = int(annotation["annotation"]["size"]["height"])

    grid_size = 7

    # grid_cells = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    B = 2
    C = 20
    depth = B * 5 + C
    grid_cells = torch.zeros((grid_size, grid_size, depth))

    grid_cell_to_class = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
    grid_cell_to_num_bboxes = [[0 for _ in range(grid_size)] for _ in range(grid_size)]


    # refer to shape of object_indexes as oi

    bounding_boxes: torch.tensor = torch.tensor(bounding_boxes.data)  # type: ignore
    # convert to center x, center y, width, height, confidence
    x_min, y_min, x_max, y_max = (
        bounding_boxes[:, 0],
        bounding_boxes[:, 1],
        bounding_boxes[:, 2],
        bounding_boxes[:, 3],
    )
    x_center = (x_min + x_max) / 2 # (oi)
    y_center = (y_min + y_max) / 2 # (oi)

    # normalize w, h by image width and height
    width_adjusted = (x_max - x_min) / image_width # (oi) 
    height_adjusted = (y_max - y_min) / image_height # (oi)

    # convert x, y to cell offset
    x_relative_to_grid = (x_center / image_width) * grid_size # (oi)
    y_relative_to_grid = (y_center / image_height) * grid_size # (oi)

    grid_x = x_relative_to_grid.int() # (oi)
    grid_y = y_relative_to_grid.int() # (oi)

    offset_x = x_relative_to_grid - int(x_relative_to_grid) # n
    offset_y = y_relative_to_grid - int(y_relative_to_grid) # n

    # take any 2 bounding boxes
    result = torch.zeros((7, 7, 30))

    result[grid_x, grid_y, 0] = offset_x
    result[grid_x, grid_y, 1] = offset_y
    result[grid_x, grid_y, 2] = width_adjusted
    result[grid_x, grid_y, 3] = height_adjusted
    result[grid_x, grid_y, 4] = 1.0

    object_1_index = 0

    # every grid cell must have two bounding box labels at most
    # 

    result[grid_x, grid_y, 2] =
    try:
        object_2_index = labels.index(labels[object_1_index], 1)
        object_indexes = [object_1_index, object_2_index]
    except ValueError: # there is only one object with the current label
        object_indexes = [object_1_index]
    


    # result[grid_x, grid_y, 5:] = torch.tensor([

    

    # create 7x7x30 tensor and populate with bounding_boxes



    bounding_boxes is of the format: [x_relative_to_grid, y_relative_to_grid, w, h, confidence]

    # you actually don't need to scale anything to the 448*448 dimension space.
    for bbox in bounding_boxes:
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


# def RandomAffine


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

    def __getitem__(self, index):
        image, annotation = super().__getitem__(index)

        # transformed_image,  = self.transform(image)

        objects = annotation["annotation"]["object"]
        image_width = int(annotation["annotation"]["size"]["width"])
        image_height = int(annotation["annotation"]["size"]["height"])

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

        boxes = tv_tensors.BoundingBoxes(
            data=boxes, format="XYXY", canvas_size=(image_height, image_width)
        )  # type: ignore missing type annotations here.

        out_image, out_boxes = self.transform(image, boxes)

        out_target = yolo_target_transform(out_boxes, labels, image_width, image_height)

        return out_image, out_target

        # boxes = tv_tensors.BoundingBoxes(
        #     data=[[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
        #     format="XYXY",
        #     canvas_size=(image_height, image_width),
        # )

        # for obje

        # # pass the image into torchvision transform v2, pass the bboxes into yolo_target_transform
        # # return the transformed image and the transformed bboxes

        # for object in annotation


def create_datasets(
    root_dir: str, transform: transforms_v2.Compose, target_transform: Callable | None
) -> tuple[Dataset, Dataset]:
    train_dataset = CustomVOCDetection(
        root=root_dir,
        year="2012",
        image_set="train",
        transform=transform,
        # download=False,
        # target_transform=target_transform,
    )

    val_dataset = datasets.VOCDetection(
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
    target_transform: Callable = yolo_target_transform,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(root_dir, transform, target_transform)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_data_loader, val_data_loader
