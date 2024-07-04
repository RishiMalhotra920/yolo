import os
import sys
from typing import Any

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

from src.utils import class_to_index


@pytest.fixture
def sample_yolo_data_1() -> dict[str, Any]:
    """ """
    sample_annotation = {
        "annotation": {
            "folder": "VOC2012",
            # "filename": "2008_000008.jpg",
            "source": {
                "database": "The VOC2008 Database",
                "annotation": "PASCAL VOC2008",
                "image": "flickr",
            },
            "size": {"width": "1344", "height": "896", "depth": "3"},
            "segmented": "0",
            "object": [
                {
                    "name": "horse",
                    "pose": "Left",
                    "truncated": "0",
                    "occluded": "1",
                    "bndbox": {
                        "xmin": "105",
                        "ymin": "84",
                        "xmax": "798",
                        "ymax": "504",
                    },
                    "difficult": "0",
                },
                {
                    "name": "horse",
                    "pose": "Left",
                    "truncated": "0",
                    "occluded": "1",
                    "bndbox": {
                        "xmin": "110",
                        "ymin": "94",
                        "xmax": "798",
                        "ymax": "504",
                    },
                    "difficult": "0",
                },
            ],
        }
    }

    tensor = torch.zeros(1, 7, 7, 30)

    one_hot = [0.0 for _ in range(20)]
    one_hot[class_to_index["horse"]] = 1

    cell_tensor = [
        0.3515625,
        0.296875,
        0.515625,
        0.46875,
        1.0,
        0.3645833432674408,
        0.3359375,
        0.511904776096344,
        0.4575892984867096,
        1.0,
    ] + one_hot

    tensor[0, 2, 2, :] = torch.tensor(cell_tensor)

    result = {"annotation": sample_annotation, "yolo_net_target_tensor": tensor}
    return result


@pytest.fixture
def sample_yolo_data_2() -> dict[str, Any]:
    """
    slight offset in the bounding box and the class probabilities
    """

    # target tensor
    cell_tensor_bboxes = [
        0.3515625,
        0.296875,
        0.515625,
        0.46875,
        1.0,
        0.3645833432674408,
        0.3359375,
        0.511904776096344,
        0.4575892984867096,
        1.0,
    ]

    one_hot = [0.0 for _ in range(20)]
    one_hot[class_to_index["horse"]] = 1

    cell_target_tensor = cell_tensor_bboxes + one_hot

    target_tensor = torch.zeros(1, 7, 7, 30)
    target_tensor[0, 2, 2, :] = torch.tensor(cell_target_tensor)

    # output tensor

    offset_bbox1 = [0.001, 0.002, 0.001, 0.002, -0.3]
    offset_bbox2 = [0.002, 0.002, 0.001, 0.002, -0.2]
    offset_bbox = offset_bbox1 + offset_bbox2

    prob_vec = [0.19 / 19 for _ in range(20)]
    prob_vec[class_to_index["horse"]] = 0.81

    cell_tensor_bboxes_with_offset = [
        x + y for x, y in zip(cell_tensor_bboxes, offset_bbox)
    ]
    cell_output_tensor = cell_tensor_bboxes_with_offset + prob_vec

    output_tensor = torch.zeros(1, 7, 7, 30)
    output_tensor[0, 2, 2, :] = torch.tensor(cell_output_tensor)

    result = {
        "annotation": None,
        "yolo_net_target_tensor": target_tensor,
        "yolo_net_output_tensor": output_tensor,
    }

    return result


@pytest.fixture
def sample_yolo_data_3() -> dict[str, Any]:
    """
    cell (2, 2): two predictions and two labels
    batch 1 cell (3, 4): two predictions and one label
    cell (4, 5): two predictions and zero labels
    many cells: zero predictions and zero labels
    """

    # target tensor
    cell_tensor_bboxes = [
        0.3515625,
        0.296875,
        0.515625,
        0.46875,
        1.0,
        0.3645833432674408,
        0.3359375,
        0.511904776096344,
        0.4575892984867096,
        1.0,
    ]

    one_hot = [0.0 for _ in range(20)]
    one_hot[class_to_index["horse"]] = 1

    cell_target_tensor = cell_tensor_bboxes + one_hot

    target_tensor = torch.zeros(2, 7, 7, 30)
    target_tensor[0, 2, 2, :] = torch.tensor(cell_target_tensor)
    target_tensor[1, 3, 4, :] = torch.tensor(cell_target_tensor)
    target_tensor[1, 3, 4, 5:10] = 0

    # output tensor

    offset_bbox1 = [0.001, 0.002, 0.001, 0.002, -0.3]
    offset_bbox2 = [0.002, 0.002, 0.001, 0.002, -0.2]
    offset_bbox = offset_bbox1 + offset_bbox2

    prob_vec = [0.19 / 19 for _ in range(20)]
    prob_vec[class_to_index["horse"]] = 0.81

    cell_tensor_bboxes_with_offset = [
        x + y for x, y in zip(cell_tensor_bboxes, offset_bbox)
    ]
    cell_output_tensor = cell_tensor_bboxes_with_offset + prob_vec

    output_tensor = torch.zeros(2, 7, 7, 30)
    output_tensor[0, 2, 2, :] = torch.tensor(cell_output_tensor)
    output_tensor[1, 3, 4, :] = torch.tensor(cell_output_tensor)
    output_tensor[0, 4, 5, :] = torch.tensor(cell_output_tensor)

    result = {
        "annotation": None,
        "yolo_net_target_tensor": target_tensor,
        "yolo_net_output_tensor": output_tensor,
    }

    return result
