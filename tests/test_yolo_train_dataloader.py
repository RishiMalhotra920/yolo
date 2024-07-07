import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402


from src.data_setup.yolo_train_data_setup import categorize_bboxes_into_grid

# disable this test for now


def test_categorize_bboxes_into_grid_computed_data(sample_yolo_data_1):
    # target = yolo_target_transform(sample_yolo_data_1["annotation"])
    output = categorize_bboxes_into_grid(
        sample_yolo_data_1["bboxes"],
        sample_yolo_data_1["labels"],
        sample_yolo_data_1["width"],
        sample_yolo_data_1["height"],
    )

    print("output.shape", output.shape)

    assert torch.equal(
        output, sample_yolo_data_1["yolo_net_target_tensor"].squeeze()
    ), "Expected target"


def test_categorize_bboxes_into_grid_sample_data(sample_yolo_bboxes_1):
    output = categorize_bboxes_into_grid(
        sample_yolo_bboxes_1["bboxes"],
        sample_yolo_bboxes_1["labels"],
        sample_yolo_bboxes_1["width"],
        sample_yolo_bboxes_1["height"],
    )

    assert torch.equal(
        output, sample_yolo_bboxes_1["yolo_net_target_tensor"].squeeze()
    ), "Expected target"
