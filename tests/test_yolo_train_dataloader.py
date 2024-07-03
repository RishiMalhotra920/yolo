import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402


from src.data_setup.yolo_train_data_setup import yolo_target_transform


def test_yolo_target_transform(sample_yolo_data_1):
    target = yolo_target_transform(sample_yolo_data_1["annotation"])

    assert torch.equal(
        target, sample_yolo_data_1["yolo_net_target_tensor"].squeeze()
    ), "Expected target"
