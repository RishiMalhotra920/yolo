import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

from src.loss_functions.yolo_loss_function import YOLOLoss


def test_yolo_loss_fn(sample_yolo_data_1):
    yolo_loss = YOLOLoss()

    loss = yolo_loss(
        sample_yolo_data_1["yolo_net_target_tensor"],
        sample_yolo_data_1["yolo_net_target_tensor"],
    )

    assert loss == 0.0, "Expected loss to be 0.0"


def test_yolo_loss_fn2(sample_yolo_data_2):
    yolo_loss = YOLOLoss()

    target_tensor_list = sample_yolo_data_2["yolo_net_target_tensor"][
        0, 2, 2, :
    ].numpy()
    pred_tensor_list = sample_yolo_data_2["yolo_net_output_tensor"][0, 2, 2, :].numpy()

    bbox_xy_1 = target_tensor_list[[0, 1, 5, 6]]
    bbox_xy_2 = pred_tensor_list[[0, 1, 5, 6]]
    bbox_wh_1 = target_tensor_list[[2, 3, 7, 8]]
    bbox_wh_2 = pred_tensor_list[[2, 3, 7, 8]]
    bbox_conf_1 = target_tensor_list[[4, 9]]
    bbox_conf_2 = pred_tensor_list[[4, 9]]
    prob_1 = target_tensor_list[10:]
    prob_2 = pred_tensor_list[10:]

    lambda_coord = 5
    lambda_noobj = 0.5

    xy_loss = np.sum((bbox_xy_1 - bbox_xy_2) ** 2)
    wh_loss = np.sum((np.sqrt(bbox_wh_1) - np.sqrt(bbox_wh_2)) ** 2)
    conf_loss = np.sum((bbox_conf_1 - bbox_conf_2) ** 2)
    conf_noobj_loss = 0
    clf_loss = np.sum((prob_1 - prob_2) ** 2)

    target_loss = (
        lambda_coord * (xy_loss + wh_loss)
        + conf_loss
        + lambda_noobj * conf_noobj_loss
        + clf_loss
    )

    print(
        "expected xy_loss: ",
        xy_loss,
        "wh_loss: ",
        wh_loss,
        "conf_loss: ",
        conf_loss,
        "conf_noobj_loss: ",
        conf_noobj_loss,
        "clf_loss: ",
        clf_loss,
    )

    loss = yolo_loss(
        sample_yolo_data_2["yolo_net_output_tensor"],
        sample_yolo_data_2["yolo_net_target_tensor"],
    )

    assert round(loss.item(), 6) == round(target_loss, 6), "Expected loss to be equal"


# TODO: swap the order of the bounding box and label predictions and see if the tests still hold.
