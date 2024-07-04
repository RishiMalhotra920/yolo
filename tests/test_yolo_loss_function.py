import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

from src.loss_functions.yolo_loss_function import YOLOLoss


def test_yolo_loss_fn_same_target_and_label(sample_yolo_data_1):
    yolo_loss = YOLOLoss()

    loss = yolo_loss(
        sample_yolo_data_1["yolo_net_target_tensor"],
        sample_yolo_data_1["yolo_net_target_tensor"],
    )

    assert loss == 0.0, "Expected loss to be 0.0"


def test_yolo_loss_fn_two_bounding_boxes_in_one_cell(sample_yolo_data_2):
    yolo_loss = YOLOLoss()

    pred_tensor_list = sample_yolo_data_2["yolo_net_output_tensor"][0, 2, 2, :].numpy()
    target_tensor_list = sample_yolo_data_2["yolo_net_target_tensor"][
        0, 2, 2, :
    ].numpy()

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


def test_yolo_loss_fn_two_bounding_boxes_in_one_cell_swapped_order(sample_yolo_data_2):
    yolo_loss = YOLOLoss()

    pred_tensor_list = sample_yolo_data_2["yolo_net_output_tensor"][0, 2, 2, :].numpy()
    target_tensor_list = sample_yolo_data_2["yolo_net_target_tensor"][
        0, 2, 2, :
    ].numpy()

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

    output_tensor = sample_yolo_data_2["yolo_net_output_tensor"]

    temp = output_tensor[0, 2, 2, :5].clone()
    output_tensor[0, 2, 2, :5] = output_tensor[0, 2, 2, 5:10]
    output_tensor[0, 2, 2, 5:10] = temp

    loss = yolo_loss(
        output_tensor,
        sample_yolo_data_2["yolo_net_target_tensor"],
    )

    assert round(loss.item(), 6) == round(target_loss, 6), "Expected loss to be equal"


def test_yolo_loss_fn_empty_one_prediction_and_no_object(sample_yolo_data_2):
    """
    testing the case where there is no object in the cell and but there is prediction.
    This should result in a loss of prediction confidence squared for the predictor.
    """
    yolo_loss = YOLOLoss()

    pred_tensor_list = sample_yolo_data_2["yolo_net_output_tensor"][0, 2, 2, :].numpy()
    target_tensor_list = np.zeros(30)

    bbox_conf_1 = target_tensor_list[[4, 9]]
    bbox_conf_2 = pred_tensor_list[[4, 9]]

    lambda_noobj = 0.5

    conf_noobj_loss = np.sum((bbox_conf_1 - bbox_conf_2) ** 2)

    target_loss = lambda_noobj * conf_noobj_loss

    loss = yolo_loss(
        sample_yolo_data_2["yolo_net_output_tensor"],
        torch.zeros((1, 7, 7, 30)),
    )

    assert round(loss.item(), 6) == round(target_loss, 6), "Expected loss to be equal"


def test_objects_in_multiple_cells_with_object_loss(sample_yolo_data_3):
    """
    final boss of all the yolo loss function tests.
    two cells:
        batch 0 cell (2, 2): two predictions and two labels
        batch 1 cell (3, 4): two predictions and one label
        batch 0 cell (4, 5): two predictions and zero labels
        many cells: zero predictions and zero labels
    one cell contains two predictions.
    """
    total_expected_loss = 0
    #################### loss for (2, 2)

    pred_tensor_list = sample_yolo_data_3["yolo_net_output_tensor"][0, 2, 2, :].numpy()
    target_tensor_list = sample_yolo_data_3["yolo_net_target_tensor"][
        0, 2, 2, :
    ].numpy()

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

    clf_loss = np.sum((prob_1 - prob_2) ** 2)

    target_loss = lambda_coord * (xy_loss + wh_loss) + conf_loss + clf_loss
    total_expected_loss += target_loss

    ############### loss for (3, 4)

    pred_tensor_list = sample_yolo_data_3["yolo_net_output_tensor"][1, 3, 4, :].numpy()
    target_tensor_list = sample_yolo_data_3["yolo_net_target_tensor"][
        1, 3, 4, :
    ].numpy()

    bbox_xy_1 = target_tensor_list[[0, 1]]
    bbox_xy_2 = pred_tensor_list[[0, 1]]
    bbox_wh_1 = target_tensor_list[[2, 3]]
    bbox_wh_2 = pred_tensor_list[[2, 3]]
    bbox_conf_1 = target_tensor_list[[4]]
    bbox_conf_2 = pred_tensor_list[[4]]
    prob_1 = target_tensor_list[10:]
    prob_2 = pred_tensor_list[10:]

    lambda_coord = 5
    lambda_noobj = 0.5

    xy_loss = np.sum((bbox_xy_1 - bbox_xy_2) ** 2)
    wh_loss = np.sum((np.sqrt(bbox_wh_1) - np.sqrt(bbox_wh_2)) ** 2)
    conf_loss = np.sum((bbox_conf_1 - bbox_conf_2) ** 2)

    bbox_conf_1 = target_tensor_list[[9]]
    bbox_conf_2 = pred_tensor_list[[9]]

    conf_noobj_loss = np.sum((bbox_conf_1 - bbox_conf_2) ** 2)

    clf_loss = np.sum((prob_1 - prob_2) ** 2)

    target_loss = (
        lambda_coord * (xy_loss + wh_loss)
        + conf_loss
        + lambda_noobj * conf_noobj_loss
        + clf_loss
    )
    total_expected_loss += target_loss

    ############### loss for (4, 5)

    pred_tensor_list = sample_yolo_data_3["yolo_net_output_tensor"][0, 4, 5, :].numpy()
    target_tensor_list = sample_yolo_data_3["yolo_net_target_tensor"][
        0, 4, 5, :
    ].numpy()

    bbox_conf_1 = target_tensor_list[[4, 9]]
    bbox_conf_2 = pred_tensor_list[[4, 9]]

    lambda_noobj = 0.5
    conf_noobj_loss = np.sum((bbox_conf_1 - bbox_conf_2) ** 2)

    target_loss = lambda_noobj * conf_noobj_loss
    total_expected_loss += target_loss

    ### expected loss for the entire thing

    yolo_loss = YOLOLoss()

    loss = yolo_loss(
        sample_yolo_data_3["yolo_net_output_tensor"],
        sample_yolo_data_3["yolo_net_target_tensor"],
    )

    assert round(loss.item(), 4) == round(
        total_expected_loss, 4
    ), "Expected loss to be equal"
