import torch

from src.loss_functions.yolo_loss_function import calculate_iou


def test_calculate_iou(sample_yolo_data_2):
    pred_box_1 = sample_yolo_data_2["yolo_net_output_tensor"][:, :, :, :4]
    target_box_1 = sample_yolo_data_2["yolo_net_target_tensor"][:, :, :, :4]

    bbox_1 = target_box_1[0, 2, 2, :4].numpy()
    bbox_2 = pred_box_1[0, 2, 2, :4].numpy()

    bbox_1_x_left = bbox_1[0] - bbox_1[2] / 2
    bbox_1_y_top = bbox_1[1] - bbox_1[3] / 2
    bbox_1_x_right = bbox_1[0] + bbox_1[2] / 2
    bbox_1_y_bottom = bbox_1[1] + bbox_1[3] / 2

    bbox_2_x_left = bbox_2[0] - bbox_2[2] / 2
    bbox_2_y_top = bbox_2[1] - bbox_2[3] / 2
    bbox_2_x_right = bbox_2[0] + bbox_2[2] / 2
    bbox_2_y_bottom = bbox_2[1] + bbox_2[3] / 2

    x_left = max(bbox_1_x_left, bbox_2_x_left)
    y_top = max(bbox_1_y_top, bbox_2_y_top)
    x_right = min(bbox_1_x_right, bbox_2_x_right)
    y_bottom = min(bbox_1_y_bottom, bbox_2_y_bottom)

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    bbox_1_area = bbox_1[2] * bbox_1[3]
    bbox_2_area = bbox_2[2] * bbox_2[3]

    union_area = bbox_1_area + bbox_2_area - intersection_area

    expected_iou = intersection_area / union_area

    pred_iou = calculate_iou(pred_box_1, target_box_1)

    assert round(pred_iou[0, 2, 2].item(), 4) == round(
        expected_iou, 4
    ), "Expected iou to be equal"


def test_calculate_iou_no_bboxes(sample_yolo_data_2):
    pred_box_1 = sample_yolo_data_2["yolo_net_output_tensor"][:, :, :, :4]
    target_box_1 = sample_yolo_data_2["yolo_net_target_tensor"][:, :, :, :4]

    pred_iou = calculate_iou(pred_box_1, target_box_1)

    assert pred_iou[0, 0, 0] == 0, "Expected no bbox iou to be 0"


def test_calculate_iou_pred_bbox_but_no_label_bbox(sample_yolo_data_2):
    pred_box_1 = sample_yolo_data_2["yolo_net_output_tensor"][:, :, :, :4]
    pred_iou = calculate_iou(pred_box_1, torch.zeros_like(pred_box_1))
    assert pred_iou[0, 2, 2].item() == 0, "Expected no bbox iou to be 0"


# test the visualization function that takes in the target tensor and the output tensor and visualizes the bounding boxes and the class labels.
