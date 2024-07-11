import torch


def calculate_iou(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    tensor1 and tensor2 are of shape (batch_size, S, S, 4)

    the last dimension of the tensor represents x_center, y_center, width, height

    calculate the intersection over each cell of the grid for tensor1 and tensor2

    Returns:
        iou: torch.Tensor of shape (batch_size, S, S)
    """
    # return torch.tensor([2, 3])

    tl_x1 = tensor1[:, :, :, 0] - tensor1[:, :, :, 2] / 2
    tl_y1 = tensor1[:, :, :, 1] - tensor1[:, :, :, 3] / 2
    br_x1 = tensor1[:, :, :, 0] + tensor1[:, :, :, 2] / 2
    br_y1 = tensor1[:, :, :, 1] + tensor1[:, :, :, 3] / 2

    tl_x2 = tensor2[:, :, :, 0] - tensor2[:, :, :, 2] / 2
    tl_y2 = tensor2[:, :, :, 1] - tensor2[:, :, :, 3] / 2
    br_x2 = tensor2[:, :, :, 0] + tensor2[:, :, :, 2] / 2
    br_y2 = tensor2[:, :, :, 1] + tensor2[:, :, :, 3] / 2

    tl_x = torch.max(tl_x1, tl_x2)
    tl_y = torch.max(tl_y1, tl_y2)
    br_x = torch.min(br_x1, br_x2)
    br_y = torch.min(br_y1, br_y2)

    intersection = torch.max(br_x - tl_x, torch.tensor(0)) * torch.max(
        br_y - tl_y, torch.tensor(0)
    )

    width1 = tensor1[:, :, :, 2]
    height1 = tensor1[:, :, :, 3]
    width2 = tensor2[:, :, :, 2]
    height2 = tensor2[:, :, :, 3]

    union = width1 * height1 + width2 * height2 - intersection

    iou = intersection / union

    # filter out nans. we define iou as 0 if there's no bboxes.
    # this works well because we intend to calculate the argmax over the iou values.
    # so if there's no bbox, the iou will be 0 and this won't be selected by the argmax
    iou[torch.isnan(iou)] = 0

    return iou


def get_yolo_metrics(pred: torch.Tensor, label: torch.Tensor) -> dict:
    """Count the number of incorrect background predictions.

    Args:
        pred (torch.Tensor): The logits or probabilities from the model. Shape: (bs, 7, 7, 30)
        label (torch.Tensor): The true labels for each input. Shape: (bs, 7, 7, 30)

    Returns:
        int: The number of incorrect background predictions.
    """
    B = 2

    does_label_1_exist_for_each_cell = label[..., 4] == 1.0
    does_label_2_exist_for_each_cell = label[..., 9] == 1.0

    # 0, 1, 2, 3 - bbox 1. 4 - bbox 1 confidence
    iou_pred_1_and_label_1 = calculate_iou(pred[..., :4], label[..., :4])  # (bs, 7, 7)

    # 5, 6, 7, 8 - bbox 2. 9 - bbox 2 confidence
    iou_pred_2_and_label_2 = calculate_iou(
        pred[..., 5:9], label[..., 5:9]
    )  # (bs, 7, 7)
    # Check if the predicted class labels are correct

    preds_argmax = torch.argmax(pred[..., B * 5 :], dim=-1)
    labels_argmax = torch.argmax(label[..., B * 5 :], dim=-1)

    is_class_correct = preds_argmax == labels_argmax  # (bs, 7, 7)

    num_correct_for_pred_box_1 = (
        (
            does_label_1_exist_for_each_cell
            & (iou_pred_1_and_label_1 > 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_correct_for_pred_box_2 = (
        (
            does_label_2_exist_for_each_cell
            & (iou_pred_2_and_label_2 > 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_localization_for_pred_box_1 = (
        (
            does_label_1_exist_for_each_cell
            & (0.1 <= iou_pred_1_and_label_1)
            & (iou_pred_1_and_label_1 <= 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_localization_for_pred_box_2 = (
        (
            does_label_2_exist_for_each_cell
            & (0.1 <= iou_pred_2_and_label_2)
            & (iou_pred_2_and_label_2 <= 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_other_for_pred_box_1 = (
        (
            does_label_1_exist_for_each_cell
            & (iou_pred_1_and_label_1 > 0.1)
            & ~is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_other_for_pred_box_2 = (
        (
            does_label_2_exist_for_each_cell
            & (iou_pred_2_and_label_2 > 0.1)
            & ~is_class_correct
        )
        .sum()
        .item()
    )

    # incorrectly classified object as background

    num_incorrect_background_for_pred_box_1 = (
        (does_label_1_exist_for_each_cell & (iou_pred_1_and_label_1 <= 0.1))
        .sum()
        .item()
    )

    num_incorrect_background_for_pred_box_2 = (
        (does_label_2_exist_for_each_cell & (iou_pred_2_and_label_2 <= 0.1))
        .sum()
        .item()
    )
    num_objects = (
        does_label_1_exist_for_each_cell.sum().item()
        + does_label_2_exist_for_each_cell.sum().item()
    )

    return {
        "num_correct": num_correct_for_pred_box_1 + num_correct_for_pred_box_2,
        "num_incorrect_localization": num_incorrect_localization_for_pred_box_1
        + num_incorrect_localization_for_pred_box_2,
        "num_incorrect_other": num_incorrect_other_for_pred_box_1
        + num_incorrect_other_for_pred_box_2,
        "num_incorrect_background": num_incorrect_background_for_pred_box_1
        + num_incorrect_background_for_pred_box_2,
        "num_objects": num_objects,
    }
