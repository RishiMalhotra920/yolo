import torch
from torch import nn


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, inputs, targets):
        lambda_coord = 5
        lambda_noobj = 0.5
        num_anchors = 2
        num_classes = 20
        batch_size = inputs.shape[0]
        grid_size = inputs.shape[2]
        stride = 1 / grid_size

        # Split the inputs into bounding box, objectness and class predictions
        bounding_box_predictions = inputs[:, :num_anchors * 5, :, :]

        objectness_predictions = inputs[:, num_anchors *
                                        5:num_anchors * 5 + num_anchors, :, :]

        class_predictions = inputs[:, num_anchors * 5 + num_anchors:, :, :]

        # Split the targets into bounding box, objectness and class targets
        bounding_box_targets = targets[:, :num_anchors * 5, :, :]
        objectness_targets = targets[:, num_anchors *
                                     5:num_anchors * 5 + num_anchors, :, :]
        class_targets = targets[:, num_anchors * 5 + num_anchors:, :, :]

        # Calculate the loss for the bounding box predictions
        bounding_box_predictions = bounding_box_predictions.view(
            batch_size, num_anchors, 5, grid_size, grid_size)
        bounding_box_targets = bounding_box_targets.view(
            batch_size, num_anchors, 5, grid_size, grid_size)
        bounding_box_predictions = bounding_box_predictions.permute(
            0, 1, 3, 4, 2)
        bounding_box_targets = bounding_box_targets.permute(0, 1, 3, 4, 2)
        bounding_box_predictions_xy = torch.sigmoid(
            bounding_box_predictions[:, :, :, :, :2])
        bounding_box_predictions_wh = torch.exp(
            bounding_box_predictions[:, :, :, :, 2:4])
        bounding_box_targets_xy = bounding_box_targets[:, :, :, :, :2]
        bounding_box_targets_wh = bounding_box_targets[:, :, :, :, 2:4]
        bounding_box_predictions_confidence = torch.sigmoid(
            bounding_box_predictions[:, :, :, :, 4:5])
        bounding_box_targets_confidence = bounding_box_targets[:, :, :, :, 4:5]
        bounding_box_targets_confidence = bounding_box_targets_confidence.unsqueeze(
            -1)
        bounding_box_predictions_xy = bounding_box_predictions_xy * stride + \
            torch.arange(grid_size).view(1, 1, -1, 1).float() * stride
        bounding_box_predictions_wh = bounding_box_predictions_wh * stride
        bounding_box_targets_xy = bounding_box_targets_xy * stride + \
            torch.arange(grid_size).view(1, 1, -1, 1).float() * stride
        bounding_box_targets_wh = bounding_box_targets_wh * stride
        bounding_box_predictions_xy_min = bounding_box_predictions_xy - \
            bounding_box_predictions_wh / 2
        bounding_box_predictions_xy_max = bounding_box_predictions_xy + \
            bounding_box_predictions_wh / 2
        bounding_box_targets_xy_min = bounding_box_targets_xy - bounding_box_targets_wh / 2
        bounding_box_targets_xy_max = bounding_box_targets_xy + bounding_box_targets_wh / 2
        bounding_box_predictions_area = bounding_box_predictions_wh.prod(
            dim=-1)
        bounding_box_targets_area = bounding_box_targets_wh.prod(dim=-1)
        bounding_box_predictions_intersection_min = torch.max(
            bounding_box_predictions_xy_min, bounding_box_targets_xy_min)
        bounding_box_predictions_intersection_max = torch.min(
            bounding_box_predictions_xy_max, bounding_box_targets_xy_max)
        bounding_box_predictions_intersection_wh = torch.clamp(
            bounding_box_predictions_intersection_max - bounding_box_predictions_intersection_min, min=0)
        bounding_box_predictions_intersection_area = bounding_box_predictions_intersection_wh.prod(
            dim=-1)
        bounding_box_predictions_union_area = bounding_box_predictions_area + \
            bounding_box_targets_area - bounding_box_predictions_intersection_area
        bounding_box_predictions_iou = bounding_box_predictions_intersection_area / \
            bounding_box_predictions_union_area
        bounding_box_targets_iou = bounding_box_targets_confidence
        bounding_box_loss = torch.sum(
            (bounding_box_predictions_iou - bounding_box_targets_iou) ** 2)
        bounding_box_loss = bounding_box_loss / batch_size

        # Calculate the loss for the objectness predictions
        objectness_predictions = objectness_predictions.view(
            batch_size, num_anchors, grid_size, grid_size)
        objectness_targets = objectness_targets.view(
            batch_size, num_anchors, grid_size, grid_size)
        objectness_predictions = objectness_predictions.unsqueeze(-1)
        objectness_targets = objectness_targets.unsqueeze(-1)
        objectness_loss = torch.sum(
            (objectness_predictions - objectness_targets) ** 2)
        objectness_loss = objectness_loss / batch_size

        # Calculate the loss for the class predictions
        class_predictions = class_predictions
        class_targets = class_targets
        class_loss = torch.nn.functional.cross_entropy(
            class_predictions, class_targets)
        class_loss = class_loss / batch_size

        # Calculate the total loss
        total_loss = lambda_coord * bounding_box_loss + \
            lambda_noobj * objectness_loss + class_loss

        return total_loss
