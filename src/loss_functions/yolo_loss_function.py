import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torchvision.datasets import VOCDetection


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

    intersection = torch.max(br_x - tl_x, torch.tensor(0)) * \
        torch.max(br_y - tl_y, torch.tensor(0))

    width1 = tensor1[:, :, :, 2]
    height1 = tensor1[:, :, :, 3]
    width2 = tensor2[:, :, :, 2]
    height2 = tensor2[:, :, :, 3]

    union = width1 * height1 + width2 * height2 - intersection

    return intersection / union

    # given


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):

        bs = preds.shape[0]  # batch size
        B = 2
        C = 20
        S = 7

        # TODO; maybe you can initialize these as zero tensors and then assign them.
        ious_per_cell = torch.zeros((bs, S, S, B*B))
        bbox_losses = torch.zeros((bs, S, S, B*B))
        wh_losses = torch.zeros((bs, S, S, B*B))
        conf_losses = torch.zeros((bs, S, S, B*B))

        for pred_idx in range(B):
            for label_idx in range(B):

                this_pred_bbox = preds[:, :, :, pred_idx * 5:pred_idx * 5 + 5]
                this_label_bbox = labels[:, :, :,
                                         label_idx * 5:label_idx * 5 + 5]
                this_bbox_loss = F.mse_loss(
                    this_pred_bbox, this_label_bbox, reduction='none')
                this_iou = calculate_iou(
                    this_pred_bbox, this_label_bbox)  # (bs, S, S)

                this_pred_w = this_pred_bbox[:, :, :, 2]
                this_pred_h = this_pred_bbox[:, :, :, 3]
                this_label_w = this_label_bbox[:, :, :, 2]
                this_label_h = this_label_bbox[:, :, :, 3]

                this_wh_loss = F.mse_loss(this_pred_w, this_label_w, reduction='none') + F.mse_loss(
                    this_pred_h, this_label_h, reduction='none')  # (bs, S, S)
                # originally 4:5. (batch_size, S, S)
                this_conf_loss = F.mse_loss(
                    this_pred_bbox[:, :, :, 4], this_label_bbox[:, :, :, 4], reduction='none')

                ious_per_cell[:, :, pred_idx * B + label_idx] = this_iou
                bbox_losses[:, :, pred_idx * B + label_idx] = this_bbox_loss
                wh_losses[:, :, pred_idx * B + label_idx] = this_wh_loss
                conf_losses[:, :, pred_idx * B + label_idx] = this_conf_loss

                # remember to multiply by lambda_noobj if the object is not present in the cell.

        argmax_iou_per_cell = torch.argmax(ious_per_cell, dim=3)  # (bs, S, S)

        # one-hot vector represents which predictor is responsible for predicting the object.
        argmax_iou_per_cell_onehot = torch.nn.functional.one_hot(
            argmax_iou_per_cell, num_classes=B*B)  # (bs, S, S, B*B)

        # multiply by one hot vector to get only the loss for the predictor responsible for the object.
        bbox_loss = torch.mean(argmax_iou_per_cell_onehot*bbox_losses)
        wh_loss = torch.mean(argmax_iou_per_cell_onehot*wh_losses)
        conf_loss = torch.mean(argmax_iou_per_cell_onehot*conf_losses)
        conf_noobj_loss = torch.mean(
            (1 - argmax_iou_per_cell_onehot) * conf_losses)

        clf_loss = F.mse_loss(preds[:, :, :, B*5:], labels[:, :, :, B*5:])

        loss = self.lambda_coord * (bbox_loss + wh_loss) + conf_loss + \
            self.lambda_noobj * clf_loss + conf_noobj_loss

        return loss
