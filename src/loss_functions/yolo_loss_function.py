import torch
import torch.nn.functional as F
from torch import nn

from src.utils import calculate_iou


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        preds and labels are of shape (batch_size, S, S, B * 5 + C)

        assuming each grid cell produces two bounding boxes and each target has two bounding boxes,
        for simplicity, we only calculate the loss for the bbox with the highest ious.
        """
        bs = preds.shape[0]  # batch size
        B = 2
        # C = 20
        S = 7

        # TODO; maybe you can initialize these as zero tensors and then assign them.
        ious_per_cell = torch.zeros((bs, S, S, B * B))
        xy_losses = torch.zeros((bs, S, S, B * B))
        wh_losses = torch.zeros((bs, S, S, B * B))
        conf_losses = torch.zeros((bs, S, S, B * B))

        # try to take a top down approach and see if you
        for pred_idx in range(B):
            for label_idx in range(B):
                this_pred_idx = pred_idx * 5
                this_label_idx = label_idx * 5

                this_pred_x = preds[:, :, :, this_pred_idx]
                this_pred_y = preds[:, :, :, this_pred_idx + 1]
                this_pred_w = preds[:, :, :, this_pred_idx + 2]
                this_pred_h = preds[:, :, :, this_pred_idx + 3]
                this_pred_conf = preds[:, :, :, this_pred_idx + 4]

                this_label_x = labels[:, :, :, this_label_idx]
                this_label_y = labels[:, :, :, this_label_idx + 1]
                this_label_w = labels[:, :, :, this_label_idx + 2]
                this_label_h = labels[:, :, :, this_label_idx + 3]
                this_label_conf = labels[:, :, :, this_label_idx + 4]

                # print("this_pred_x", this_pred_x, "this_label_x", this_label_x)

                this_xy_loss = F.mse_loss(
                    this_pred_x, this_label_x, reduction="none"
                ) + F.mse_loss(this_pred_y, this_label_y, reduction="none")
                this_wh_loss = F.mse_loss(
                    torch.sqrt(this_pred_w), torch.sqrt(this_label_w), reduction="none"
                ) + F.mse_loss(
                    torch.sqrt(this_pred_h), torch.sqrt(this_label_h), reduction="none"
                )
                this_conf_loss = F.mse_loss(
                    this_pred_conf, this_label_conf, reduction="none"
                )

                this_pred_bbox = preds[:, :, :, this_pred_idx : this_pred_idx + 4]
                this_label_bbox = labels[
                    :, :, :, this_label_idx : this_label_idx * 5 + 4
                ]

                this_iou = calculate_iou(this_pred_bbox, this_label_bbox)  # (bs, S, S)

                ious_per_cell[:, :, :, pred_idx * B + label_idx] = (
                    this_iou  # (bs, S, S)
                )
                xy_losses[:, :, :, pred_idx * B + label_idx] = this_xy_loss
                wh_losses[:, :, :, pred_idx * B + label_idx] = this_wh_loss
                conf_losses[:, :, :, pred_idx * B + label_idx] = this_conf_loss

                # remember to multiply by lambda_noobj if the object is not present in the cell.

        # we will create two matches between labels and predictions
        # if argmax is 0 or 3, then label 0 responsible for pred 0, and label 1 responsible for pred 1.
        # if argmax is 1 or 2, then label 1 responsible for pred 0, and label 0 responsible for pred 1.

        match_one_per_cell = torch.argmax(ious_per_cell, dim=3)  # (bs, S, S)
        match_two_per_cell = ((torch.zeros((bs, S, S)) + 3) - match_one_per_cell).long()

        # one-hot vector represents which predictor is responsible for predicting the object.
        match_one_per_cell_onehot = torch.nn.functional.one_hot(
            match_one_per_cell, num_classes=B * B
        )  # (bs, S, S, B*B)

        match_two_per_cell_onehot = torch.nn.functional.one_hot(
            match_two_per_cell, num_classes=B * B
        )  # (bs, S, S, B*B)

        # we unsqueeze to help with broadcasting
        obj_at_label_one_mask = (labels[:, :, :, 4] == 1).unsqueeze(-1)  # (bs, S, S, 1)
        obj_at_label_two_mask = (labels[:, :, :, 9] == 1).unsqueeze(-1)  # (bs, S, S, 1)

        # this mask does two things:
        # 1. figures out which xy losses to consider for the loss calculation.
        # eg: a [1, 0, 0, 1] at pos [bs, S, S] means consider xywh loss for pred 0-label 0 and pred 1-label 1 in the cell [bs, S, S].
        # a [0, 1, 1, 0] at pos [bs, S, S] means consider xywh loss for pred 1-label 0 and pred 0-label 1 in the cell [bs, S, S].
        # 2. only considers the xywh conf losses if the object is present in the cell.

        matches_and_one_obj_ij_mask = (
            obj_at_label_one_mask * match_one_per_cell_onehot
            + obj_at_label_two_mask * match_two_per_cell_onehot
        )  # (bs, S, S, B*B)
        print("matches_per_cell_twohot", matches_and_one_obj_ij_mask[0, 2, 2])

        # multiply by one hot vector to get only the loss for the predictor responsible for the object.
        print("trace 1 sum of xy losses", xy_losses[0, 2, 2, 1], xy_losses[0, 2, 2, 2])
        xy_loss = torch.sum(matches_and_one_obj_ij_mask * xy_losses)
        wh_loss = torch.sum(matches_and_one_obj_ij_mask * wh_losses)
        conf_loss = torch.sum(matches_and_one_obj_ij_mask * conf_losses)

        # if no object is present at label 0, then the confidence for pred 0 should be zero.
        # if no object is present at label 1, then the confidence for pred 1 should be zero.
        pred_bbox_one_confidence = preds[:, :, :, 4]  # (bs, S, S)
        pred_bbox_two_confidence = preds[:, :, :, 9]

        conf_noobj_loss = torch.sum(
            torch.logical_not(obj_at_label_one_mask.squeeze())  # (bs, S, S)
            * (pred_bbox_one_confidence**2)
        ) + torch.sum(
            torch.logical_not(obj_at_label_two_mask.squeeze())  # (bs, S, S)
            * (pred_bbox_two_confidence**2)
        )

        obj_in_cell_i_mask = obj_at_label_one_mask  # (bs, S, S, 1)

        clf_loss = F.mse_loss(
            obj_in_cell_i_mask * preds[:, :, :, B * 5 :],
            labels[:, :, :, B * 5 :],
            reduction="sum",
        )

        # TODO: if you really want to mean these, create the mean over the number of preds and labels
        # also don't just randomly use mean everywhere. stick to the formulae.

        print(
            "xy_loss: ",
            xy_loss,
            "wh_loss: ",
            wh_loss,
            "conf_loss: ",
            conf_loss,
            "conf_noobj_loss: ",
            self.lambda_noobj * conf_noobj_loss,
            "clf_loss: ",
            clf_loss,
        )
        loss = (
            self.lambda_coord * (xy_loss + wh_loss)
            + conf_loss
            + self.lambda_noobj * conf_noobj_loss
            + clf_loss
        )

        return loss
