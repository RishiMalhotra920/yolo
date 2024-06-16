import torch
from torch import nn


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, inputs, targets):
        return 0.5 * torch.mean((inputs - targets) ** 2)
