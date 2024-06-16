import torch
import torch.nn as nn
from torchinfo import summary


class IncompleteYoloPretrainConvNet(nn.Module):
    def __init__(self, dropout):
        """
        13M parameters
        """
        super(IncompleteYoloPretrainConvNet, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1),  # originally stride 2
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # adaptive avg pool is used to allow for any input size
        # and output a fixed size tensor
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # batch size x 1024
        x = self.dropout(x)
        x = self.fc(x)
        return x
