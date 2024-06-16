import torch
import torch.nn as nn
from torchinfo import summary


class YOLONet(nn.Module):
    def __init__(self, dropout):
        """

        """
        super(YOLONet, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2,
                      padding=3),  # originally stride 2
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.block_5_part_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.block_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.fc1 = nn.Linear(1024*7*7, 4096)
        self.fc2 = nn.Linear(4096, 1470)

        # adaptive avg pool is used to allow for any input size
        # and output a fixed size tensor
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_5_part_2(x)
        x = self.block_6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 7, 7, 30)

        # 30 units =>
        # [0 .. 2]: bbox_1_x, bbox_2_x
        # [3 .. 4]: bbox_1_y, bbox_2_y
        # [5 .. 6]: bbox_1_w, bbox_2_w
        # [7 .. 8]: bbox_1_h, bbox_2_h
        # [9]: bbox_1_confidence
        # [10]: bbox_2_confidence
        # [11 .. 30]: class probabilities

        return x


if __name__ == "__main__":
    # model = TinyVGG(hidden_units=128, output_shape=3)

    # print(summary(model))
    # print('-------')
    model = YOLONet(dropout=0)
    # print(summary(model))
    summary(model, (1, 3, 448, 448))
