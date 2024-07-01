import torch.nn as nn
from torchinfo import summary


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class FeatureExtractor(nn.Module):
    def __init__(self, dropout):
        super(FeatureExtractor, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # originally stride 2
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
            nn.MaxPool2d(kernel_size=2, stride=2),
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
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.feature_extractor = nn.Sequential(
            self.block_1,
            self.block_2,
            self.block_3,
            self.block_4,
            self.block_5,
        )

    def forward(self, x):
        return self.feature_extractor(x)


class YOLOPretrainNet(nn.Module):
    def __init__(self, dropout):
        super(YOLOPretrainNet, self).__init__()

        self.pretrain_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1000),
        )
        self.network = nn.Sequential(FeatureExtractor(dropout), self.pretrain_head)

    def forward(self, x):
        return self.network(x)


class YOLONet(nn.Module):
    def __init__(self, dropout):
        super(YOLONet, self).__init__()

        self.block_5_part_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.block_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.finetune_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1470),
        )

        self.network = nn.Sequential(
            FeatureExtractor(dropout),
            self.block_5_part_2,
            self.block_6,
            self.finetune_head,
            Reshape((-1, 7, 7, 30)),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # YOLO pretrain net
    model = YOLOPretrainNet(dropout=0)

    summary(
        model,
        (1, 3, 224, 224),
        depth=5,
        col_names=["input_size", "output_size", "num_params", "params_percent"],
        row_settings=["var_names"],
    )

    # YOLO net
    # model = YOLONet(dropout=0)

    # summary(model, (1, 3, 448, 448),
    #         depth=5,
    #         col_names=["input_size", "output_size",
    #                    "num_params", "params_percent"],
    #         row_settings=["var_names"])
