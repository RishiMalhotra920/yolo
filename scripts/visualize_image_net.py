import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

from torchvision.transforms import v2 as transforms_v2

from src.checkpoint_loader import load_checkpoint
from src.data_setup import yolo_pretrain_data_setup
from src.models import yolo_net
from src.utils import predict_on_random_image_net_images

config = yaml.safe_load(open("config.yaml"))


def visualize(args):
    data_transform = transforms_v2.Compose(
        [
            # transforms.RandomResizedCrop(50),
            transforms_v2.Resize((224, 224)),
            transforms_v2.RandomHorizontalFlip(),
            transforms_v2.RandomRotation((-30, 30)),
            transforms_v2.ColorJitter(brightness=0.5, contrast=0.5),
            transforms_v2.ToTensor(),
            # transforms.RandomErasing()
        ]
    )

    # to_tensor_transform = transforms_v2.Compose([
    # transforms.ToTensor()
    # ])

    _, mini_val_dataset = yolo_pretrain_data_setup.create_datasets(
        config["image_net_data_dir"], data_transform
    )

    # model = model_builder.DeepConvNet(
    # hidden_units=args.hidden_units,
    # output_shape=len(classes)
    # ).to("cpu")

    model = yolo_net.YOLOPretrainNet(dropout=0).to("cpu")
    model = torch.nn.DataParallel(model)

    load_checkpoint(model, args.checkpoint_signature)

    predict_on_random_image_net_images(model, mini_val_dataset, n=10, seed=4200)

    # display_random_images(train_dataset,
    #   class_names=class_names, n=5, seed=4)


if __name__ == "__main__":
    # for example
    # python visualize_image_net.py --checkpoint_signature=IM-122:checkpoints/epoch_14

    # --hidden_units 256
    parser = argparse.ArgumentParser(description="Visualize the model's predictions")
    # parser.add_argument("--hidden_units", type=int,
    # help="The number of hidden units", required=True)

    parser.add_argument(
        "--checkpoint_signature",
        type=str,
        help="The signature for the checkpoint",
        required=True,
    )
    args = parser.parse_args()
    visualize(args)
