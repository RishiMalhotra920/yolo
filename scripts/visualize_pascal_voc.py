import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

from torchvision.transforms import v2 as transforms_v2

from src.checkpoint_loader import load_checkpoint
from src.data_setup.yolo_train_data_setup import create_datasets
from src.models import yolo_net
from src.utils import predict_on_random_pascal_voc_images

config = yaml.safe_load(open("config.yaml"))


def visualize(args):
    data_transform = transforms_v2.Compose(
        [
            # transforms.RandomResizedCrop(50),
            # transforms_v2.Resize((224, 224)),
            # transforms_v2.RandomHorizontalFlip(),
            # transforms_v2.RandomRotation((-30, 30)),
            # transforms_v2.ColorJitter(brightness=0.5, contrast=0.5),
            # transforms_v2.ToTensor(),
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #  std=[0.229, 0.224, 0.225]),
            # transforms.RandomErasing()
        ]
    )

    # to_tensor_transform = transforms_v2.Compose(
    #     [
    #         transforms_v2.ToImage(),
    #         transforms_v2.ToDtype(torch.float32, scale=True),
    #     ]
    # )

    train_dataset, mini_val_dataset = create_datasets(
        config["pascal_voc_root_dir"], data_transform
    )

    model = yolo_net.YOLONet(dropout=0).to("cpu")

    if args.checkpoint_signature is not None:
        load_checkpoint(model, args.checkpoint_signature)

    predict_on_random_pascal_voc_images(model, mini_val_dataset, n=5, seed=4)

    # display_random_images(train_dataset,
    #   class_names=class_names, n=5, seed=4)


if __name__ == "__main__":
    # for example
    # python visualize.py --run_id "IM-28" --checkpoint_path checkpoints/epoch_1
    # --hidden_units 256
    parser = argparse.ArgumentParser(description="Visualize the model's predictions")
    # parser.add_argument("--hidden_units", type=int,
    # help="The number of hidden units", required=True)
    parser.add_argument(
        "--checkpoint_signature",
        type=str,
        help="The path to the checkpoint in the format RUN_ID:CHECKPOINT_PATH eg: IM-28:checkpoints/epoch_5",
        required=False,
    )
    args = parser.parse_args()
    if args.checkpoint_signature is None:
        print("Not loading checkpoint. Model will be randomly initialized.")
    visualize(args)
