import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import yaml
from torchvision.transforms import v2 as transforms_v2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

import color_code_error_messages  # noqa: F401
import matplotlib.patches as patches

from src.data_setup.yolo_train_data_setup import create_datasets

config = yaml.safe_load(open("config.yaml"))


def plot(ax, image, bboxes):
    """
    rec of the form [x1, y1, x2, y2]
    """
    ax.imshow(image.permute(1, 2, 0))
    if bboxes is None:
        return

    for bbox in bboxes:
        ax.add_patch(
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                edgecolor="red",
                fill=False,
            )
        )


def visualize(args):
    transform = transforms_v2.Compose(
        [
            transforms_v2.ToTensor(),
            transforms_v2.RandomAffine(
                degrees=(0, 0), translate=(0.1, 0.1), scale=(1.0, 1.2), shear=0
            ),
            transforms_v2.Resize((448, 448)),
            transforms_v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_dataset, val_dataset = create_datasets(
        config["pascal_voc_root_dir"], transform
    )

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))

    for i in range(4):
        image, bboxes, label = val_dataset[i]
        print("trace 5", image, bboxes)
        image = image.squeeze(0)
        print("this is bboxes:", bboxes)
        if len(bboxes) > 0:
            plot(ax[i], image, bboxes)
        else:
            plot(ax[i], image, None)
            print("No bboxes found in image ", i)

    plt.show()


if __name__ == "__main__":
    # for example
    # python visualize_pascal_voc.py --checkpoint_signature IM-232:checkpoints/epoch_85 --threshold 0.5 --seed 420 --show_preds true
    # --hidden_units 256
    parser = argparse.ArgumentParser(description="Visualize the model's predictions")
    # parser.add_argument("--hidden_units", type=int,
    # help="The number of hidden units", required=True)
    args: dict[Any, Any] = {}
    visualize(args)
