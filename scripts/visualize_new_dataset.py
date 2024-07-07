import argparse
import os
import sys

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402

import color_code_error_messages  # noqa: F401

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
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                edgecolor="red",
                fill=False,
            )
        )


def visualize(args):
    train_dataset, val_dataset = create_datasets(config["pascal_voc_root_dir"])

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))

    for i in range(4):
        image, bboxes = val_dataset[i]
        image = image.squeeze(0)
        print("this is bboxes:", bboxes)
        print("image type:", type(image))
        print("image shape:", image.shape)
        print("bboxes:", bboxes)
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
    args = {}
    visualize(args)
