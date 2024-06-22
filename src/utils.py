import random
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch


def count_top_k_correct(output: torch.Tensor, target: torch.Tensor, k: int):
    """Compute top-k accuracy for the given predictions and labels.

    Args:
        output (torch.Tensor): The logits or probabilities from the model.
        target (torch.Tensor): The true labels for each input.
        k (int): The top 'k' predictions considered to calculate the accuracy.

    Returns:
        float: The top-k accuracy.
    """
    # Get the top k predictions from the model for each input
    _, predicted = output.topk(k, 1, True, True)

    # View target to make it [batch_size, 1]
    target = target.view(-1, 1)

    # Check if the true labels are in the top k predictions
    correct = predicted.eq(target).sum().item()

    # Calculate the accuracy
    return correct


def display_random_images(dataset,
                          *,
                          class_names: list[str] | None = None,
                          n: int = 5,
                          seed: int = 42) -> None:

    random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(15, 5))

    for idx, random_sample_idx in enumerate(random_samples_idx):
        image, label = dataset[random_sample_idx]
        ax = axes[idx]
        ax.imshow(image.permute(1, 2, 0))
        if class_names:
            ax.set_title(f"Label: {label},\nclass:{class_names[label]}")
        else:
            ax.set_title(f"Label:{label}")
    # ax[1].imshow(image_net_val[0][0].permute(1, 2, 0))
    # ax[1].set_title('Validation image')
        ax.axis('off')
    plt.show()


def predict_on_random_pascal_voc_images(model: torch.nn.Module,
                                        dataset,
                                        *,
                                        class_names: list[str] | None = None,
                                        n: int = 5,
                                        seed: int | None = None):
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(15, 3))

    images = []
    for i, index in enumerate(random_samples_idx):
        image, annotations = dataset[index]
        ax = axes[i]
        # pred_logits = model(image.unsqueeze(0))
        # print('pred_logits', pred_logits)
        # pred = int(torch.argmax(pred_logits.squeeze()).item())
        # print('pred', pred)

        ax.imshow(image.permute(1, 2, 0))
        # if class_names:
        #     ax.set_title(
        #         f"Label: {label} {class_names[label]}\nPred: {pred} {class_names[pred]}")
        # else:
        #     ax.set_title(f"Label: {label}\nPred: {pred}")

        # Loop through the objects and draw each one
        print('annotations', dataset[index])
        objects = annotations['annotation']['object']
        for obj in objects:
            bbox = obj['bndbox']
            x = int(bbox['xmin'])
            y = int(bbox['ymin'])
            width = int(bbox['xmax']) - x
            height = int(bbox['ymax']) - y

            # Create a rectangle patch for each object and add it to the plot
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 10, obj['name'], color='white',
                    fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        ax.axis('off')
    plt.show()


def predict_on_random_image_net_images(model: torch.nn.Module,
                                       dataset,
                                       *,
                                       class_names: list[str] | None = None,
                                       n: int = 5,
                                       seed: int | None = None):
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(15, 3))

    images = []
    for i, index in enumerate(random_samples_idx):
        image, label = dataset[index]
        ax = axes[i]
        pred_logits = model(image.unsqueeze(0))
        print('pred_logits', pred_logits)
        pred = int(torch.argmax(pred_logits.squeeze()).item())
        print('pred', pred)

        ax.imshow(image.permute(1, 2, 0))
        if class_names:
            ax.set_title(
                f"Label: {label} {class_names[label]}\nPred: {pred} {class_names[pred]}")
        else:
            ax.set_title(f"Label: {label}\nPred: {pred}")

        ax.axis('off')
    plt.show()
