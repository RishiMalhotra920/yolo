import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch


def calculate_iou(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    tensor1 and tensor2 are of shape (batch_size, S, S, 4)

    the last dimension of the tensor represents x_center, y_center, width, height

    calculate the intersection over each cell of the grid for tensor1 and tensor2

    Returns:
        iou: torch.Tensor of shape (batch_size, S, S)
    """
    # return torch.tensor([2, 3])

    tl_x1 = tensor1[:, :, :, 0] - tensor1[:, :, :, 2] / 2
    tl_y1 = tensor1[:, :, :, 1] - tensor1[:, :, :, 3] / 2
    br_x1 = tensor1[:, :, :, 0] + tensor1[:, :, :, 2] / 2
    br_y1 = tensor1[:, :, :, 1] + tensor1[:, :, :, 3] / 2

    tl_x2 = tensor2[:, :, :, 0] - tensor2[:, :, :, 2] / 2
    tl_y2 = tensor2[:, :, :, 1] - tensor2[:, :, :, 3] / 2
    br_x2 = tensor2[:, :, :, 0] + tensor2[:, :, :, 2] / 2
    br_y2 = tensor2[:, :, :, 1] + tensor2[:, :, :, 3] / 2

    tl_x = torch.max(tl_x1, tl_x2)
    tl_y = torch.max(tl_y1, tl_y2)
    br_x = torch.min(br_x1, br_x2)
    br_y = torch.min(br_y1, br_y2)

    intersection = torch.max(br_x - tl_x, torch.tensor(0)) * torch.max(
        br_y - tl_y, torch.tensor(0)
    )

    width1 = tensor1[:, :, :, 2]
    height1 = tensor1[:, :, :, 3]
    width2 = tensor2[:, :, :, 2]
    height2 = tensor2[:, :, :, 3]

    union = width1 * height1 + width2 * height2 - intersection

    return intersection / union


def get_yolo_metrics(pred: torch.Tensor, label: torch.Tensor) -> dict:
    """Count the number of incorrect background predictions.

    Args:
        pred (torch.Tensor): The logits or probabilities from the model. Shape: (bs, 7, 7, 30)
        label (torch.Tensor): The true labels for each input. Shape: (bs, 7, 7, 30)

    Returns:
        int: The number of incorrect background predictions.
    """
    B = 2

    does_label_1_exist_for_each_cell = label[..., 2] > 0
    does_label_2_exist_for_each_cell = label[..., 7] > 0

    # 0, 1, 2, 3 - bbox 1. 4 - bbox 1 confidence
    iou_pred_1_and_label_1 = calculate_iou(pred[..., :4], label[..., :4])  # (bs, 7, 7)

    # 5, 6, 7, 8 - bbox 2. 9 - bbox 2 confidence
    iou_pred_2_and_label_2 = calculate_iou(
        pred[..., 5:9], label[..., 5:9]
    )  # (bs, 7, 7)
    # Check if the predicted class labels are correct
    is_class_correct = (pred[..., B * 5 :] == label[..., B * 5 :]).all(
        dim=-1
    )  # (bs, 7, 7)

    num_correct_for_pred_box_1 = (
        torch.logical_and((iou_pred_1_and_label_1 > 0.5), is_class_correct).sum().item()
    )
    num_correct_for_pred_box_2 = (
        torch.logical_and((iou_pred_2_and_label_2 > 0.5), is_class_correct).sum().item()
    )

    num_incorrect_localization_for_pred_box_1 = (
        torch.logical_and(
            torch.logical_and(
                does_label_1_exist_for_each_cell,
                torch.logical_and(
                    0.1 <= iou_pred_1_and_label_1, iou_pred_1_and_label_1 <= 0.5
                ),
            ),
            is_class_correct,
        )
        .sum()
        .item()
    )

    num_incorrect_localization_for_pred_box_2 = (
        torch.logical_and(
            torch.logical_and(
                does_label_2_exist_for_each_cell,
                torch.logical_and(
                    0.1 <= iou_pred_2_and_label_2, iou_pred_2_and_label_2 <= 0.5
                ),
            ),
            is_class_correct,
        )
        .sum()
        .item()
    )

    num_incorrect_other_for_pred_box_1 = (
        torch.logical_and((iou_pred_1_and_label_1 > 0.1), ~is_class_correct)
        .sum()
        .item()
    )

    num_incorrect_other_for_pred_box_2 = (
        torch.logical_and((iou_pred_2_and_label_2 > 0.1), ~is_class_correct)
        .sum()
        .item()
    )

    num_incorrect_background_for_pred_box_1 = (
        torch.logical_and((iou_pred_1_and_label_1 <= 0.1), is_class_correct)
        .sum()
        .item()
    )

    num_incorrect_background_for_pred_box_2 = (
        torch.logical_and((iou_pred_2_and_label_2 <= 0.1), is_class_correct)
        .sum()
        .item()
    )

    return {
        "num_correct": num_correct_for_pred_box_1 + num_correct_for_pred_box_2,
        "num_incorrect_localization": num_incorrect_localization_for_pred_box_1
        + num_incorrect_localization_for_pred_box_2,
        "num_incorrect_other": num_incorrect_other_for_pred_box_1
        + num_incorrect_other_for_pred_box_2,
        "num_incorrect_background": num_incorrect_background_for_pred_box_1
        + num_incorrect_background_for_pred_box_2,
    }


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


def display_random_images(
    dataset, *, class_names: list[str] | None = None, n: int = 5, seed: int = 42
) -> None:
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
        ax.axis("off")
    plt.show()


def predict_on_random_pascal_voc_images(
    model: torch.nn.Module,
    dataset,
    *,
    class_names: list[str] | None = None,
    n: int = 5,
    seed: int | None = None,
):
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
        print("annotations", dataset[index])
        objects = annotations["annotation"]["object"]
        for obj in objects:
            bbox = obj["bndbox"]
            x = int(bbox["xmin"])
            y = int(bbox["ymin"])
            width = int(bbox["xmax"]) - x
            height = int(bbox["ymax"]) - y

            # Create a rectangle patch for each object and add it to the plot
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y - 10,
                obj["name"],
                color="white",
                fontsize=12,
                bbox=dict(facecolor="red", alpha=0.5),
            )

        ax.axis("off")
    plt.show()


def predict_on_random_image_net_images(
    model: torch.nn.Module,
    dataset,
    *,
    class_names: list[str] | None = None,
    n: int = 5,
    seed: int | None = None,
):
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(15, 3))

    images = []
    for i, index in enumerate(random_samples_idx):
        image, label = dataset[index]
        ax = axes[i]
        pred_logits = model(image.unsqueeze(0))
        print("pred_logits", pred_logits)
        pred = int(torch.argmax(pred_logits.squeeze()).item())
        print("pred", pred)

        ax.imshow(image.permute(1, 2, 0))
        if class_names:
            ax.set_title(
                f"Label: {label} {class_names[label]}\nPred: {pred} {class_names[pred]}"
            )
        else:
            ax.set_title(f"Label: {label}\nPred: {pred}")

        ax.axis("off")
    plt.show()
