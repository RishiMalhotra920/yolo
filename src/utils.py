import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

class_to_index = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}
index_to_class = {idx: cls_name for idx, cls_name in enumerate(VOC_CLASSES)}


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
        print("annotations", annotations)
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

        # predictions

        print("this is image shape", image.unsqueeze(0).shape)
        preds = model(image.unsqueeze(0))  # (1, 7, 7, 30)
        print("preds", preds)
        # get the predicted bounding boxes

        image_width = 448
        grid_cell_size = image_width / 7
        preds_grid_cell_x_pixel_start_coordinates = (
            torch.arange(0, 7) * image_width / 7
        ).repeat(7, 1)  # (7, 7)
        preds_grid_cell_y_pixel_start_coordinates = (
            preds_grid_cell_x_pixel_start_coordinates.T
        )  # (7, 7)

        # bbox 1
        preds[..., 0] = preds_grid_cell_x_pixel_start_coordinates + (
            preds[..., 0] * grid_cell_size
        )
        preds[..., 1] = preds_grid_cell_y_pixel_start_coordinates + (
            preds[..., 1] * grid_cell_size
        )

        preds[..., 2] = preds[..., 2] * image_width
        preds[..., 3] = preds[..., 3] * image_width
        # preds[..., 4] is confidence

        # bbox 2
        preds[..., 5] = preds_grid_cell_x_pixel_start_coordinates + (
            preds[..., 5] * grid_cell_size
        )
        preds[..., 6] = preds_grid_cell_y_pixel_start_coordinates + (
            preds[..., 6] * grid_cell_size
        )

        preds[..., 7] = preds[..., 7] * image_width
        preds[..., 8] = preds[..., 8] * image_width
        # preds[..., 9] is confidence

        confidence_threshold = 0.01

        for grid_x in range(7):
            for grid_y in range(7):
                for b in range(2):
                    confidence = preds[0, grid_x, grid_y, b * 5 + 4]
                    print("this is confidence", confidence)
                    if confidence > confidence_threshold:
                        bbox = preds[0, grid_x, grid_y, b * 5 : b * 5 + 4]
                        label_vec = preds[0, grid_x, grid_y, 10:]
                        label_idx = int(torch.argmax(label_vec).item())

                        label = index_to_class[label_idx]

                        x_center = bbox[0].item()
                        y_center = bbox[1].item()
                        width = bbox[2].item()
                        height = bbox[3].item()

                        x = x_center - width / 2
                        y = y_center - height / 2

                        print("this is x, y, width, height", x, y, width, height)

                        # Create a rectangle patch for each object and add it to the plot
                        rect = patches.Rectangle(
                            (x, y),
                            width,
                            height,
                            linewidth=2,
                            edgecolor="g",
                            facecolor="none",
                        )
                        ax.add_patch(rect)
                        ax.text(
                            x,
                            y - 10,
                            label,
                            color="white",
                            fontsize=12,
                            bbox=dict(facecolor="green", alpha=0.5),
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
