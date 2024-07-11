import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as transforms_v2

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


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # Convert to (x1, y1, x2, y2) format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = (ovr <= iou_threshold).nonzero().squeeze()
        order = order[inds + 1]

    return torch.tensor(keep)


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

    iou = intersection / union

    # filter out nans. we define iou as 0 if there's no bboxes.
    # this works well because we intend to calculate the argmax over the iou values.
    # so if there's no bbox, the iou will be 0 and this won't be selected by the argmax
    iou[torch.isnan(iou)] = 0

    return iou


def get_yolo_metrics(pred: torch.Tensor, label: torch.Tensor) -> dict:
    """Count the number of incorrect background predictions.

    Args:
        pred (torch.Tensor): The logits or probabilities from the model. Shape: (bs, 7, 7, 30)
        label (torch.Tensor): The true labels for each input. Shape: (bs, 7, 7, 30)

    Returns:
        int: The number of incorrect background predictions.
    """
    B = 2

    does_label_1_exist_for_each_cell = label[..., 4] == 1.0
    does_label_2_exist_for_each_cell = label[..., 9] == 1.0

    # 0, 1, 2, 3 - bbox 1. 4 - bbox 1 confidence
    iou_pred_1_and_label_1 = calculate_iou(pred[..., :4], label[..., :4])  # (bs, 7, 7)

    # 5, 6, 7, 8 - bbox 2. 9 - bbox 2 confidence
    iou_pred_2_and_label_2 = calculate_iou(
        pred[..., 5:9], label[..., 5:9]
    )  # (bs, 7, 7)
    # Check if the predicted class labels are correct

    preds_argmax = torch.argmax(pred[..., B * 5 :], dim=-1)
    labels_argmax = torch.argmax(label[..., B * 5 :], dim=-1)

    is_class_correct = preds_argmax == labels_argmax  # (bs, 7, 7)

    num_correct_for_pred_box_1 = (
        (
            does_label_1_exist_for_each_cell
            & (iou_pred_1_and_label_1 > 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_correct_for_pred_box_2 = (
        (
            does_label_2_exist_for_each_cell
            & (iou_pred_2_and_label_2 > 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_localization_for_pred_box_1 = (
        (
            does_label_1_exist_for_each_cell
            & (0.1 <= iou_pred_1_and_label_1)
            & (iou_pred_1_and_label_1 <= 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_localization_for_pred_box_2 = (
        (
            does_label_2_exist_for_each_cell
            & (0.1 <= iou_pred_2_and_label_2)
            & (iou_pred_2_and_label_2 <= 0.5)
            & is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_other_for_pred_box_1 = (
        (
            does_label_1_exist_for_each_cell
            & (iou_pred_1_and_label_1 > 0.1)
            & ~is_class_correct
        )
        .sum()
        .item()
    )

    num_incorrect_other_for_pred_box_2 = (
        (
            does_label_2_exist_for_each_cell
            & (iou_pred_2_and_label_2 > 0.1)
            & ~is_class_correct
        )
        .sum()
        .item()
    )

    # incorrectly classified object as background

    num_incorrect_background_for_pred_box_1 = (
        (does_label_1_exist_for_each_cell & (iou_pred_1_and_label_1 <= 0.1))
        .sum()
        .item()
    )

    num_incorrect_background_for_pred_box_2 = (
        (does_label_2_exist_for_each_cell & (iou_pred_2_and_label_2 <= 0.1))
        .sum()
        .item()
    )
    num_objects = (
        does_label_1_exist_for_each_cell.sum().item()
        + does_label_2_exist_for_each_cell.sum().item()
    )

    return {
        "num_correct": num_correct_for_pred_box_1 + num_correct_for_pred_box_2,
        "num_incorrect_localization": num_incorrect_localization_for_pred_box_1
        + num_incorrect_localization_for_pred_box_2,
        "num_incorrect_other": num_incorrect_other_for_pred_box_1
        + num_incorrect_other_for_pred_box_2,
        "num_incorrect_background": num_incorrect_background_for_pred_box_1
        + num_incorrect_background_for_pred_box_2,
        "num_objects": num_objects,
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


from torchvision.ops import nms


def nms_by_class(
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
):
    """
    Perform non-maximum suppression on the bounding boxes by class.

    Args:
        bboxes: torch.Tensor of shape (num_bboxes, 4) - the bounding boxes
        scores: torch.Tensor of shape (num_bboxes) - the confidence scores
        classes: torch.Tensor of shape (num_bboxes) - the class labels
        iou_threshold: float - the iou threshold for non-maximum suppression

    Returns:
        tuple: (bboxes, classes, scores) - the filtered bounding boxes, classes, and scores
    """

    bboxes_and_conf_by_class: dict[int, list[torch.Tensor]] = {i: [] for i in range(20)}
    for bbox, bbox_class, conf in zip(bboxes, classes, scores):
        bboxes_and_conf_by_class[int(bbox_class.item())].append(
            torch.cat([bbox, conf.unsqueeze(-1)], dim=-1)
        )

    # print("this is bboxes_and_conf_by_class", bboxes_and_conf_by_class)
    # we have a
    # {0: [tensor([0, 0, 10, 10, 0, 0.9]), tensor([1, 1, 11, 11, 0, 0.8]), tensor([20, 20, 30, 30, 0, 0.7])],
    #  1: [tensor([0, 0, 10, 10, 1, 0.9]), tensor([21, 21, 31, 31, 1, 0.8])]}
    # and so on

    bboxes_to_return = []
    classes_to_return = []
    confidences_to_return = []
    for class_idx, bboxes_and_conf in bboxes_and_conf_by_class.items():
        # print("this is", bboxes_and_conf)
        bbox_tensor = (
            torch.stack(bboxes_and_conf)[:, :4] if bboxes_and_conf else torch.tensor([])
        )
        # print("this is bbox_tensor", bbox_tensor)
        conf_tensor = (
            torch.stack(bboxes_and_conf)[:, -1] if bboxes_and_conf else torch.tensor([])
        )

        if len(bbox_tensor) != 0:
            nms_indices = nms(bbox_tensor, conf_tensor, iou_threshold)
            bboxes_to_return.append(bbox_tensor[nms_indices, :4])
            classes_to_return.append(class_idx * torch.ones(len(nms_indices)))
            confidences_to_return.append(conf_tensor[nms_indices])

    # print("this is bboxes", bboxes_to_return)
    # # we have an array of tensors for each class
    # # bboxes_to_return = [tensor([[0, 0, 10, 10], [20, 20, 30, 30]]), tensor([[0, 0, 10, 10], [21, 21, 31, 31]])]
    # # classes_to_return = [tensor([0, 0]), tensor([1, 1])]
    # # confidences_to_return = [tensor(0.9), tensor(0.8)]

    return (
        torch.cat(bboxes_to_return, dim=0) if bboxes_to_return else torch.tensor([]),
        torch.cat(confidences_to_return, dim=0)
        if confidences_to_return
        else torch.tensor([]),
        torch.cat(classes_to_return, dim=0)
        if classes_to_return
        else torch.tensor([])
        if confidences_to_return
        else torch.tensor([]),
    )


def transform_yolo_grid_into_bboxes_confidences_and_labels(
    preds: torch.Tensor,
    image_width: float,
    image_height: float,
) -> tuple[list[list[float]], list[float], list[str]]:
    """
    takes in the preds or labels tensor and returns a list of bboxes
    Args:
        preds: shape (7, 7, 30)
        image_width: int - the original image width
        image_height: int - the original image height

    Returns:
        bboxes: list[list[float]] - list of bboxes in xyxy format
        confidences: list[float] - list of confidences
        labels: list[str] - list of labels
    """

    bboxes = []
    confidences = []
    labels = []
    cell_width = image_width / 7
    cell_height = image_height / 7

    for grid_x in range(7):
        for grid_y in range(7):
            for b in range(2):
                x_center_relative = float(preds[grid_x, grid_y, b * 5].item())
                y_center_relative = float(preds[grid_x, grid_y, b * 5 + 1].item())
                width_relative = float(preds[grid_x, grid_y, b * 5 + 2].item())
                height_relative = float(preds[grid_x, grid_y, b * 5 + 3].item())

                x_start = image_width / 7 * grid_x
                y_start = image_height / 7 * grid_y

                x_center = x_start + x_center_relative * cell_width
                y_center = y_start + y_center_relative * cell_height

                width = width_relative * image_width
                height = height_relative * image_height
                x = x_center - width / 2
                y = y_center - height / 2

                confidence = float(preds[grid_x, grid_y, b * 5 + 4].item())

                label_vec = preds[grid_x, grid_y, 10:]

                # yolo_grids without labels (eg: empty squares in a dataset) will have all 0s
                # some yolo_grids have one bbox and the other bbox is filled with 0s -> 0 width
                if not torch.all(label_vec == 0) and width > 0 and height > 0:
                    label_idx = int(torch.argmax(label_vec).item())
                    label = index_to_class[label_idx]

                    # let's be consistent and use xyxy format everywhere
                    bboxes.append([x, y, x + width, y + height])
                    confidences.append(confidence)
                    labels.append(label)

    return bboxes, confidences, labels


def predict_on_random_pascal_voc_images(
    model: torch.nn.Module,
    dataset,
    threshold,
    *,
    show_preds: bool,
    show_labels: bool,
    n: int = 5,
    seed: int | None = None,
    apply_nms: bool = True,
    nms_iou_threshold: float = 0.5,
):
    # TODO: should refer to the YOLO grid output as yolo_grid and tensor of labels as labels
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(15, 3))

    print("in red are the ground truth bounding boxes")
    print("in green are the predicted bounding boxes")

    print("image paths from left to right:")
    for i, index in enumerate(random_samples_idx):
        image, target_yolo_grid, metadata = dataset[index]

        ax = axes[i]
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(metadata["image_path"])
        target_bboxes, _, target_labels = (
            transform_yolo_grid_into_bboxes_confidences_and_labels(
                target_yolo_grid, metadata["image_width"], metadata["image_height"]
            )
        )
        print(i, metadata["image_path"])
        for label in target_labels:
            print("Label: ", label)

        if show_labels:
            for i in range(len(target_bboxes)):
                x1, y1, x2, y2 = target_bboxes[i]
                label = target_labels[i]
                width = x2 - x1
                height = y2 - y1

                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )

                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 10,
                    target_labels[i],
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.5),
                )

        if show_preds:
            image_color_transform = transforms_v2.Compose(
                [
                    transforms_v2.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                ]
            )
            image = image_color_transform(image)
            pred_yolo_grid = model(image.unsqueeze(0)).squeeze()  # (7, 7, 30)

            image_width = metadata["image_width"]
            image_height = metadata["image_height"]

            all_bboxes, all_confidences, all_labels = (
                transform_yolo_grid_into_bboxes_confidences_and_labels(
                    pred_yolo_grid, image_width, image_height
                )
            )

            if apply_nms:
                bboxes, confidences, labels = nms_by_class(
                    torch.tensor(all_bboxes),
                    torch.tensor(all_confidences),
                    torch.tensor([class_to_index[label] for label in all_labels]),
                    nms_iou_threshold,
                )
                labels = [
                    index_to_class[int(this_label.item())] for this_label in labels
                ]
                print("after nms", len(bboxes))
            else:
                bboxes, confidences, labels = all_bboxes, all_confidences, all_labels

            # get the predicted bounding boxes

            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width = x2 - x1
                height = y2 - y1
                confidence = confidences[i]
                label = labels[i]

                if confidence > threshold:
                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="g",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x1,
                        y1 - 10,
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

    image_color_transform = transforms_v2.Compose(
        [transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    for i, index in enumerate(random_samples_idx):
        image, label = dataset[index]
        ax = axes[i]
        ax.imshow(image.permute(1, 2, 0))

        image = image_color_transform(image)

        pred_logits = model(image.unsqueeze(0))

        pred = int(torch.argmax(pred_logits.squeeze()).item())

        if class_names:
            ax.set_title(
                f"Label: {label} {class_names[label]}\nPred: {pred} {class_names[pred]}"
            )
        else:
            ax.set_title(f"Label: {label}\nPred: {pred}")

        ax.axis("off")
    plt.show()
