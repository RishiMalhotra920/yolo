import os
import random
import sys
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.utils.nms import nms_by_class

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402


from torchvision.transforms import v2 as transforms_v2

from src.data_setup.yolo_train_data_setup import create_datasets
from src.utils.yolo_classes import class_to_index, index_to_class


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


def predict_on_image(
    ax: Any,
    image: torch.Tensor,
    target_yolo_grid: torch.Tensor | None,
    metadata: dict[str, Any],
    model: torch.nn.Module,
    threshold: float,
    show_labels: bool,
    show_preds: bool,
    apply_nms: bool,
    nms_iou_threshold: float,
) -> None:
    print("in here!")
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(metadata["image_path"])

    if show_labels:
        if target_yolo_grid is None:
            raise ValueError("target_yolo_grid must be provided if show_labels is True")

        target_bboxes, _, target_labels = (
            transform_yolo_grid_into_bboxes_confidences_and_labels(
                target_yolo_grid, metadata["image_width"], metadata["image_height"]
            )
        )
        for label in target_labels:
            print("Label: ", label)

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
            labels = [index_to_class[int(this_label.item())] for this_label in labels]
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


def predict_with_yolo(
    model: torch.nn.Module,
    threshold,
    *,
    image_dir_path: str | None = None,
    pascal_voc_root_dir: str | None = None,
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

    print("in red are the ground truth bounding boxes")
    print("in green are the predicted bounding boxes")
    print("image paths from left to right:")

    if image_dir_path:
        image_paths = [
            os.path.join(image_dir_path, image_path)
            for image_path in os.listdir(image_dir_path)
        ]
        fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 3))

        transform = transforms_v2.Compose(
            [
                transforms_v2.PILToTensor(),
                transforms_v2.Resize((448, 448)),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True),
            ]
        )
        for i, image_path in enumerate(image_paths):
            image_PIL = Image.open(image_path)

            image = transform(image_PIL)

            print("this is image", image.shape)

            ax = axes[i]
            metadata = {
                "image_id": i,
                "image_width": image_PIL.width,
                "image_height": image_PIL.height,
                "image_path": "",
            }
            predict_on_image(
                ax,
                image,
                None,
                metadata,
                model,
                threshold,
                show_labels,
                show_preds,
                apply_nms,
                nms_iou_threshold,
            )
    else:
        if not pascal_voc_root_dir:
            raise ValueError(
                "Either image_dir_path or pascal_voc_root_dir must be provided"
            )

        data_transform = transforms_v2.Compose(
            [
                transforms_v2.Resize((448, 448)),
                transforms_v2.RandomHorizontalFlip(),
                transforms_v2.RandomAffine(
                    degrees=(0, 30), translate=(0.1, 0.1), scale=(1.0, 1.2), shear=0
                ),
                # transforms_v2.ColorJitter(brightness=0.5, contrast=0.5),
                transforms_v2.ToTensor(),
                # transforms_v2.Normalize(
                # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )
        # use validation dataset
        _, dataset = create_datasets(pascal_voc_root_dir, data_transform)

        random_samples_idx = random.sample(range(len(dataset)), k=n)  # type: ignore

        print("this is random_samples_idx", random_samples_idx)

        fig, axes = plt.subplots(1, n, figsize=(15, 3))
        for i, index in enumerate(random_samples_idx):
            print("this is index", index)
            image, target_yolo_grid, metadata = dataset[index]
            ax = axes[i]
            predict_on_image(
                ax,
                image,
                target_yolo_grid,
                metadata,
                model,
                threshold,
                show_labels,
                show_preds,
                apply_nms,
                nms_iou_threshold,
            )
            ax.axis("off")

    plt.show()
