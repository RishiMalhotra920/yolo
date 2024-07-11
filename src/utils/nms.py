import torch
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
