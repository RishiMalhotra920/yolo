import color_code_error_messages  # noqa: F401
import torch
from torchvision.ops import nms  # noqa: F401


def nms_by_class(
    bboxes: torch.Tensor,
    classes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
):
    bboxes_and_conf_by_class: dict[int, list[torch.Tensor]] = {i: [] for i in range(20)}
    for bbox, bbox_class, conf in zip(bboxes, classes, scores):
        bboxes_and_conf_by_class[int(bbox_class.item())].append(
            torch.cat([bbox, conf.unsqueeze(-1)], dim=-1)
        )

    print("this is bboxes_and_conf_by_class", bboxes_and_conf_by_class)
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
        print("this is bbox_tensor", bbox_tensor)
        conf_tensor = (
            torch.stack(bboxes_and_conf)[:, -1] if bboxes_and_conf else torch.tensor([])
        )

        if len(bbox_tensor) != 0:
            # pass
            print(bbox_tensor.shape)
            nms_indices = nms(bbox_tensor, conf_tensor, iou_threshold)
            print("this is nms_indices", nms_indices)

            bboxes_to_return.append(bbox_tensor[nms_indices, :4])
            classes_to_return.append(class_idx * torch.ones(len(nms_indices)))
            confidences_to_return.append(conf_tensor[nms_indices])

    print("this is bboxes", bboxes_to_return)
    # # we have an array of tensors for each class
    # # bboxes_to_return = [tensor([[0, 0, 10, 10], [20, 20, 30, 30]]), tensor([[0, 0, 10, 10], [21, 21, 31, 31]])]
    # # classes_to_return = [tensor([0, 0]), tensor([1, 1])]
    # # confidences_to_return = [tensor(0.9), tensor(0.8)]

    return (
        torch.cat(bboxes_to_return, dim=0),
        torch.cat(classes_to_return, dim=0),
        torch.cat(confidences_to_return, dim=0),
    )


bboxes = torch.tensor(
    [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=torch.float
)
classes = torch.tensor([0, 0, 0])
scores = torch.tensor([0.9, 0.8, 0.7])


result_bboxes, result_classes, result_scores = nms_by_class(
    bboxes, classes, scores, iou_threshold=0.5
)
print("this is result_bboxes", result_bboxes)
assert len(result_bboxes) == 2
assert torch.allclose(
    result_bboxes, torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float)
)
assert torch.all(result_classes == torch.tensor([0, 0]))
