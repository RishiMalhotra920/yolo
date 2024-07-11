import torch

from src.utils import nms_by_class  # Import your function


def test_nms_by_class_single_class():
    bboxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=torch.float
    )
    classes = torch.tensor([0, 0, 0])
    scores = torch.tensor([0.9, 0.8, 0.7])

    result_bboxes, result_scores, result_classes = nms_by_class(
        bboxes, scores, classes, iou_threshold=0.5
    )
    assert len(result_bboxes) == 2
    assert torch.allclose(
        result_bboxes,
        torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float),
    )
    assert torch.all(result_classes == torch.tensor([0, 0], dtype=torch.float))
    assert torch.allclose(result_scores, torch.tensor([0.9, 0.7], dtype=torch.float))


def test_nms_by_class_multiple_classes():
    bboxes = torch.tensor(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
            [0, 0, 10, 10],
            [21, 21, 31, 31],
        ],
        dtype=torch.float,
    )
    classes = torch.tensor([0, 0, 0, 1, 1])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.9, 0.8])

    result_bboxes, result_scores, result_classes = nms_by_class(
        bboxes, scores, classes, iou_threshold=0.5
    )
    assert len(result_bboxes) == 4
    expected_bboxes = torch.tensor(
        [
            [0, 0, 10, 10],  # Highest score for class 0
            [20, 20, 30, 30],  # Non-overlapping box for class 0
            [0, 0, 10, 10],  # Highest score for class 1
            [21, 21, 31, 31],  # Non-overlapping box for class 1
        ],
        dtype=torch.float,
    )
    expected_classes = torch.tensor([0, 0, 1, 1])
    expected_scores = torch.tensor([0.9, 0.7, 0.9, 0.8])

    assert torch.allclose(result_bboxes, expected_bboxes)
    assert torch.all(result_classes == expected_classes)
    assert torch.allclose(result_scores, expected_scores)


def test_nms_by_class_empty_input():
    bboxes = torch.tensor([], dtype=torch.float).reshape(0, 4)
    classes = torch.tensor([])
    scores = torch.tensor([])

    result_bboxes, result_scores, result_classes = nms_by_class(
        bboxes, scores, classes, iou_threshold=0.5
    )
    assert len(result_bboxes) == 0
    assert len(result_classes) == 0
    assert len(result_scores) == 0


def test_nms_by_class_no_suppressions():
    bboxes = torch.tensor(
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=torch.float
    )
    classes = torch.tensor([0, 1, 2])
    scores = torch.tensor([0.9, 0.8, 0.7])

    result_bboxes, result_scores, result_classes = nms_by_class(
        bboxes, scores, classes, iou_threshold=0.5
    )
    assert len(result_bboxes) == 3
    assert torch.allclose(result_bboxes, bboxes)
    assert torch.all(result_classes == classes)
    assert torch.allclose(result_scores, scores)


def test_nms_by_class_mixed_suppressions():
    bboxes = torch.tensor(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
            [0, 0, 10, 10],
            [21, 21, 31, 31],
            [40, 40, 50, 50],
        ],
        dtype=torch.float,
    )
    classes = torch.tensor([0, 0, 0, 1, 1, 2])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.9, 0.8, 0.7])

    result_bboxes, result_scores, result_classes = nms_by_class(
        bboxes, scores, classes, iou_threshold=0.5
    )
    assert len(result_bboxes) == 5
    expected_bboxes = torch.tensor(
        [
            [0, 0, 10, 10],  # Highest score for class 0
            [20, 20, 30, 30],  # Non-overlapping box for class 0
            [0, 0, 10, 10],  # Highest score for class 1
            [21, 21, 31, 31],  # Non-overlapping box for class 1
            [40, 40, 50, 50],  # Only box for class 2
        ],
        dtype=torch.float,
    )
    expected_classes = torch.tensor([0, 0, 1, 1, 2], dtype=torch.float)
    expected_scores = torch.tensor([0.9, 0.7, 0.9, 0.8, 0.7], dtype=torch.float)

    assert torch.allclose(result_bboxes.float(), expected_bboxes)
    assert torch.all(result_classes.float() == expected_classes)
    assert torch.allclose(result_scores.float(), expected_scores)
