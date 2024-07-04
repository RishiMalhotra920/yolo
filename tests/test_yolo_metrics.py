# get_yolo_metrics classifies into these categories:
# "num_correct": num_correct_for_pred_box_1 + num_correct_for_pred_box_2,
# "num_incorrect_localization": num_incorrect_localization_for_pred_box_1
# + num_incorrect_localization_for_pred_box_2,
# "num_incorrect_other": num_incorrect_other_for_pred_box_1
# + num_incorrect_other_for_pred_box_2,
# "num_incorrect_background": num_incorrect_background_for_pred_box_1
# + num_incorrect_background_for_pred_box_2,

# test: see if num_correct is calculated correctly

from src.utils import get_yolo_metrics


def test_num_correct_same_data(sample_yolo_data_1):
    metrics = get_yolo_metrics(
        sample_yolo_data_1["yolo_net_target_tensor"],
        sample_yolo_data_1["yolo_net_target_tensor"],
    )

    # two correct predictions

    assert metrics["num_correct"] == 2, "Expected num_correct to be 1"

    assert (
        metrics["num_incorrect_localization"] == 0
    ), "Expected num_incorrect_localization to be 0"
    assert metrics["num_incorrect_other"] == 0, "Expected num_incorrect_other to be 0"
    assert (
        metrics["num_incorrect_background"] == 0
    ), "Expected num_incorrect_background to be 0"


def test_num_correct_different_data(sample_yolo_data_2):
    metrics = get_yolo_metrics(
        sample_yolo_data_2["yolo_net_output_tensor"],
        sample_yolo_data_2["yolo_net_target_tensor"],
    )

    # one correct prediction, one incorrect prediction

    assert metrics["num_correct"] == 2, "Expected num_correct to be 1"

    assert (
        metrics["num_incorrect_localization"] == 0
    ), "Expected num_incorrect_localization to be 1"
    assert metrics["num_incorrect_other"] == 0, "Expected num_incorrect_other to be 0"
    assert (
        metrics["num_incorrect_background"] == 0
    ), "Expected num_incorrect_background to be 0"


from src.utils import calculate_iou


def test_num_incorrect_localization(sample_yolo_data_2):
    """
    one correct bbox and one incorrect bbox due to localization
    """

    output_tensor = sample_yolo_data_2["yolo_net_output_tensor"]
    target_tensor = sample_yolo_data_2["yolo_net_target_tensor"]

    # slight offset in the bounding box x and y
    output_tensor[0, 2, 2, 0] += 0.2
    output_tensor[0, 2, 2, 1] += 0.2

    ious_per_cell = calculate_iou(
        output_tensor[:, :, :, :4], target_tensor[:, :, :, :4]
    )

    print("this is ious_per_cell", ious_per_cell[0, 2, 2])

    assert (
        0.1 < ious_per_cell[0, 2, 2] < 0.5
    ), "Expected iou to be between 0.1 and 0.5 for the rest of this test to work"

    metrics = get_yolo_metrics(
        sample_yolo_data_2["yolo_net_output_tensor"],
        sample_yolo_data_2["yolo_net_target_tensor"],
    )

    # one correct prediction, one incorrect prediction

    assert metrics["num_correct"] == 1, "Expected num_correct to be 1"

    assert (
        metrics["num_incorrect_localization"] == 1
    ), "Expected num_incorrect_localization to be 1"
    assert metrics["num_incorrect_other"] == 0, "Expected num_incorrect_other to be 0"
    assert (
        metrics["num_incorrect_background"] == 0
    ), "Expected num_incorrect_background to be 0"


def test_num_incorrect_other(sample_yolo_data_2):
    """
    one correct bbox and one incorrect bbox due to class probability
    """

    output_tensor = sample_yolo_data_2["yolo_net_output_tensor"]
    target_tensor = sample_yolo_data_2["yolo_net_target_tensor"]

    # switch the class probabilities label
    index = target_tensor[0, 2, 2, 10:].argmax()
    output_tensor[0, 2, 2, 10 + index] = 0.0
    output_tensor[0, 2, 2, 10 + index - 1] = 1.0

    metrics = get_yolo_metrics(
        sample_yolo_data_2["yolo_net_output_tensor"],
        sample_yolo_data_2["yolo_net_target_tensor"],
    )

    # one correct prediction, one incorrect prediction

    assert metrics["num_correct"] == 0, "Expected num_correct to be 1"

    assert (
        metrics["num_incorrect_localization"] == 0
    ), "Expected num_incorrect_localization to be 1"
    assert metrics["num_incorrect_other"] == 2, "Expected num_incorrect_other to be 0"
    assert (
        metrics["num_incorrect_background"] == 0
    ), "Expected num_incorrect_background to be 0"


def test_num_incorrect_background(sample_yolo_data_2):
    """
    one correct bbox and one incorrect bbox due to background
    """

    output_tensor = sample_yolo_data_2["yolo_net_output_tensor"]
    target_tensor = sample_yolo_data_2["yolo_net_target_tensor"]

    # slight offset in the bounding box x and y
    output_tensor[0, 2, 2, 0] += 0.4
    output_tensor[0, 2, 2, 1] += 0.4

    ious_per_cell = calculate_iou(
        output_tensor[:, :, :, :4], target_tensor[:, :, :, :4]
    )

    assert (
        ious_per_cell[0, 2, 2] < 0.1
    ), "Expected iou to be between 0.1 and 0.5 for the rest of this test to work"

    metrics = get_yolo_metrics(
        sample_yolo_data_2["yolo_net_output_tensor"],
        sample_yolo_data_2["yolo_net_target_tensor"],
    )

    # one correct prediction, one incorrect prediction

    assert metrics["num_correct"] == 1, "Expected num_correct to be 1"

    assert (
        metrics["num_incorrect_localization"] == 0
    ), "Expected num_incorrect_localization to be 1"
    assert metrics["num_incorrect_other"] == 0, "Expected num_incorrect_other to be 0"
    assert (
        metrics["num_incorrect_background"] == 1
    ), "Expected num_incorrect_background to be 0"


# test: see if num_incorrect_background is calculated correctly
