# from src.loss_functions.yolo_loss_function import calculate_iou
from src.utils import transform_yolo_grid_into_bboxes_confidences_and_labels


def test_transform_preds_into_bboxes(sample_yolo_data_1):
    """ """

    target_bboxes = sample_yolo_data_1["bboxes"].tolist()
    image_width = sample_yolo_data_1["width"]
    image_height = sample_yolo_data_1["height"]

    bboxes, confidences, labels = (
        transform_yolo_grid_into_bboxes_confidences_and_labels(
            sample_yolo_data_1["yolo_net_target_tensor"].squeeze(),
            image_width,
            image_height,
        )
    )

    # get bboxes where confidence is greater than 0

    pred_bboxes = [
        bbox for bbox, confidence in zip(bboxes, confidences) if confidence > 0
    ]

    rounded_pred_bboxes = []

    for bbox in pred_bboxes:
        rounded_pred_bbox = [round(x, 1) if isinstance(x, float) else x for x in bbox]

        rounded_pred_bboxes.append(rounded_pred_bbox)

    print("rounded_pred_bboxes", rounded_pred_bboxes)
    print("target_bboxes", target_bboxes)

    assert rounded_pred_bboxes == target_bboxes, "pred and target bboxes do not align"
