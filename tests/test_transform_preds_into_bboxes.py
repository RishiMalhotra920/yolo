# from src.loss_functions.yolo_loss_function import calculate_iou
from src.utils import transform_preds_into_bboxes


def test_transform_preds_into_bboxes(sample_yolo_data_1):
    """ """
    annotations = sample_yolo_data_1["annotation"]

    image_width = float(annotations["annotation"]["size"]["width"])
    image_height = float(annotations["annotation"]["size"]["height"])

    bboxes = transform_preds_into_bboxes(
        sample_yolo_data_1["yolo_net_target_tensor"], image_width, image_height
    )

    # get bboxes where confidence is greater than 0

    pred_bboxes = [bbox for bbox in bboxes if bbox[4] > 0]

    target_bboxes = []

    for bbox_dict in annotations["annotation"]["object"]:
        bbox = bbox_dict["bndbox"]
        bbox = (
            float(bbox["xmin"]),
            float(bbox["ymin"]),
            float(bbox["xmax"]) - float(bbox["xmin"]),
            float(bbox["ymax"]) - float(bbox["ymin"]),
            1.0,
            bbox_dict["name"],
        )

        target_bboxes.append(bbox)

    rounded_pred_bboxes = []

    for bbox in pred_bboxes:
        rounded_pred_bbox = tuple(
            round(x, 1) if isinstance(x, float) else x for x in bbox
        )
        rounded_pred_bboxes.append(rounded_pred_bbox)

    print(type(pred_bboxes[0][0]))

    assert rounded_pred_bboxes == target_bboxes, "pred and target bboxes do not align"
