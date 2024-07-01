# TODO: figure out how to run pytests here.
# and do that


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa: E402


from src.data_setup.yolo_train_data_setup import yolo_target_transform
from src.utils import class_to_index


def test_yolo_target_transform():
    sample_annotation = {
        "annotation": {
            "folder": "VOC2012",
            # "filename": "2008_000008.jpg",
            "source": {
                "database": "The VOC2008 Database",
                "annotation": "PASCAL VOC2008",
                "image": "flickr",
            },
            "size": {"width": "1344", "height": "896", "depth": "3"},
            "segmented": "0",
            "object": [
                {
                    "name": "horse",
                    "pose": "Left",
                    "truncated": "0",
                    "occluded": "1",
                    "bndbox": {
                        "xmin": "105",
                        "ymin": "84",
                        "xmax": "798",
                        "ymax": "504",
                    },
                    "difficult": "0",
                },
                {
                    "name": "horse",
                    "pose": "Left",
                    "truncated": "0",
                    "occluded": "1",
                    "bndbox": {
                        "xmin": "110",
                        "ymin": "94",
                        "xmax": "798",
                        "ymax": "504",
                    },
                    "difficult": "0",
                },
            ],
        }
    }

    target = yolo_target_transform(sample_annotation)
    print("target shape", target.shape)
    expected_tensor_with_object_info = target[2, 2, :].tolist()
    print("expected_tensor_with_object_info", expected_tensor_with_object_info)

    one_hot = [0.0 for _ in range(20)]
    one_hot[class_to_index["horse"]] = 1

    print("")

    assert (
        expected_tensor_with_object_info
        == [
            0.3515625,
            0.296875,
            0.515625,
            0.46875,
            1.0,
            0.3645833432674408,
            0.3359375,
            0.511904776096344,
            0.4575892984867096,
            1.0,
        ]
        + one_hot
    )

    print("Test passed")
    # print("image shape", train_dataset[0][0].shape)
    # print("target", train_dataset[0][1])


if __name__ == "__main__":
    test_yolo_target_transform()
