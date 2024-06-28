import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


# from src.loss_functions.yolo_loss_function import \
# convert_annotations_to_tensors
import torch
import yaml
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2 as transforms_v2

from src.data_setup.yolo_train_data_setup import create_datasets
from src.loss_functions.yolo_loss_function import YOLOLoss

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    root_dir = config["pascal_voc_root_dir"]
    temp_dataset = VOCDetection(
        root=root_dir, year="2012", image_set="train", download=False
    )

    _, val_dataset = create_datasets(
        root_dir,
        transforms_v2.Compose([transforms_v2.ToTensor()]),
    )

    sample_label = val_dataset[11][1].unsqueeze(0)
    sample_pred = torch.randn((1, 7, 7, 30))

    print(f"sample_label shape: {sample_label.shape}")
    print(f"sample_pred shape: {sample_pred.shape}")
    loss_fn = YOLOLoss()
    loss = loss_fn(sample_pred, sample_label)
    print(loss)

    # print('this', sample_ann.shape)
    # print(sample_ann)
    # converted_anns = convert_annotations_to_tensors(
    # sample_ann, class_to_index=class_to_index)

    # print(converted_anns)
