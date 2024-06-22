import argparse

import torch
import yaml
from data_setup import yolo_pretrain_data_setup
from models import yolo_net
from run_manager import load_checkpoint
from torchvision import transforms
from utils import display_random_images, predict_on_random_images

config = yaml.safe_load(open("config.yaml"))


def visualize(args):

    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(50),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing()
    ])

    to_tensor_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    classes = ["n01986214", "n02009912", "n01924916"]
    # class_names = get_class_names_from_folder_names(classes)

    train_dataset, mini_val_dataset = yolo_pretrain_data_setup.create_datasets(
        config["image_net_data_dir"],
        data_transform)

    # model = model_builder.DeepConvNet(
    # hidden_units=args.hidden_units,
    # output_shape=len(classes)
    # ).to("cpu")

    model = yolo_net.YOLOPretrainNet(dropout=0).to("cpu")
    model = torch.nn.DataParallel(model)

    load_checkpoint(model, args.checkpoint_signature)

    predict_on_random_images(model, mini_val_dataset, n=10, seed=420)

    # display_random_images(train_dataset,
    #   class_names=class_names, n=5, seed=4)


if __name__ == "__main__":
    # for example
    # python visualize.py --checkpoint_signature=IM-122:checkpoints/epoch_14

    # --hidden_units 256
    parser = argparse.ArgumentParser(
        description="Visualize the model's predictions")
    # parser.add_argument("--hidden_units", type=int,
    # help="The number of hidden units", required=True)

    parser.add_argument("--checkpoint_signature", type=str,
                        help="The signature for the checkpoint", required=True)
    args = parser.parse_args()
    visualize(args)
