from run_manager import RunManager
import mlflow
import argparse
import os
import torch
import data_setup
import engine
import model_builder
from torchvision import transforms
import yaml
config = yaml.safe_load(open("config.yaml"))

# to call this script, run the following command:
# python train.py --num_epochs 10 --batch_size 32 --hidden_units 128 --learning_rate 0.001 --run_id trial_run_with_128_hidden_units


def train(args) -> None:

    # Setup target device
    assert args.device in ["cpu", "cuda"], "Invalid device"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise Exception("CUDA is not available on this device")

    # Create transforms
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(50),
        transforms.Resize((50, 50)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing()
    ])

    classes = ["n01986214", "n02009912", "n01924916"]

    cpu_count = os.cpu_count()
    num_workers = cpu_count if cpu_count is not None else 0

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_mini_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        classes=classes,
        transform=data_transform,
        batch_size=args.batch_size,
        num_workers=num_workers
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(args.device)

    run_manager = RunManager(args.run_dir, args.run_id)
    epoch_start = run_manager.load_checkpoint_if_it_exists(
        model, checkpoint_path=args.continue_from_checkpoint_path)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)

    with mlflow.start_run(run_name=args.run_id):
        params = {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "hidden_units": args.hidden_units,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "Adam",
            "device": args.device,
        }
        mlflow.log_params(params)

        with open("model_summary.txt", "w") as f:
            f.write(str(model))

        mlflow.log_artifact("model_builder.py")
        mlflow.log_artifact("model_summary.txt")
        os.remove("model_summary.txt")

        engine.train(model=model,
                     train_dataloader=train_dataloader,
                     val_dataloader=test_dataloader,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     epoch_start=epoch_start,
                     epoch_end=epoch_start + args.num_epochs,
                     run_manager=run_manager,
                     checkpoint_interval=args.num_epochs // 5,
                     device=args.device)


if __name__ == "__main__":
    mlflow.set_tracking_uri(config["mlflow_uri"])

    parser = argparse.ArgumentParser(
        prog='Computer Vision Model Trainer',
        description='Trains a computer vision model for image classification',
        epilog='Enjoy the program! :)')

    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for training the model')
    parser.add_argument('--hidden_units', type=int, required=True,
                        help='Number of hidden units in the model')
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='Learning rate for the optimizer')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Unique identifier for the run')

    parser.add_argument('--continue_from_checkpoint_path', type=str, default=None,
                        help='Run ID to continue training from')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to train the model on')
    parser.add_argument('--train_dir', type=str, default=config["image_net_train_data_path"],
                        help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default=config["image_net_val_data_path"],
                        help='Directory containing validation data')
    parser.add_argument('--run_dir', type=str, default=config["run_dir"],
                        help='Directory to store runs')

    args = parser.parse_args()

    inp = input(f"Confirm that run_id is {args.run_id}: yes or no: ")

    if inp.lower() != "yes":
        raise Exception("Type yes on input...")

    print("Starting training for run_id:", args.run_id)

    train(args)
