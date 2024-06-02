import argparse
import os
import torch
import data_setup
import engine
import model_builder
from torchvision import transforms

# to call this script, run the following command:
# python train.py --num_epochs 10 --batch_size 32 --hidden_units 128 --learning_rate 0.01 --run_id trial_run_with_128_hidden_units


def train(args) -> None:
    # Setup hyperparameters
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

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

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 val_dataloader=test_dataloader,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 epochs=args.num_epochs,
                 run_dir=args.run_dir,
                 run_id=args.run_id,
                 continue_from_checkpoint={
                     "run_id": args.continue_from_checkpoint_run_id, "epoch": args.continue_from_checkpoint_epoch},
                 num_checkpoints=2,
                 device=args.device)


if __name__ == "__main__":
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

    parser.add_argument('--continue_from_checkpoint_run_id', type=str, default=None,
                        help='Run ID to continue training from')
    parser.add_argument('--continue_from_checkpoint_epoch', type=int, default=None,
                        help='Epoch to continue training from')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to train the model on')
    parser.add_argument('--train_dir', type=str, default="/Users/rishimalhotra/projects/cv/image_classification/image_net_data/train",
                        help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default="/Users/rishimalhotra/projects/cv/image_classification/image_net_data/val",
                        help='Directory containing validation data')
    parser.add_argument('--run_dir', type=str, default="/Users/rishimalhotra/projects/cv/image_classification/repo_hub/training_repo/runs",
                        help='Directory to store runs')

    args = parser.parse_args()
    train(args)
