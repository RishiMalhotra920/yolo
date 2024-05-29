import os
import torch
import data_setup
import engine
import model_builder
import utils

from torchvision import transforms

# TODO: create this wrapper class
# TODO: tbh there should be a class called
# run that stores info about each run, logs to tensorboard, manages checkpoints etc.


def train() -> None:
    # Setup hyperparameters
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "../image_net_data/train"
    val_dir = "../image_net_data/val"
    run_dir = "/Users/rishimalhotra/projects/cv/src/runs/"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(50),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ])

    classes = ["n01986214", "n02009912", "n01924916"]

    cpu_count = os.cpu_count()
    num_workers = cpu_count if cpu_count is not None else 0

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_mini_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        classes=classes,
        transform=data_transform,
        batch_size=BATCH_SIZE,
        num_workers=num_workers
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 val_dataloader=test_dataloader,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 epochs=NUM_EPOCHS,
                 run_id="testing33",
                 continue_from_checkpoint=None,
                 num_checkpoints=2,
                 device=device)


if __name__ == "__main__":
    train()
