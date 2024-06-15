import argparse
import os

import data_setup
import engine
import model_builder
import torch
import yaml
from lr_scheduler import get_custom_lr_scheduler, get_fixed_lr_scheduler
from run_manager import RunManager, load_checkpoint
from torchvision import transforms

config = yaml.safe_load(open("config.yaml"))

# to call this script, run the following command:
# start with learning rate 0.01 to speed the fuck out of the training. if it starts to bounce around, then we can decrease it.
# python train.py --num_epochs 10 --batch_size 32 --hidden_units 128 --learning_rate 0.01 --run_name cpu_run_on_image_net

# GPU training command:
# python train.py --num_epochs 50 --batch_size 128 --hidden_units 256 --learning_rate 0.001 --run_name cuda_run_with_256_hidden_units --device cuda

# python train.py --num_epochs 100 --batch_size 1024 --hidden_units 256 --learning_rate 0.005 --run_name image_net_train_deeper_network_and_dropout --device cuda


def train(args) -> None:

    # Setup target device
    assert args.device in ["cpu", "cuda"], "Invalid device"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise Exception("CUDA is not available on this device")

    # yolo uses 448x448 images
    # Create transforms
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

    cpu_count = os.cpu_count()
    num_workers = cpu_count if cpu_count is not None else 0

    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        root_dir=config["image_net_data_dir"],
        transform=data_transform,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    # print('input data shape is', next(iter(train_dataloader))[0].shape)
    # input()

    model = model_builder.DeepConvNet(dropout=args.dropout).to(args.device)

    run_manager = RunManager(new_run_name=args.run_name)
    if args.continue_from_checkpoint_signature is not None:
        epoch_start = load_checkpoint(
            model, checkpoint_signature=args.continue_from_checkpoint_signature)
        run_manager.add_tags(
            ["run_continuation"])
        run_manager.set_checkpoint_to_continue_from(
            args.continue_from_checkpoint_signature)
    else:
        epoch_start = 0
        run_manager.add_tags(["new_run"])

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    if args.lr_scheduler == "custom":
        lr_scheduler = get_custom_lr_scheduler(optimizer)
    else:
        lr_scheduler = get_fixed_lr_scheduler(optimizer)

    for epoch in range(epoch_start):
        # step the lr_scheduler to match with the current_epoch
        lr_scheduler.step()
        print("epoch", epoch, "lr", optimizer.param_groups[0]['lr'])

    print('this is lr_scheduler current lr', optimizer.param_groups[0]['lr'])
    # input()

    parameters = {
        "num_epochs": args.num_epochs,
        "lr_scheduler": args.lr_scheduler,
        "batch_size": args.batch_size,
        "loss_fn": "CrossEntropyLoss",
        "optimizer": "Adam",
        "device": args.device,
        "dropout": args.dropout,
    }

    run_manager.log_data({"parameters": parameters,
                          "model/summary": str(model),
                          })

    run_manager.log_files({"model/code": "model_builder.py",
                           "lr_scheduler/code": "lr_scheduler.py",
                           })

    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 val_dataloader=test_dataloader,
                 lr_scheduler=lr_scheduler,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 epoch_start=epoch_start,
                 epoch_end=epoch_start + args.num_epochs,
                 k_top=5,
                 run_manager=run_manager,
                 checkpoint_interval=args.checkpoint_interval,
                 log_interval=args.log_interval,
                 device=args.device)

    run_manager.end_run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Computer Vision Model Trainer',
        description='Trains a computer vision model for image classification',
        epilog='Enjoy the program! :)')

    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for training the model')
    parser.add_argument('--lr_scheduler', type=str, required=True,
                        help='Scheduler for the optimizer or custom')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate for fixed scheduler or starting learning rate for custom scheduler')
    parser.add_argument('--dropout', type=float, required=True,
                        help='Dropout rate for the model')
    # parser.add_argument('--num_workers', type=int, required=True,
    # help='Number of workers for the dataloader. Eight for your macbook. Number of cores for your gpu')

    parser.add_argument('--run_name', type=str, required=False,
                        help='A name for the run')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='The number of epochs to wait before saving model checkpoint')
    parser.add_argument('--continue_from_checkpoint_signature', type=str, default=None,
                        help='Checkpoint signature for the run continue training from in the format: RunId:CheckpointPath eg: IM-23:checkpoints/epoch_10.pth')

    parser.add_argument('--log_interval', type=int, default=10,
                        help='The number of batches to wait before logging training status')

    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to train the model on')
    parser.add_argument('--train_dir', type=str, default=f'{config["image_net_data_dir"]}/train',
                        help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default=f'{config["image_net_data_dir"]}/val',
                        help='Directory containing validation data')

    args = parser.parse_args()

    # must have a run_name for all runs
    assert args.run_name is not None
    assert args.lr_scheduler in ["custom", "fixed"], "Invalid lr_scheduler"

    try:
        if args.run_name is None:
            inp = input(
                f"Confirm that you want to continue training from {args.continue_from_checkpoint_run_id}:{args.continue_from_checkpoint_path} and log the run to {args.continue_from_checkpoint_run_id}: yes or no: ")
        else:
            inp = input(
                f"Confirm that run_name is {args.run_name}: yes or no: ")

        if inp.lower() != "yes":
            raise Exception("Type yes on input...")

        print("Starting training for run_name:", args.run_name)
        train(args)

    except KeyboardInterrupt:
        # without this: weird issues where KeyboardInterrupt causes a ton of torch_smh_manager processes that never close.
        print("Training interrupted by user")
