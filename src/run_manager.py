import io
import os
import torch
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import PIL.Image as Image
import neptune
import yaml
import shutil
config = yaml.safe_load(open("config.yaml"))


class RunManager:
    '''
    The job of the run manager is to manage experiment runs. It integrates
    '''

    def __init__(self, run_name: str):
        self.run_id = run_name
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

        self.run = neptune.init_run(
            project="towards-hi/image-classification",
            api_token=config["neptune_api_token"],
            name=run_name
        )

    def log_data(self, data: Dict[str, Any]) -> None:
        """
        Log data to the run.

        Args:
          data: a dictionary of data to log.

        Example:
          data = {
            "num_epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "hidden_units": 512,
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "Adam",
            "device": "cuda"
          }
        """
        for key in data:
            self.run[key] = data[key]

    def log_files(self, files: Dict[str, str]) -> None:
        """
        Log files to the run.

        Args:
          files: a dictionary of files to log.

        Example:
          files = {
            "model/code": "model_builder.py"
          }
        """
        for key in files:
            self.run[key].upload(files[key])

    def log_metrics(self, metrics: Dict[str, float], epoch: float) -> None:
        """
        Track metrics for the run and plot it on neptune.

        epoch can be a float - a float epoch denotes that we are logging a fraction of the way through the epoch

        Args:
          metrics: a dictionary of metrics to track.

        Example:
          metrics = {
            "train/loss": 0.5,
            "val/loss": 0.3,
            "train/accuracy": 0.8,
            "val/accuracy": 0.9
          }
        """

        for metric_name in metrics:
            self.run[metric_name].append(metrics[metric_name], step=epoch)
            # print(f"\nEpoch: {epoch}, {metric_name}: {metrics[metric_name]}")

    def end_run(self):
        self.run.stop()
        try:
            shutil.rmtree(self.temp_dir)

        except Exception as e:
            print(f"Failed to remove directory: {e}")

    def save_model(self, model: torch.nn.Module, epoch: int) -> None:
        """Saves a PyTorch model to a target directory.

        Args:
          model: A target PyTorch model to save.
          epoch: The epoch number to save the model at.

        Example usage:
          save_model(model=model_0, epoch=5)
        """
        # Note that model should be saved as epoch_{epoch}.pth
        # to add more info, do this: epoch_{epoch}_lr_{lr}_bs_{bs}.pth
        # later on you can implement a json file to store info about checkpoints

        model_save_path = self.temp_dir / f"{epoch}.pth"
        print(f"[INFO] Saving model to {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)
        self.run[f"checkpoints/epoch_{epoch}"].upload(str(model_save_path))


def load_checkpoint(model: torch.nn.Module, run_id: str, checkpoint_path: str) -> int:
    """
    Loads a PyTorch model weights from a run at an epoch.
    Args:
        model: A target PyTorch model to load.
        epoch: The epoch number to load the model from.

    Example usage:
        load_model(model=model_0, epoch=5)
    """
    assert not checkpoint_path.endswith(
        ".pth"), "checkpoint_path should not end with .pth"

    # save to temp_dir/{file_name}.pth
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    run = neptune.init_run(
        project="towards-hi/image-classification",
        with_id=run_id,
        mode="read-only",
        api_token=config["neptune_api_token"])
    run[checkpoint_path].download(destination=str(temp_dir))

    # load from temp_dir/{file_name}.pth into model
    file_name = checkpoint_path.split("/")[-1]
    params = torch.load(temp_dir/f"{file_name}.pth",
                        map_location=torch.device('cpu'))
    model.load_state_dict(params)

    # file_name should be epoch_{epoch}.pth
    epoch_number = int(file_name.split("_")[1])

    print('this is epoch_number', epoch_number)

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Failed to remove directory: {e}")

    return epoch_number
