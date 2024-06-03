import os
import mlflow
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
from typing import Dict


class RunManager:
    '''
    The job of the run manager is to manage experiment runs. It integrates
    '''

    def __init__(self, run_dir: str, run_id: str):
        self.run_id = run_id
        self.writer = SummaryWriter(log_dir=f"{run_dir}/tensorboard/{run_id}")
        self.checkpoints_dir = Path(f"{run_dir}")

        self.checkpoints_dir.mkdir(parents=True,
                                   exist_ok=True)

    def track_metrics(self, metrics: Dict[str, Dict[str, float]], epoch: int) -> None:
        """
        Track metrics for the run and plot it on tensorboard.

        Args:
          metrics: a dictionary of metrics to track.

        Example:
          metrics = {
            'loss': {'train': 0.1, 'val': 0.2},
            'accuracy': {'train': 0.9, 'val': 0.8}
          }
        """

        for metric_name, metric_dict in metrics.items():
            self.writer.add_scalars(metric_name, metric_dict, epoch)
            print(f"\nEpoch: {epoch}, {metric_name}: {metric_dict}")

            # writer.add_scalar('training/train_loss', train_loss, epoch)
        # writer.add_scalar('training/val_loss', test_loss, epoch)
    def end_run(self) -> None:
        self.writer.close()

    def save_model(self, model: torch.nn.Module, epoch: int) -> None:
        """Saves a PyTorch model to a target directory.

        Args:
          model: A target PyTorch model to save.
          epoch: The epoch number to save the model at.

        Example usage:
          save_model(model=model_0, epoch=5)
        """
        model_save_path = self.checkpoints_dir / f"{epoch}.pth"

        # Save the model state_dict()
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(),
                   f=model_save_path)
        mlflow.log_artifact(str(model_save_path), artifact_path="checkpoints")
        os.remove(model_save_path)

    def load_checkpoint_if_it_exists(self, model: torch.nn.Module, checkpoint_path: str) -> int:
        """
        Loads a PyTorch model weights from a run at an epoch.
        Args:
          model: A target PyTorch model to load.
          epoch: The epoch number to load the model from.

        Example usage:
          load_model(model=model_0, epoch=5)
        """
        if checkpoint_path is None:
            # checkpoint_path = self.checkpoints_dir / f"{self.run_id}.pth"
            print(f"[INFO] Not loading model")
            return 0

        if not Path(checkpoint_path).exists():
            print(f"[INFO] Checkpoint not found at: {checkpoint_path}")
            return 0

        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        checkpoint_run_id = checkpoint_path.split("/")[-2]
        epoch_start = int(checkpoint_path.split("/")[-1].split(".")[0])
        params = torch.load(checkpoint_path)
        model.load_state_dict(params)
        return epoch_start
