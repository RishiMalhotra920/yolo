import shutil
from pathlib import Path

import neptune
import torch
import yaml

config = yaml.safe_load(open("config.yaml"))


def download_checkpoint(checkpoint_signature: str) -> tuple[Path, str]:
    """
    Downloads a checkpoint from Neptune.

    Args:
        run_id: The Neptune run id.
        checkpoint_path: The checkpoint path.

    Example usage:
        download_checkpoint(run_id="IM-23", checkpoint_path="checkpoints/epoch_10.pth")
    """
    run_id, checkpoint_path = checkpoint_signature.split(":")
    assert not checkpoint_path.endswith(
        ".pth"
    ), "checkpoint_path should not end with .pth"

    # save to temp_dir/{file_name}.pth
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    run = neptune.init_run(
        project="towards-hi/image-classification",
        with_id=run_id,
        mode="read-only",
        api_token=config["neptune_api_token"],
    )
    run[checkpoint_path].download(destination=str(temp_dir))

    # load from temp_dir/{file_name}.pth into model
    file_name = checkpoint_path.split("/")[-1]

    return temp_dir, file_name


def delete_directory(temp_dir: Path) -> None:
    """
    Deletes a checkpoint from the file system.

    Args:
        run_id: The Neptune run id.
        checkpoint_path: The checkpoint path.

    Example usage:
        delete_checkpoint_from_file_system(run_id="IM-23", checkpoint_path="checkpoints/epoch_10.pth")
    """
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Failed to remove directory: {e}")


def load_checkpoint(model: torch.nn.Module, checkpoint_signature: str) -> int:
    """
    Loads a PyTorch model weights from a run at an epoch.
    Args:
        model: A target PyTorch model to load.
        epoch: The epoch number to load the model from.

    Example usage:
        load_model(model=model_0, epoch=5)
    """
    assert (
        ":" in checkpoint_signature
    ), "checkpoint_signature should be in the format RunId:CheckpointPath"

    temp_dir, file_name = download_checkpoint(checkpoint_signature)

    params = torch.load(temp_dir / f"{file_name}.pth", map_location=torch.device("cpu"))
    model.load_state_dict(params)
    # file_name should be epoch_{epoch}.pth
    epoch_number = int(file_name.split("_")[1])

    # we save logs for the epoch number that was completed
    # we should start logging from the next epoch
    start_epoch = epoch_number + 1

    print("this is epoch_number", epoch_number)

    delete_directory(temp_dir)

    return start_epoch


def load_checkpoint_for_yolo_from_pretrained_image_net_model(
    model: torch.nn.Module, checkpoint_signature: str
) -> None:
    """
    the weights loaded in are for nn.DataParallel(FeatureExtractor -> YOLOPretrainNet)
    the model in parameters is nn.DataParallel(FeatureExtractor -> YOLONet)
    load the weights for Feature Extractor from pretrained model and random initialize the rest
    """
    temp_dir, file_name = download_checkpoint(checkpoint_signature)
    params = torch.load(temp_dir / f"{file_name}.pth", map_location=torch.device("cpu"))
    model_dict = model.state_dict()
    # model.load_state_dict(params)
    # 1. filter out keys in pretrained model not in current model
    pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
    print(
        f"image net model keys: {params.keys()}\n yolo model keys: {model_dict.keys()}\n keys loading: {pretrained_dict.keys()}"
    )
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print(f"[INFO] Loaded model from {checkpoint_signature}")

    delete_directory(temp_dir)
