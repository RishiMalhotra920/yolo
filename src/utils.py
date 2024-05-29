from data_setup import BaseDataset
from typing import Optional
from pathlib import Path
import random
import matplotlib.pyplot as plt
import torch


def count_top_k_correct(output: torch.Tensor, target: torch.Tensor, k: int):
    """Compute top-k accuracy for the given predictions and labels.

    Args:
        output (torch.Tensor): The logits or probabilities from the model.
        target (torch.Tensor): The true labels for each input.
        k (int): The top 'k' predictions considered to calculate the accuracy.

    Returns:
        float: The top-k accuracy.
    """
    # Get the top k predictions from the model for each input
    _, predicted = output.topk(k, 1, True, True)

    # View target to make it [batch_size, 1]
    target = target.view(-1, 1)

    # Check if the true labels are in the top k predictions
    correct = predicted.eq(target).sum().item()

    # Calculate the accuracy
    return correct


def display_random_images(dataset: BaseDataset,
                          *,
                          class_names: list[str] | None = None,
                          n: int = 5,
                          seed: int = 42) -> None:

    random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(15, 5))

    for idx, random_sample_idx in enumerate(random_samples_idx):
        image, label = dataset[random_sample_idx]
        ax = axes[idx]
        ax.imshow(image.permute(1, 2, 0))
        if class_names:
            ax.set_title(f"Label: {label},\nclass:{class_names[label]}")
        else:
            ax.set_title(f"Label:{label}")
    # ax[1].imshow(image_net_val[0][0].permute(1, 2, 0))
    # ax[1].set_title('Validation image')
        ax.axis('off')
    plt.show()


def predict_on_random_images(model, dataset, *, class_names=None, n=5, seed=None):
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(15, 3))

    images = []
    for i, index in enumerate(random_samples_idx):
        image, label = dataset[index]
        ax = axes[i]
        pred_logits = model(image.unsqueeze(0))
        print('pred_logits', pred_logits)
        pred = torch.argmax(pred_logits.squeeze()).item()
        print('pred', pred)

        ax.imshow(image.permute(1, 2, 0))
        if class_names:
            ax.set_title(
                f"Label: {label} {class_names[label]}\nPred: {pred} {class_names[pred]}")
        else:
            ax.set_title(f"Label: {label} Pred: {pred}")

        ax.axis('off')
    plt.show()


# def save_model(model: torch.nn.Module,
#                target_dir: str,
#                model_name: str):
#     """Saves a PyTorch model to a target directory.

#     Args:
#       model: A target PyTorch model to save.
#       target_dir: A directory for saving the model to.
#       model_name: A filename for the saved model. Should include
#         either ".pth" or ".pt" as the file extension.

#     Example usage:
#       save_model(model=model_0,
#                  target_dir="models",
#                  model_name="05_going_modular_tingvgg_model.pth")
#     """
#     # Create target directory
#     target_dir_path = Path(target_dir)
#     target_dir_path.mkdir(parents=True,
#                           exist_ok=True)

#     # Create model save path
#     assert model_name.endswith(".pth") or model_name.endswith(
#         ".pt"), "model_name should end with '.pt' or '.pth'"
#     model_save_path = target_dir_path / model_name

#     # Save the model state_dict()
#     print(f"[INFO] Saving model to: {model_save_path}")
#     torch.save(obj=model.state_dict(),
#                f=model_save_path)


# def load_model(model, target_dir, model_name):
#     """Loads a PyTorch model from a target directory."""

#     print(f"[INFO] Loading model from: {model_save_path}")

#     model_save_path = Path(target_dir) / model_name

#     params = torch.load(model_save_path)
#     model.load_state_dict(params)
