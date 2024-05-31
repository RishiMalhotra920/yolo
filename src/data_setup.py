from typing import cast
import torchvision
from torch.utils.data import DataLoader
import random
from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader


from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        # super().__init__()
        pass
        # Initialize common properties if any

    def __len__(self):
        # raise NotImplementedError
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class SubsetImageFolder(BaseDataset):
    def __init__(self, root: str, classes: list[str], num_samples_per_class: int, transform: torchvision.transforms.Compose):
        """
        root: Root directory path.
        classes: List of class folders to include.
        num_samples: Number of samples to include from each class.
        transform: Pytorch transforms for preprocessing the data.
        """
        self.root = root
        self.transform = transform
        self.dataset = []

        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                images = [os.path.join(class_path, img) for img in os.listdir(
                    class_path) if img.endswith('.JPEG')]
                sampled_images = random.sample(images, num_samples_per_class)
                for image in sampled_images:
                    self.dataset.append((image, class_idx))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.dataset[idx]
        image = Image.open(image_path).convert('RGB')  # Convert image to RGB
        image = self.transform(image)
        image = cast(torch.Tensor, image)

        return image, label


meta = torch.load('/Users/rishimalhotra/projects/cv/image_net_data/meta.bin')


def get_class_names_from_folder_names(classes: list[str]):
    return [meta[0][s][0] for s in classes]


def create_mini_datasets(train_dir: str, val_dir: str, classes: list[str], transform: torchvision.transforms.Compose):
    mini_train_dataset = SubsetImageFolder(
        root=train_dir, classes=classes, num_samples_per_class=1000, transform=transform)
    mini_val_dataset = SubsetImageFolder(
        root=val_dir, classes=classes, num_samples_per_class=50, transform=transform)

    assert len(mini_train_dataset) > 0, "Training dataset is empty"
    assert len(mini_val_dataset) > 0, "Validation dataset is empty"

    return mini_train_dataset, mini_val_dataset


def create_mini_dataloaders(train_dir: str, val_dir: str, classes: list[str], transform: torchvision.transforms.Compose, batch_size: int, num_workers: int):

    mini_train_dataset, mini_val_dataset = create_mini_datasets(
        train_dir, val_dir, classes, transform)

    class_names = get_class_names_from_folder_names(classes)

    mini_train_data_loader = DataLoader(
        mini_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    mini_val_data_loader = DataLoader(
        mini_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return mini_train_data_loader, mini_val_data_loader, class_names
