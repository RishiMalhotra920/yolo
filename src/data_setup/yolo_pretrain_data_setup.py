import torchvision.datasets as datasets
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2

config = yaml.safe_load(open("config.yaml"))


def create_datasets(
    root_dir: str, transform: transforms_v2._container.Compose
) -> tuple[Dataset, Dataset]:
    train_dataset = datasets.ImageNet(
        root=root_dir, split="train", transform=transform, target_transform=None
    )

    val_dataset = datasets.ImageNet(
        root=root_dir, split="val", transform=transform, target_transform=None
    )
    return train_dataset, val_dataset


def create_dataloaders(
    root_dir: str,
    transform: transforms_v2._container.Compose,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(root_dir, transform)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_data_loader, val_data_loader


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    data_transform = transforms_v2.Compose(
        [
            # transforms.RandomResizedCrop(50),
            transforms_v2.Resize((224, 224)),
            transforms_v2.RandomHorizontalFlip(),
            transforms_v2.RandomRotation((-30, 30)),
            transforms_v2.ColorJitter(brightness=0.5, contrast=0.5),
            transforms_v2.ToTensor(),
            transforms_v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            # transforms.RandomErasing()
        ]
    )

    train_dataset, _ = create_datasets(config["image_net_data_dir"], data_transform)

    print(train_dataset[0])
