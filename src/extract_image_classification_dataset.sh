mkdir dataset
cd dataset
echo "Curling the train dataset"
curl --progress-bar -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
echo "Curling the val dataset"
curl --progress-bar -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
echo "Curling the devkit"
curl --progress-bar -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
echo "pip installing"
pip install torch torchvision scipy
echo "writing to file"
cat > open_up.py << EOF
import torchvision
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

def create_datasets(root_dir: str) -> tuple[Dataset, Dataset]:
    train_dataset = datasets.ImageNet(root=root_dir,
                                      split='train',
                                      target_transform=None)

    val_dataset = datasets.ImageNet(root=root_dir,
                                    split='val',
                                    target_transform=None)
    return train_dataset, val_dataset


create_datasets("/root/dataset")
EOF
echo "Unzipping the tarred files"
python open_up.py
echo "Dataset is ready"