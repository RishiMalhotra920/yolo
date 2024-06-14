mkdir image_net_data
cd image_net_data
echo "Start time: $(date)"
echo "Curling the train dataset"
curl --progress-bar -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
echo "Time after curling the train dataset: $(date)"
echo "Curling the val dataset"
curl --progress-bar -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
echo "Time after curling the val dataset: $(date)"
echo "Curling the devkit"
curl --progress-bar -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
echo "Time after curling the devkit: $(date)"
echo "pip installing"
pip install torch torchvision scipy
echo "Time after pip installing: $(date)"
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


create_datasets("/root/image_net_data")
EOF
echo "Unzipping the tarred files"
python open_up.py
echo "Time after opening up the tarred files: $(date)"
echo "Dataset is ready"