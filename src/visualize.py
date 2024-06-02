from utils import display_random_images, predict_on_random_images
from torchvision import transforms
import yaml
from data_setup import create_mini_datasets, get_class_names_from_folder_names
from run_manager import RunManager
import model_builder
config = yaml.safe_load(open("config.yaml"))


if __name__ == "__main__":

    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(50),
        transforms.Resize((50, 50)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #  std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing()
    ])

    to_tensor_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    classes = ["n01986214", "n02009912", "n01924916"]
    class_names = get_class_names_from_folder_names(classes)

    print('condif', config)
    mini_train_dataset, mini_val_dataset = create_mini_datasets(
        config["image_net_train_data_path"],
        config["image_net_val_data_path"],
        classes,
        data_transform)

    display_random_images(mini_train_dataset,
                          class_names=class_names, n=5, seed=4)

    # mini_model = RunManager().load_model(
    model = model_builder.TinyVGG(
        hidden_units=128,
        output_shape=len(classes)
    ).to("cpu")

    run_id = "128_channels"
    RunManager(config["run_dir"], run_id).load_model(
        model,
        epoch=29
    )

    predict_on_random_images(model, mini_val_dataset,
                             class_names=class_names, n=5, seed=4200)
    # display_random_images(mini_train_dataset,
    #   class_names=class_names, n=5, seed=4)
