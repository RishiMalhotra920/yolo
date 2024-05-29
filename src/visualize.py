    to_tensor_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mini_train_dataset_no_transform = SubsetImageFolder(
        root="image_net_data/train", classes=classes, num_samples_per_class=1000, transform=to_tensor_transform)
    mini_val_dataset_no_transform = SubsetImageFolder(
        root="image_net_data/val", classes=classes, num_samples_per_class=50, transform=to_tensor_transform)

        predict_on_random_images(mini_model, mini_val_dataset,
                         class_names=class_names, n=5, seed=4200)
display_random_images(mini_train_dataset, class_names=class_names, n=5, seed=4)