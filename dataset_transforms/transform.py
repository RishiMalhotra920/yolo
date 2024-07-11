import os
from concurrent.futures import ProcessPoolExecutor

import cv2

# Global variable for the image directory to avoid passing it around
IMAGE_DIRECTORY = "/Users/rishimalhotra/projects/cv/image_classification/image_net_data"
OUTPUT_DIRECTORY = (
    "/Users/rishimalhotra/projects/cv/image_classification/image_net_data_resized"
)


def resize_image(image_path):
    size = (448, 448)
    base_path, filename = os.path.split(image_path)
    # Creating a similar directory structure in the resized_images directory
    output_dir = os.path.join(
        OUTPUT_DIRECTORY, os.path.relpath(base_path, start=IMAGE_DIRECTORY)
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        # Load the image
        img = cv2.imread(image_path)
        # Resize the image
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # Save the resized image
        cv2.imwrite(output_path, img)
        return f"Resized {filename}"
    except Exception as e:
        return f"Error resizing {filename}: {str(e)}"


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Recursively list all image paths
    image_paths = []
    for root, dirs, files in os.walk(IMAGE_DIRECTORY):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_paths.append(os.path.join(root, filename))

    print("Total images:", len(image_paths))

    # Process images in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(resize_image, image_paths)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
    # after this, run `tar -czvf imagenet.tar.gz /path/to/imagenet`
    # or faster
    # tar -cvf - '/Users/rishimalhotra/projects/cv/image_classification/image_net_data_resized' | pigz > image_net_data.tar.gz
