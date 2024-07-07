import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

# Define the image path
image_path = "/Users/rishimalhotra/projects/cv/image_classification/cv-projects/assets/peacock.jpeg"

# Define the bounding box coordinates [x1, y1, x2, y2]
bounding_box = [100, 100, 300, 300]

# Define the translation and scaling parameters
translate_params = (
    0.1,
    0.1,
)  # (horizontal_shift, vertical_shift) as fractions of image size
scale_factor = 1.2

# Load the image
image = Image.open(image_path)

# Create a copy of the image for transformed version
transformed_image = image.copy()

# Apply translation transform
translation_transform = transforms.RandomAffine(degrees=0, translate=translate_params)
transformed_image = translation_transform(transformed_image)

# Apply scaling transform
scaling_transform = transforms.RandomAffine(
    degrees=0, scale=(scale_factor, scale_factor)
)
transformed_image = scaling_transform(transformed_image)

# Modify the bounding box coordinates based on the applied transforms
x1, y1, x2, y2 = bounding_box
width, height = image.size
x1 += int(translate_params[0] * width)
y1 += int(translate_params[1] * height)
x2 += int(translate_params[0] * width)
y2 += int(translate_params[1] * height)
x1 = int(x1 * scale_factor)
y1 = int(y1 * scale_factor)
x2 = int(x2 * scale_factor)
y2 = int(y2 * scale_factor)
modified_bounding_box = [x1, y1, x2, y2]

# Resize both images to 448x448
resize_transform = transforms.Resize((448, 448))
image = resize_transform(image)
transformed_image = resize_transform(transformed_image)

# Draw bounding boxes on the original and transformed images
draw = ImageDraw.Draw(image)
draw.rectangle(bounding_box, outline="red", width=2)
draw_transformed = ImageDraw.Draw(transformed_image)
draw_transformed.rectangle(modified_bounding_box, outline="red", width=2)

# Convert the images to PyTorch tensors
to_tensor = transforms.ToTensor()
image_tensor = to_tensor(image)
transformed_image_tensor = to_tensor(transformed_image)

# Plot the original and transformed images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image_tensor.permute(1, 2, 0))
ax1.set_title("Original Image")
ax1.axis("off")
ax2.imshow(transformed_image_tensor.permute(1, 2, 0))
ax2.set_title("Transformed Image")
ax2.axis("off")
plt.tight_layout()
plt.show()
