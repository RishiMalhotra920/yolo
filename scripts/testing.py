import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import functional as F


def check_if_bbox_outside_image_bounds(bbox, width, height):
    # if bbox starts after the image ends or ends before the image starts, then return None
    print("call ", bbox[0], width, bbox[2], 0, bbox[0] >= width)
    if bbox[0] >= width or bbox[2] <= 0:
        return True

    # if bbox starts after the image ends or ends before the image starts, then return None
    if bbox[1] >= height or bbox[3] <= 0:
        return True

    return False


def plot(ax, image, rec):
    """
    rec of the form [x1, y1, x2, y2]
    """
    ax.imshow(image.permute(1, 2, 0))
    if rec is not None:
        ax.add_patch(
            plt.Rectangle(
                (rec[0], rec[1]),
                rec[2] - rec[0],
                rec[3] - rec[1],
                edgecolor="red",
                fill=False,
            )
        )


# Load your image
pil_image = Image.open(
    "/Users/rishimalhotra/projects/cv/image_classification/cv-projects/assets/peacock.jpeg"
)
width, height = pil_image.size
image = F.to_tensor(pil_image)
print("image shape:", image.shape)

# x1, y1, x2, y2
# rec = np.array([150, 200, 1550, 1000])
rec = np.array([700, 500, 900, 1000])
# rec = np.array([1500, 1000, 1550, 1020])


# set figsize to really big
fig, ax = plt.subplots(1, 2, figsize=(20, 10))


# Set fixed translations in pixels
translate_x_pixels = 200  # translate 50 pixels to the right
translate_y_pixels = 230  # translate 30 pixels down

# resized_image = F.resize(image, [448, 448])

# Apply the affine transformation with fixed translations

scaling_factor = 1.5
scaled_image = F.affine(
    image,
    angle=0,
    translate=[translate_x_pixels, translate_y_pixels],
    # translate=[translate_x_pixels, translate_y_pixels],
    scale=scaling_factor,
    shear=0,
)

resize_x_scaling_factor = 448 / width
resize_y_scaling_factor = 448 / height

scaled_then_resized_image = F.resize(scaled_image, [448, 448])

original_width, original_height = pil_image.size


side_crop = ((scaling_factor - 1) * original_width) / 2
top_crop = ((scaling_factor - 1) * original_height) / 2

# apply mins and maxes


def bbox_x_transform(x):
    coord_scaled = x * scaling_factor - side_crop
    coord_scaled_and_translated = coord_scaled + translate_x_pixels
    coord_scaled_and_translated_and_clipped = max(
        0, min(coord_scaled_and_translated, original_width)
    )
    return coord_scaled_and_translated_and_clipped


def bbox_y_transform(y):
    coord_scaled = y * scaling_factor - top_crop
    coord_scaled_and_translated = coord_scaled + translate_y_pixels
    coord_scaled_and_translated_and_clipped = max(
        0, min(coord_scaled_and_translated, original_height)
    )
    return coord_scaled_and_translated_and_clipped


def bbox_x_resize_transform(x):
    return int(x * resize_x_scaling_factor)


def bbox_y_resize_transform(y):
    return int(y * resize_y_scaling_factor)


# bbox_x_transform = lambda x: int(x * scaling_factor - side_crop) + translate_x_pixels
# bbox_y_transform = lambda y: int(y * scaling_factor - top_crop) + translate_y_pixels


print("original x: ", rec[0])
print("image width: ", original_width)
print("side_crop: ", side_crop)
print("x*scaling_factor: ", rec[0] * scaling_factor)
print("bbox_x_transform(x): ", bbox_x_transform(rec[0]))

# max

# there needs to be a clip step to ensure that the bounding box doesn't go out of bounds

rec2 = [
    bbox_x_transform(rec[0]),
    bbox_y_transform(rec[1]),
    bbox_x_transform(rec[2]),
    bbox_y_transform(rec[3]),
]

is_box_oob = check_if_bbox_outside_image_bounds(rec2, original_width, original_height)
# print("this is is-box-oob", is_box_oob)

plot(ax[0], image, rec)
if not is_box_oob:
    rec3 = [
        bbox_x_resize_transform(rec2[0]),
        bbox_y_resize_transform(rec2[1]),
        bbox_x_resize_transform(rec2[2]),
        bbox_y_resize_transform(rec2[3]),
    ]

    is_box_2_oob = check_if_bbox_outside_image_bounds(rec3, 448, 448)

    if not is_box_2_oob:
        print("resized_image_size", scaled_then_resized_image.shape)
        plot(ax[1], scaled_then_resized_image, rec3)
    else:
        print("in else 1")
        plot(ax[1], scaled_then_resized_image, None)

else:
    print("in else 2")
    plot(ax[1], scaled_then_resized_image, None)
    # print("scaled_image_size", scaled_image.shape)


plt.show()
