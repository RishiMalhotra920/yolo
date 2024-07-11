VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

class_to_index = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}
index_to_class = {idx: cls_name for idx, cls_name in enumerate(VOC_CLASSES)}
