import cv2
import numpy as np


# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(image_path):
    """
    Loads an image using OpenCV
    and converts BGR to RGB.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read the image."
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ==========================================
# RESIZE IMAGE
# ==========================================

def resize_image(
    image,
    size=(224, 224)
):
    """
    Resizes image to 224 x 224.
    """

    resized = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )

    return resized


# ==========================================
# NORMALIZE IMAGE
# ==========================================

def normalize_image(image):
    """
    Converts pixel values from
    0-255 to 0-1.
    """

    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ==========================================
# COMPLETE PREPROCESSING
# ==========================================

def preprocess_image(image_path):
    """
    DeepGuard image preprocessing pipeline.

    Steps:
    1. Load image
    2. Convert BGR to RGB
    3. Resize to 224 x 224
    4. Normalize pixels
    """

    image = load_image(
        image_path
    )

    image = resize_image(
        image,
        (224, 224)
    )

    image = normalize_image(
        image
    )

    return image, False