"""Utility functions for image processing and manipulation."""

from typing import Optional, Tuple
import numpy as np
from PIL import Image


def crop_bbox(image: Image.Image, bbox: Optional[Tuple[float, float, float, float]]) -> Image.Image:
    """
    Crop the image to the specified bounding box.
    
    Args:
        image: The input image.
        bbox: The bounding box to crop the image to.
    Returns:
        The cropped image.
    """
    if bbox is None:
        return image

    x_min, y_min, x_max, y_max = bbox
    w, h = image.size
    x1, x2 = max(0, int(x_min)), min(w, int(x_max))
    y1, y2 = max(0, int(y_min)), min(h, int(y_max))

    if x2 > x1 and y2 > y1:
        return image.crop((x1, y1, x2, y2))
    return image


def image_to_pixel_string(
    img_input: Image.Image | np.ndarray, 
    target_size: Optional[Tuple[int, int]] = (112, 112)
) -> str:
    """
    Converts an image to a string of pixel values, optionally resizing it to a target size.

    Args:
        img_input: The input image (PIL Image or NumPy array).
        target_size: The size to resize the image to, if desired.

    Returns:
        A string of pixel values separated by spaces.
    """
    if not isinstance(img_input, Image.Image):
        arr = np.asarray(img_input)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = img_input

    gray_img = img.convert("L")
    if target_size is not None:
        gray_img = gray_img.resize(target_size, Image.Resampling.BILINEAR)

    return " ".join(np.array(gray_img, dtype=np.uint8).flatten().astype(str))