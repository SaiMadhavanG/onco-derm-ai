from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


def resize_image(img: Image, size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize image.
    Args:
        img: Image to be resized.
        size: New size for the image.
    Returns:
        Resized image.
    """
    img = np.array(img.convert("RGB")).astype(np.float32)
    img = F.to_tensor(img)
    img = F.resize(img, size)
    return img


def normalize_image(
    img: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]
) -> torch.tensor:
    """
    Normalize image.
    Args:
        img: Image to be normalized.
        mean: Mean value for normalization.
        std: Standard deviation for normalization.
    Returns:
        Normalized image.
    """
    img = F.normalize(img, mean, std)
    return img
