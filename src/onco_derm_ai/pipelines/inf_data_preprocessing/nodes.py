from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pytorch_ood.detector import MultiMahalanobis
from pytorch_ood.model import WideResNet


class OutOfDistributionError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


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
    img = F.to_tensor(img) / 255.0
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


def prepare_data_for_ood(img: Image) -> torch.Tensor:
    """
    Prepare image for OOD detection.
    Args:
        img: Image to be prepared.
    Returns:
        Image prepared for OOD detection
    """
    transform = WideResNet.transform_for("cifar10-pt")
    img = transform(img)
    return img


def ood_detection(
    img: torch.Tensor, detector: MultiMahalanobis, threshold: float, device: str
) -> float:
    """
    Detect out-of-distribution samples.
    Args:
        img: Image to be detected.
        detector: MultiMahalanobis OOD detector.
        threshold: Threshold for OOD detection.
        device: Device.
    Returns:
        OOD score.
    """
    img = img.unsqueeze(0)
    detector.model = [model.to(device) for model in detector.model]
    detector.mu = [mu.to(device) for mu in detector.mu]
    detector.precision = [precision.to(device) for precision in detector.precision]
    score = detector(img.to(device)).item()
    if score > threshold:
        raise OutOfDistributionError(
            f"Image detected as OOD with score {score} which is above threshold {threshold}"
        )
    return score
