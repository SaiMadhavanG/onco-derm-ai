"""
This is a boilerplate pipeline 'ood_detection'
generated using Kedro 0.19.9
"""

from typing import Tuple

import medmnist
import mlflow
import numpy as np
import torch
from medmnist import INFO
from pytorch_ood.detector import RMD, MaxSoftmax
from pytorch_ood.utils import OODMetrics, ToUnknown
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm


def custom_collate_fn(batch):
    """
    Custom collate function to squeeze the labels in the batch.
    """
    data, targets = zip(*batch)  # Unpack batch into data and targets
    data = torch.stack(data)  # Combine data into a single tensor
    targets = torch.tensor(np.array(targets)).squeeze()  # Squeeze labels
    return data, targets


def prepare_data(
    out_ds: str,
    size: Tuple[int, int],
    normal_mean: Tuple[float, float, float],
    normal_std: Tuple[float, float, float],
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Prepare the data for the OOD detection pipeline.

    Args:
        out_ds: The dataset to use as the OOD dataset.
        size: The size to resize the images to.
        normal_mean: The mean to normalize the images.
        normal_std: The standard deviation to normalize the images.

    Returns:
        The in-distribution dataset and the out-of-distribution dataset.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(normal_mean, normal_std),
        ]
    )

    info = INFO["dermamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    dataset_in_train = DataClass(split="train", download=True, transform=transform)
    dataset_in_val = DataClass(split="val", download=True, transform=transform)
    dataset_in_test = DataClass(split="test", download=True, transform=transform)
    if out_ds == "cifar10":
        dataset_out_test = CIFAR10(
            root="~/.data",
            download=True,
            transform=transform,
            target_transform=ToUnknown(),
        )
    else:
        raise ValueError("Invalid out_ds")

    in_dataset = ConcatDataset([dataset_in_train, dataset_in_val, dataset_in_test])

    return in_dataset, dataset_out_test


def msp_detector(
    best_model_uri: str,
    in_dataset: torch.utils.data.Dataset,
    out_dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: str,
) -> dict:
    """
    Run the Maximum Softmax Probability (MSP) detector on the given datasets.

    Args:
        best_model_uri: The URI of the best model.
        in_dataset: The in-distribution dataset.
        out_dataset: The out-of-distribution dataset.
        batch_size: The batch size to use.
        device: The device to use.

    Returns:
        The metrics of the detector."""

    in_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
    out_loader = DataLoader(out_dataset, batch_size=batch_size, shuffle=True)

    model = mlflow.pytorch.load_model(best_model_uri).to(device).eval()

    detector = MaxSoftmax(model)

    metrics = OODMetrics()

    for loader in [in_loader, out_loader]:
        for _x, _y in tqdm(loader):
            x, y = _x.to(device), _y.to(device)
            metrics.update(detector(x), y.squeeze())

    return metrics.compute()


def rmd_detector(
    best_model_uri: str,
    in_dataset: torch.utils.data.Dataset,
    out_dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: str,
) -> Tuple[dict, RMD]:
    """
    Run the Relative Mahalanobis Distance (RMD) detector on the given datasets.

    Args:
        best_model_uri: The URI of the best model.
        in_dataset: The in-distribution dataset.
        out_dataset: The out-of-distribution dataset.
        batch_size: The batch size to use.
        device: The device to use.

    Returns:
        The metrics of the detector and the detector itself.
    """

    in_loader = DataLoader(
        in_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    out_loader = DataLoader(out_dataset, batch_size=batch_size, shuffle=True)

    model = mlflow.pytorch.load_model(best_model_uri).to(device).eval()

    detector = RMD(model)
    detector.fit(in_loader, device=device)

    metrics = OODMetrics()

    for loader in [in_loader, out_loader]:
        for _x, _y in tqdm(loader):
            x, y = _x.to(device), _y.to(device)
            metrics.update(detector(x), y.squeeze())

    return metrics.compute(), detector
