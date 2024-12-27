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
from pytorch_ood.detector import RMD, MaxSoftmax, MultiMahalanobis
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm


class MyLayer(nn.Module):
    def __init__(self, bn1, relu):
        super().__init__()
        self.bn1 = bn1
        self.relu = relu

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        return x


def custom_collate_fn(batch):
    """
    Custom collate function to squeeze the labels in the batch.
    """
    data, targets = zip(*batch)  # Unpack batch into data and targets
    data = torch.stack(data)  # Combine data into a single tensor
    targets = list(targets)  # Convert targets to list
    targets = [np.array(t).squeeze() for t in targets]
    targets = torch.tensor(np.array(targets)).squeeze()  # Squeeze labels

    return data, targets


def prepare_data(
    out_ds: str,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Prepare the data for the OOD detection pipeline.

    Args:
        out_ds: The out-of-distribution dataset to use.

    Returns:
        The in-distribution dataset and the out-of-distribution dataset.
    """

    transform = WideResNet.transform_for("cifar10-pt")

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

    in_dataset_test = ConcatDataset([dataset_in_val, dataset_in_test])

    return dataset_in_train, in_dataset_test, dataset_out_test


def train_wide_resnet(
    in_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    n_epochs: int,
    batch_size: int,
    device: str,
) -> nn.Module:
    """
    Train a WideResNet model on the given datasets.

    Args:
        in_dataset: The in-distribution dataset.
        test_dataset: The test dataset.
        n_epochs: The number of epochs to train for.
        batch_size: The batch size to use.
        device: The device to use.

    Returns:
        The trained model.
    """

    in_loader = DataLoader(
        in_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    model = WideResNet(num_classes=7).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for _x, _y in tqdm(in_loader):
            x, y = _x.to(device), _y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            for _x, _y in tqdm(test_loader):
                x, y = _x.to(device), _y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_running_loss += loss.item()

    return model


def multi_mahalanobis_detector(
    wide_resnet: nn.Module,
    train_in_dataset: torch.utils.data.Dataset,
    test_in_dataset: torch.utils.data.Dataset,
    out_dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> Tuple[dict, MultiMahalanobis]:
    """
    Run the Multi-Mahalanobis detector on the given datasets.

    Args:
        wide_resnet: The WideResNet model.
        train_in_dataset: The in-distribution training dataset.
        test_in_dataset: The in-distribution test dataset.
        out_dataset: The out-of-distribution dataset.
        batch_size: The batch size to use.

    Returns:
        The metrics of the detector and the detector itself."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer1 = wide_resnet.conv1
    layer2 = wide_resnet.block1
    layer3 = wide_resnet.block2
    layer4 = wide_resnet.block3

    layer5 = MyLayer(wide_resnet.bn1, wide_resnet.relu)

    detector = MultiMahalanobis([layer1, layer2, layer3, layer4, layer5])

    train_loader = DataLoader(
        train_in_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    test_dataset = ConcatDataset([test_in_dataset, out_dataset])
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )

    detector.fit(train_loader, device=device)

    metrics = OODMetrics()

    for x, y in test_loader:
        metrics.update(detector(x.to(device)), y)

    detector.model = [model.to("cpu") for model in detector.model]
    detector.mu = [mu.to("cpu") for mu in detector.mu]
    detector.cov = [cov.to("cpu") for cov in detector.cov]
    detector.precision = [precision.to("cpu") for precision in detector.precision]

    return metrics.compute(), detector


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
