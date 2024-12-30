from typing import Tuple

import numpy as np
import torch
from medmnist import DermaMNIST
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import RAPS
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def custom_collate_fn(batch):
    """
    Custom collate function to squeeze the labels in the batch.
    """
    data = []
    targets = []
    for row in batch:
        if isinstance(row, dict):
            data.append(row["image"])
            targets.append(row["label"])
        elif isinstance(row, tuple):
            data.append(row[0])
            targets.append(row[1])
    data = torch.stack(data)  # Combine data into a single tensor
    targets = list(targets)  # Convert targets to list
    targets = [np.array(t).squeeze() for t in targets]
    targets = torch.tensor(np.array(targets)).squeeze()  # Squeeze labels

    return data, targets


def data_prep(size, mean, std) -> Tuple[Dataset, Dataset]:
    """
    Prepare the calibration and test sets for the conformal prediction pipeline.

    Args:
        size: The size to which the images should be resized.
        mean: The mean values for normalization.
        std: The standard deviation values for normalization.

    Returns:
        A tuple containing the calibration and test sets.
    """
    transform = Compose([Resize(size), ToTensor(), Normalize(mean, std)])
    calibration_set = DermaMNIST(split="val", download=True, transform=transform)
    test_set = DermaMNIST(split="test", download=True, transform=transform)
    return calibration_set, test_set


def calibrate_predictor(
    calibration_set: Dataset,
    best_model: nn.Module,
    alpha: float,
    penalty: float,
    batch_size: int,
) -> SplitPredictor:
    """
    Calibrate a SplitPredictor using the calibration set.

    Args:
        calibration_set: The calibration set.
        best_model_uri: The URI of the best model.
        alpha: The significance level.
        penalty: The penalty parameter for the RAPS score.
        batch_size: The batch size.

    Returns:
        The calibrated SplitPredict
    """
    model = best_model

    predictor = SplitPredictor(score_function=RAPS(penalty=penalty), model=model)

    cal_loader = DataLoader(
        calibration_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    predictor.calibrate(cal_loader, alpha=alpha)

    return predictor


def evaluate_predictor(
    predictor: SplitPredictor, test_set: Dataset, batch_size: int
) -> dict:
    """
    Evaluate a SplitPredictor using the test set.

    Args:
        predictor: The SplitPredictor to evaluate.
        test_set: The test set.
        batch_size: The batch size.

    Returns:
        A dictionary containing the metrics."""
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )

    metrics = predictor.evaluate(test_loader)

    return metrics
