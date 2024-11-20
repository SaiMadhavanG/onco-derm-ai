import logging
from typing import List

import torch
from torchcp.classification.predictors import SplitPredictor


def conformal_prediction(
    output: torch.Tensor, predictor: SplitPredictor
) -> List[float]:
    """
    Perform conformal prediction on the output tensor.

    Args:
        output: The output tensor.
        predictor: The SplitPredictor.

    Returns:
        The prediction.
    """
    if output.dim() == 1:
        output = output.unsqueeze(0)
    return predictor.predict_with_logits(output)[0]


def log_prediction(prediction: List[float]) -> None:
    """
    Log the prediction.

    Args:
        prediction: The prediction.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Prediction: {prediction}")
