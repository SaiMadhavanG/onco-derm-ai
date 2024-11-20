"""onco-derm-ai"""

from .pipelines.data_preprocessing.nodes import normalizing_images, tensoring_resizing
from .pipelines.inf_data_preprocessing.nodes import (
    normalize_image,
    ood_detection,
    prepare_data_for_ood,
    resize_image,
)
from .pipelines.model_inference.nodes import predict
from .pipelines.model_training.nodes import (
    DermaMNISTDataset,
    evaluate_model,
    log_model,
    model_finetune,
    model_select,
    preprocess_data_input,
)
from .pipelines.ood_detection.nodes import (
    multi_mahalanobis_detector,
    prepare_data,
    train_wide_resnet,
)

__version__ = "0.1"

__all__ = [
    "normalizing_images",
    "tensoring_resizing",
    "DermaMNISTDataset",
    "model_finetune",
    "model_select",
    "preprocess_data_input",
    "evaluate_model",
    "log_model",
    "resize_image",
    "normalize_image",
    "predict",
    "prepare_data",
    "train_wide_resnet",
    "multi_mahalanobis_detector",
    "prepare_data_for_ood",
    "ood_detection",
]
