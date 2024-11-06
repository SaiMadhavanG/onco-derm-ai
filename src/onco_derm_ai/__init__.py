"""onco-derm-ai"""

from .pipelines.data_preprocessing.nodes import normalizing_images, tensoring_resizing

__version__ = "0.1"

__all__ = ["normalizing_images", "tensoring_resizing"]
