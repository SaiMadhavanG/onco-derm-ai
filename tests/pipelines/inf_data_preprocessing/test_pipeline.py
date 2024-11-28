"""
This is a boilerplate test file for pipeline 'inf_data_preprocessing'
generated using Kedro 0.19.9.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import numpy as np
import torch
import torchvision
from PIL import Image

from onco_derm_ai.pipelines.inf_data_preprocessing.nodes import (
    normalize_image,
    resize_image,
)


def test_resize_image():
    # Create a sample image using PIL
    img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))

    # Define the new size
    new_size = (28, 28)

    # Call the resize_image function
    result = resize_image(img, new_size)

    # Assert that the image is resized correctly
    assert result.shape == (3, 28, 28)
    assert isinstance(result, torch.Tensor)


def test_normalize_image():
    # Create a sample image tensor
    img = torch.rand(3, 32, 32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Call the normalize_image function
    result = normalize_image(img, mean, std)

    # Assert that the image is normalized correctly
    expected = torchvision.transforms.functional.normalize(img, mean, std)
    assert torch.allclose(result, expected)
