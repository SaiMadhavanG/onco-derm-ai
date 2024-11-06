import numpy as np
import pandas as pd

from onco_derm_ai.pipelines.data_preprocessing.nodes import (
    normalizing_images,
    tensoring_resizing,
)

# FILE: src/onco_derm_ai/pipelines/data_preprocessing/test_nodes.py


def test_normalizing_images():
    # Create a sample DataFrame with image data
    data = pd.DataFrame({"image": [np.array([[0, 128, 255], [64, 192, 32]])]})

    # Call the normalizing_images function
    result = normalizing_images(data)

    # Assert that the pixel values are normalized correctly
    expected = pd.DataFrame(
        {
            "image": [
                np.array(
                    [[0.0, 128 / 255.0, 1.0], [64 / 255.0, 192 / 255.0, 32 / 255.0]]
                )
            ]
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_tensoring_resizing():
    # Create a sample DataFrame with image data
    data = pd.DataFrame(
        {"image": [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)]}
    )

    # Call the tensoring_resizing function
    result = tensoring_resizing(data)

    # Assert that the images are resized and converted to tensors correctly
    assert result["image"].iloc[0].shape == (28, 28, 3)
    assert result["image"].iloc[0].dtype == np.float32
