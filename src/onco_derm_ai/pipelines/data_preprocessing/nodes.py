"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from torchvision import transforms


def class_imbalance(data: pd.DataFrame, class_imbalance: bool) -> pd.DataFrame:
    if class_imbalance:
        train_images = np.stack(data["image"])
        train_labels = data["label"].to_numpy(dtype="int32")
        X = train_images.reshape(train_images.shape[0], -1)  # Flatten images
        y = train_labels
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        train_images = X_resampled.reshape(
            -1, 28, 28, 3
        )  # Replace 28, 28 with actual dimensions
        train_labels = y_resampled
        data.drop(data.index, inplace=True)
        train_ids = [f"train_{i}" for i in range(train_images.shape[0])]
        data["id"] = train_ids
        data["image"] = list(train_images)
        data["label"] = list(train_labels)
        # data = pd.DataFrame([train_images, train_labels], columns=["image", "label"])
    return data


def normalizing_images(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the pixel values of images in the given DataFrame.

    This function takes a DataFrame containing image data and normalizes the pixel values
    by dividing each pixel value by 255.0. The normalized pixel values will be in the range [0, 1].

    Args:
        data (pd.DataFrame): A DataFrame containing image data. The DataFrame must have a column
            named "image" where each entry is an image represented as a numerical array.

    Returns:
        pd.DataFrame: A DataFrame with the same structure as the input, but with normalized image pixel values.
    """
    data["image"] = data["image"].apply(lambda x: x / 255.0)
    return data


def tensoring_resizing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a series of transformations to the 'image' column of a pandas DataFrame.

    The transformations include converting images to PIL format, resizing them to 28x28 pixels,
    and converting them to tensors. The transformed images are then permuted and converted back
    to numpy arrays.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing an 'image' column with image data.

    Returns:
        pd.DataFrame: The input DataFrame with the 'image' column transformed.
    """
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((28, 28)), transforms.ToTensor()]
    )

    data["image"] = data["image"].apply(lambda x: transform(x).permute(1, 2, 0).numpy())
    return data
