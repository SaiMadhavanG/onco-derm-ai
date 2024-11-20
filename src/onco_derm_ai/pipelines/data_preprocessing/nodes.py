"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from PIL import Image
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


data_augmentation = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.75),
        transforms.RandomVerticalFlip(p=0.75),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]
)


def data_aug(
    data: pd.DataFrame, data_augmentation_flg: bool, num_augmented_per_image: int
) -> pd.DataFrame:
    """
    Augments the images in the DataFrame and appends the augmented rows.

    Args:
        data (pd.DataFrame): Original DataFrame containing images and metadata.
        augment_fn (callable): Augmentation function to apply.
        num_augmented_per_image (int): Number of augmented images to create for each original image.

    Returns:
        pd.DataFrame: DataFrame with new augmented rows added.
    """
    if not data_augmentation_flg:
        return data

    augmented_rows = []

    for idx, row in data.iterrows():
        original_image = row["image"]
        label = row["label"]
        image_id = row["id"]

        # Convert image array to PIL Image
        pil_image = Image.fromarray(np.array(original_image, dtype=np.uint8))

        for i in range(num_augmented_per_image):
            # Apply augmentations
            augmented_image = data_augmentation(pil_image)
            # Convert back to numpy array for storage
            augmented_array = np.array(augmented_image.permute(1, 2, 0) * 255).astype(
                np.uint8
            )

            # Create a new unique ID for the augmented image
            new_id = f"{image_id}_aug_{i}"

            # Add the new row
            augmented_rows.append(
                {"id": new_id, "image": augmented_array, "label": label}
            )

    # Convert augmented rows to a DataFrame
    augmented_df = pd.DataFrame(augmented_rows)

    # Append the augmented DataFrame to the original
    return pd.concat([data, augmented_df], ignore_index=True)


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
