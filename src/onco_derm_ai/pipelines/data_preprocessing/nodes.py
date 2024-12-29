"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

from typing import Tuple

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from imblearn.over_sampling import SMOTE
from torchvision import transforms


def split_data(data: DatasetDict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        data (DatasetDict): A dictionary containing the dataset split into keys like "train",
            "validation", and "test".

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and test datasets.
    """
    return data["train"], data["validation"], data["test"]


def class_imbalance(
    data: Dataset, class_imbalance: bool, image_size: Tuple[int, int]
) -> Dataset:
    """
    Handles class imbalance in the dataset using SMOTE (Synthetic Minority Oversampling Technique).

    Args:
        data (Dataset): The dataset to balance, containing "image" and "label" fields.
        class_imbalance (bool): Flag to indicate whether to apply class balancing.
        image_size (Tuple[int, int]): The size (height, width) to which images will be reshaped.

    Returns:
        Dataset: The balanced dataset with images and labels after oversampling, if applicable.
    """
    if class_imbalance:
        train_images = np.stack(data["image"])
        train_labels = np.array(data["label"], dtype=int)
        X = train_images.reshape(train_images.shape[0], -1)  # Flatten images
        y = train_labels
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        train_images = X_resampled.reshape(-1, image_size[0], image_size[1], 3)
        train_labels = y_resampled
        data = Dataset.from_dict({"image": train_images, "label": train_labels})
    return data


data_augmentation = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.75),
        transforms.RandomVerticalFlip(p=0.75),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
)


def data_aug(
    data: Dataset, data_augmentation_flg: bool, num_augmented_per_image: int
) -> Dataset:
    """
    Applies data augmentation to the dataset.

    Args:
        data (Dataset): The dataset to augment, containing "image" and "label" fields.
        data_augmentation_flg (bool): Flag to indicate whether to apply data augmentation.
        num_augmented_per_image (int): The number of augmented versions to generate per image.

    Returns:
        Dataset: The augmented dataset containing the original and augmented samples.
    """
    if not data_augmentation_flg:
        return data

    def augment_generator(dataset):
        for example in dataset:
            augmented_image = data_augmentation(example["image"])
            yield {"image": augmented_image, "label": example["label"]}

    # Create an augmented dataset using the generator
    augmented_datasets = []

    for i in range(num_augmented_per_image):
        augmented_datasets.append(
            Dataset.from_generator(
                lambda: augment_generator(data), features=data.features
            )
        )

    concatenated_ds = concatenate_datasets([data, *augmented_datasets])

    return concatenated_ds


def tensoring_resizing(data: Dataset, image_size: Tuple[int, int]) -> Dataset:
    """
    Applies resizing and tensor conversion to the dataset.

    Args:
        data (Dataset): The dataset to process, containing "image" and "label" fields.
        image_size (Tuple[int, int]): The target size (height, width) for resizing images.

    Returns:
        Dataset: The processed dataset with images resized and converted to tensors.
    """
    transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )

    def transforms_fn(example):
        image = example["image"]
        image = transform(image)
        return {"image": image, "label": example["label"]}

    data.map(transforms_fn, batched=False, writer_batch_size=200)
    return data
