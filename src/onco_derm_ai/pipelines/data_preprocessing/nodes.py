"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from torchvision import transforms

# def get_class_imbalance():
# global train_labels, val_labels, test_labels

# # Combine all labels to assess overall imbalance
# train_labels_list = train_labels.ravel().tolist()
# val_labels_list = val_labels.ravel().tolist()
# test_labels_list = test_labels.ravel().tolist()
# all_labels = np.concatenate([train_labels_list, val_labels_list, test_labels_list])

# # Function to calculate class distribution
# def calculate_class_distribution(labels):
#     print(type(labels))
#     class_counts = Counter(labels)  # Count occurrences of each class
#     total_samples = len(labels)
#     class_distribution = {
#         cls: count / total_samples for cls, count in class_counts.items()
#     }  # Normalize
#     return class_counts, class_distribution

# # Function to calculate imbalance ratio
# def calculate_imbalance_ratio(class_counts):
#     min_class_count = min(class_counts.values())
#     max_class_count = max(class_counts.values())
#     return max_class_count / min_class_count

# # Function to calculate entropy of class distribution
# def calculate_class_entropy(class_distribution):
#     return -sum(p * np.log2(p) for p in class_distribution.values() if p > 0)

# # Calculate metrics for train, val, and test sets
# train_counts, train_distribution = calculate_class_distribution(train_labels_list)
# val_counts, val_distribution = calculate_class_distribution(val_labels_list)
# test_counts, test_distribution = calculate_class_distribution(test_labels_list)
# overall_counts, overall_distribution = calculate_class_distribution(all_labels)

# Imbalance ratios
# train_imbalance = calculate_imbalance_ratio(train_counts)
# val_imbalance = calculate_imbalance_ratio(val_counts)
# test_imbalance = calculate_imbalance_ratio(test_counts)

# # Class entropies
# train_entropy = calculate_class_entropy(train_distribution)
# val_entropy = calculate_class_entropy(val_distribution)
# test_entropy = calculate_class_entropy(test_distribution)

# Display results
# print("Train Set:")
# print(f"Class Counts: {train_counts}")
# print(f"Imbalance Ratio: {train_imbalance:.2f}")
# print(f"Entropy: {train_entropy:.2f}\n")

# print("Validation Set:")
# print(f"Class Counts: {val_counts}")
# print(f"Imbalance Ratio: {val_imbalance:.2f}")
# print(f"Entropy: {val_entropy:.2f}\n")

# print("Test Set:")
# print(f"Class Counts: {test_counts}")
# print(f"Imbalance Ratio: {test_imbalance:.2f}")
# print(f"Entropy: {test_entropy:.2f}\n")

# Plot class distribution
# def plot_class_distribution(class_counts, title):
#     classes = list(class_counts.keys())
#     counts = list(class_counts.values())
#     plt.bar(classes, counts, color="skyblue")
#     plt.xlabel("Class")
#     plt.ylabel("Count")
#     plt.title(title)
#     plt.show()

# plot_class_distribution(train_counts, "Train Set Class Distribution")
# plot_class_distribution(val_counts, "Validation Set Class Distribution")
# plot_class_distribution(test_counts, "Test Set Class Distribution")


def class_imbalance(data: pd.DataFrame, class_imbalance: bool) -> pd.DataFrame:
    if class_imbalance:
        train_images = np.stack(data["image"])
        train_labels = data["label"].to_numpy(dtype="int32")
        X = train_images.reshape(train_images.shape[0], -1)  # Flatten images
        y = train_labels
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        train_images = X_resampled.reshape(
            -1, 3, 28, 28
        )  # Replace 28, 28 with actual dimensions
        train_labels = y_resampled
        data = pd.DataFrame([train_images, train_labels], columns=["image", "label"])

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
