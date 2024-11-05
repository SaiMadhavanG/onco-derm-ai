"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""
import torchvision
from torchvision import transforms
import pandas as pd


def normalizing_images(data:pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the pixel values of images in the given DataFrame.

    This function takes a DataFrame containing image data and normalizes the pixel values
    by dividing each pixel value by 255.0. The normalized pixel values will be in the range [0, 1].

    Args:
        data (pd.DataFrame): A DataFrame containing image data. The DataFrame must have a column named "image"
                                where each entry is an image represented as a numerical array.

    Returns:
        pd.DataFrame: A DataFrame with the same structure as the input, but with normalized image pixel values.
    """
    data["image"] = data["image"].apply(lambda x: x / 255.0)
    return data

def tensoring_resizing(data:pd.DataFrame) -> pd.DataFrame:
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
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    data["image"] = data["image"].apply(lambda x: transform(x).permute(1, 2, 0).numpy())
    return data

