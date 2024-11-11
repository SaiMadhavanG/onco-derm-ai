"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class DermaMNISTDataset(Dataset):
    """
    A custom Dataset class for the DermaMNIST dataset.
    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the dataset.
                                  It should have two columns: 'image' and 'label'.
                                  'image' should contain (28, 28, 3) numpy arrays.
        transform (callable, optional): Optional transform to be applied on a sample.
    Attributes:
        dataframe (pd.DataFrame): The dataframe containing the dataset.
        transform (callable): The transform to be applied on a sample.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the image and label at the given index.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe["image"][idx]  # (28, 28, 3) numpy array
        # image = (image * 255).astype(np.uint8)  # Convert to uint8 for transforms
        label = torch.tensor(self.dataframe["label"][idx])

        # Convert numpy image to PIL Image for applying transforms
        image = transforms.ToPILImage()(image)
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)

        # print(type(image))
        return image, label


def preprocess_data_input(train_data: pd.DataFrame) -> DermaMNISTDataset:
    """
    Preprocesses the input training data for the DermaMNIST dataset.

    This function applies a series of transformations to the input training data,
    including resizing the images to 224x224 pixels and normalizing them using
    the specified mean and standard deviation values.

    Args:
        train_data (pd.DataFrame): The input training data in the form of a pandas DataFrame.

    Returns:
        DermaMNISTDataset: The preprocessed training dataset ready for model training.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224
            # transforms.ToTensor(),             # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )
    train_dataset = DermaMNISTDataset(train_data, transform=transform)
    return train_dataset


def model_select(
    model_name: str, num_outputs: int = 7, pretrained: bool = False
) -> models:
    """
    Selects and returns a pre-trained model based on the provided model name.
    Parameters:
    model_name (str): The name of the model to select. Supported values are "ResNet18" and "VGG16".
    Returns:
    models: The selected pre-trained model.
    Raises:
    ValueError: If the provided model name is not supported.
    """
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model


def model_finetune(
    train_dataset: DermaMNISTDataset,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    device: str = "cpu",
    lr: float = 0.001,
) -> dict:
    """
    Finetunes a pre-trained model on the given training dataset.
    Args:
        train_dataset (DermaMNISTDataset): The dataset to train the model on.
        model_name (str): The name of the model to finetune.
        num_epochs (int): The number of epochs to train the model for.
        batch_size (int): The batch size to use during training.
        device (str): The device to train the model on (e.g., "cpu" or "cuda").
        lr (float): The learning rate for the optimizer.
    Returns:
        dict: The state dictionary of the finetuned model.
    """
    model = model_select(model_name, 7, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    # num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, lbs in train_loader:
            # print(type(images))
            images, labels = i.to(device), lbs.to(device)

            # Forward pass
            outputs = model(images)
            # print(labels)
            labels_output = labels.squeeze().long()
            loss = criterion(outputs, labels_output)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    # torch.save(model.state_dict(), "resnet18_dermamnist.pth")
    return model.state_dict()


def evaluate_model(
    model_name: str,
    model_state_dict: dict,
    test_dataset: DermaMNISTDataset,
    batch_size: int,
    device: str = "cpu",
) -> dict:
    """
    Evaluates a pre-trained model on the given test dataset.
    Args:
        model_name (str): The name of the model being evaluated.
        model_state_dict (dict): The state dictionary of the pre-trained model.
        test_dataset (DermaMNISTDataset): The dataset to evaluate the model on.
        batch_size (int): The batch size to use during evaluation.
        device (str): The device to evaluate the model on (e.g., "cpu" or "cuda").
    Returns:
        dict: The classification report of the model's performance.
    """
    model = model_select(model_name, 7, False)
    model.load_state_dict(model_state_dict)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    y_true = []
    y_pred = []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    report = classification_report(y_true, y_pred, output_dict=True)
    return report
