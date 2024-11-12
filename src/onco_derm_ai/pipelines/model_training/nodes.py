"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


class DermaMNISTDataset(Dataset):
    """
    A custom Dataset class for the DermaMNIST dataset.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the dataset.
            It should have two columns: 'image' and 'label'. 'image' should
            contain (28, 28, 3) numpy arrays.
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

    This function applies transformations, including resizing images to
    224x224 pixels and normalizing them with specified mean and standard
    deviation values.

    Args:
        train_data (pd.DataFrame): The input training data in the form of a
            pandas DataFrame.

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

    Args:
        model_name (str): The name of the model to select. Supported values
            are "ResNet18" and "VGG16".
        num_outputs (int): The number of output classes for the model.
        pretrained (bool): If True, loads a pre-trained model.

    Returns:
        models: The selected pre-trained model.

    Raises:
        ValueError: If the provided model name is not supported.
    """
    if model_name == "ResNet18":
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model


def model_finetune(
    train_dataset: DermaMNISTDataset,
    model_name: str,
    train_params: dict,
    device: str = "cpu",
) -> dict:
    """
    Finetunes a pre-trained model on the given training dataset.

    Args:
        train_dataset (DermaMNISTDataset): The dataset to train the model on.
        model_name (str): The name of the model to finetune.
        train_params (dict): A dictionary containing the training parameters.
        device (str): The device to train the model on (e.g., "cpu" or "cuda").

    Returns:
        dict: The state dictionary of the finetuned model.
    """
    model = model_select(model_name, 7, True)
    train_loader = DataLoader(
        train_dataset, batch_size=train_params["batch_size"], shuffle=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_params["lr"])
    model = model.to(device)
    losses = []
    for epoch in tqdm(range(train_params["num_epochs"])):
        model.train()
        running_loss = 0.0
        for i, lbs in train_loader:
            images, labels = i.to(device), lbs.to(device)

            # Forward pass
            outputs = model(images)
            labels_output = labels.squeeze().long()
            loss = criterion(outputs, labels_output)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

    # generate loss plot
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    return model.state_dict(), fig


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
    for imgs, labs in test_loader:
        images, labels = imgs.to(device), labs.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    report = classification_report(y_true, y_pred, output_dict=True)
    return report
