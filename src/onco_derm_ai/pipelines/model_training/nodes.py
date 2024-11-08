"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class DermaMNISTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe["image"][idx]  # (28, 28, 3) numpy array
        # image = (image * 255).astype(np.uint8)  # Convert to uint8 for transforms
        label = self.dataframe["label"][idx]

        # Convert numpy image to PIL Image for applying transforms
        image = transforms.ToPILImage()(image)
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)

        # print(type(image))
        return image, label


def preprocess_data_input(train_data: pd.DataFrame) -> DermaMNISTDataset:
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


def model_select(model_name: str) -> models:
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "VGG16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model


def model_finetune(
    train_dataset: DermaMNISTDataset, model: models, num_epochs: int
) -> dict:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    num_classes = 7
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
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
