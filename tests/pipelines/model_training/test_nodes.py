import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import models, transforms

from onco_derm_ai.pipelines.model_training.nodes import (
    DermaMNISTDataset,
    evaluate_model,
    model_finetune,
    model_select,
    preprocess_data_input,
)

# Sample data for testing
data = {
    "image": [np.random.rand(28, 28, 3) for _ in range(10)],
    "label": [np.random.randint(0, 7) for _ in range(10)],
}
dataframe = pd.DataFrame(data)


def test_derma_mnist_dataset():
    dataset = DermaMNISTDataset(dataframe)
    num = 10
    assert len(dataset) == num
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_preprocess_data_input():
    dataset = preprocess_data_input(dataframe)
    assert isinstance(dataset, DermaMNISTDataset)
    image, label = dataset[0]
    assert image.shape == torch.Size([3, 224, 224])
    # assert isinstance(label, int)


def test_model_select():
    model = model_select("ResNet18")
    assert isinstance(model, models.ResNet)
    with pytest.raises(ValueError):
        model_select("UnsupportedModel")


class MockDermaMNISTDataset:
    def __init__(self, size=10):
        self.size = size
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Returns random tensors for image data and a random integer for the label.
        return torch.rand((3, 224, 224)), torch.tensor(1)  # Label 1 for simplicity


@pytest.fixture
def mock_dataset():
    return MockDermaMNISTDataset(size=10)  # A small dataset for testing


@pytest.fixture
def train_params():
    return {"batch_size": 2, "lr": 0.001, "num_epochs": 1}


@pytest.fixture
def model_name():
    return "ResNet18"


def test_model_finetune(mock_dataset, model_name, train_params):
    """Test model_finetune function for correct output types."""
    device = "cpu"
    state_dict, fig = model_finetune(
        train_dataset=mock_dataset,
        model_name=model_name,
        train_params=train_params,
        device=device,
    )
    assert isinstance(state_dict, dict), "Expected state_dict to be of type dict"
    assert hasattr(
        fig, "savefig"
    ), "Expected fig to have a savefig method (matplotlib Figure)"


def test_evaluate_model(mock_dataset, model_name):
    """Test evaluate_model function for correct output format."""
    device = "cpu"

    # Fine-tune model to get a state_dict to test evaluation
    train_params = {"batch_size": 2, "lr": 0.001, "num_epochs": 1}
    state_dict, _ = model_finetune(
        train_dataset=mock_dataset,
        model_name=model_name,
        train_params=train_params,
        device=device,
    )

    # Evaluate the fine-tuned model
    report = evaluate_model(
        model_name=model_name,
        model_state_dict=state_dict,
        test_dataset=mock_dataset,
        batch_size=2,
        device=device,
    )

    assert isinstance(report, dict), "Expected report to be a dictionary"
    assert "accuracy" in report, "Expected accuracy key in report dictionary"
