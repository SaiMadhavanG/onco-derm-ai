import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import models

from onco_derm_ai.pipelines.model_training.nodes import (
    DermaMNISTDataset,
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
    model = model_select("VGG16")
    assert isinstance(model, models.VGG)
    with pytest.raises(ValueError):
        model_select("UnsupportedModel")


def test_model_finetune():
    dataset = preprocess_data_input(dataframe)
    model = model_select("ResNet18")
    state_dict = model_finetune(dataset, model, num_epochs=1)
    assert isinstance(state_dict, dict)
    assert "fc.weight" in state_dict


# if __name__ == "__main__":
#     pytest.main()
