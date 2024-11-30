"""
This is a boilerplate pipeline 'model_eval'
generated using Kedro 0.19.9
"""

import datetime

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from onco_derm_ai.pipelines.model_training.nodes import (
    DermaMNISTDataset,
    model_select,
)


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


def log_model(
    model_name: str,
    model_state_dict: dict,
    hyperparams: dict,
    metrics: dict,
    loss_plot: plt.Figure,
) -> str:
    """
    Logs the model, hyperparameters, and metrics to MLFlow.

    Args:
        model_name (str): The name of the model.
        model_state_dict (dict): The state dictionary of the model.
        hyperparams (dict): The hyperparameters used during training.
        metrics (dict): The evaluation metrics of the model.
        loss_plot (plt.Figure): The plot of the training loss.

    Returns:
        str: The URI of the logged model.
    """
    mlflow.set_tracking_uri("http://localhost:5000")

    model = model_select(model_name, 7, False)
    model.load_state_dict(model_state_dict)

    # reformat metrics to the format expected by MLflow
    new_metrics = {}
    new_metrics["f1"] = metrics["macro avg"]["f1-score"]

    # Start an MLflow run
    with mlflow.start_run(run_name=model_name, nested=True):
        # Log model
        mlflow.pytorch.log_model(
            model,
            artifact_path=model_name,
            input_example=np.random.randn(1, 3, 224, 224),
            registered_model_name=model_name,
        )

        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Log metrics
        mlflow.log_metrics(new_metrics)

        # Log loss plot
        # mlflow.log_figure(loss_plot, f"{model_name}_loss_plot.png")

        # Get the URI of the logged model
        model_uri = mlflow.get_artifact_uri(model_name)

    return model_uri


def compare_models(model1_uri: str, model2_uri: str, file_name: str) -> None:
    """
    Compares the performance of two models.

    Args:
        model1_uri (str): The URI of the first model.
        model2_uri (str): The URI of the second model.

    Returns:
        Tuple[float, float]: A tuple containing the F1 score and accuracy of the first model,
            and the F1 score and accuracy of the second model.
    """
    client = MlflowClient()
    model1 = client.get_model_version(model1_uri)
    model2 = client.get_model_version(model2_uri)
    model1_f1 = model1.data.metrics["f1"]
    # model1_accuracy = model1.data.metrics["accuracy"]
    model2_f1 = model2.data.metrics["f1"]
    # model2_accuracy = model2.data.metrics["accuracy"]
    if model1_f1 < model2_f1:
        with open(file_name, "a") as f:
            f.write(
                f"Retrain True, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
    # return model1_f1, model1_accuracy, model2_f1, model2_accuracy
