import mlflow
import torch


def predict(best_model_uri: str, input_img: torch.Tensor, device: str) -> torch.Tensor:
    """
    Predicts the output of the input image using the best model.
    Args:
        best_model_uri: URI of the best model.
        input_img: Input image to predict.
        device: Device to run the model on.
    Returns:
        output: The output tensor.
    """
    model = mlflow.pytorch.load_model(best_model_uri).to(device)
    output = model(input_img.unsqueeze(0).to(device))
    return output.detach().cpu()
