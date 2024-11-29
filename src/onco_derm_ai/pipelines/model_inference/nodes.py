import torch


def predict(
    best_model: torch.nn.Module, input_img: torch.Tensor, device: str
) -> torch.Tensor:
    """
    Predicts the output of the input image using the best model.
    Args:
        best_model: The best model to use for prediction.
        input_img: Input image to predict.
        device: Device to run the model on.
    Returns:
        output: The output tensor.
    """
    model = best_model.to(device)
    output = model(input_img.unsqueeze(0).to(device))
    return output.detach().cpu()
