"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from .pipelines.inf_data_preprocessing import (
    create_pipeline as create_inf_data_preprocessing_pipeline,
)
from .pipelines.inf_postprocessing import (
    create_pipeline as create_inf_postprocessing_pipeline,
)
from .pipelines.model_inference import (
    create_pipeline as create_model_inference_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # data_preprocessing_pipeline = create_data_preprocessing_pipeline()
    # model_training_pipeline = create_model_training_pipeline()
    # conformal_prediction_pipeline = create_conformal_prediction_pipeline()
    # ood_detection_pipeline = create_ood_detection_pipeline()
    inf_data_preprocessing_pipeline = create_inf_data_preprocessing_pipeline()
    model_inference_pipeline = create_model_inference_pipeline()
    inf_postprocessing_pipeline = create_inf_postprocessing_pipeline()

    inf_data_preprocessing_nodes = inf_data_preprocessing_pipeline.only_nodes_with_tags(
        "inference"
    )
    model_inference_nodes = model_inference_pipeline.only_nodes_with_tags("inference")
    inf_postprocessing_nodes = inf_postprocessing_pipeline.only_nodes_with_tags(
        "inference"
    )

    inference_pipeline = pipeline(
        [inf_data_preprocessing_nodes, model_inference_nodes, inf_postprocessing_nodes],
        inputs=["cp_predictor", "best_model", "ood_detector"],
        parameters=[
            "ood_threshold",
            "normal_mean",
            "normal_std",
            "device",
            "img_size",
        ],
        # outputs={"conformal_prediction": "inference.conformal_prediction", "integrated_gradients": "inference.integrated_gradients"},
        # namespace="inference"
    )

    pipelines = find_pipelines()
    pipelines["inference"] = inference_pipeline
    return pipelines
