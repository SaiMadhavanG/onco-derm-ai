"""
This is a boilerplate pipeline 'ood_detection'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    multi_mahalanobis_detector,
    prepare_data,
    train_wide_resnet,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_data,
                inputs=[
                    "params:ood_detection_out_ds",
                ],
                outputs=["train_in_ds", "test_in_ds", "out_ds"],
                name="ood_prepare_data",
                tags=["model_retrained"],
            ),
            node(
                func=train_wide_resnet,
                inputs=[
                    "train_in_ds",
                    "test_in_ds",
                    "params:wide_resnet_epochs",
                    "params:wide_resnet_batch_size",
                    "params:device",
                ],
                outputs="wide_resnet_model",
                name="train_wide_resnet",
                tags=["model_retrained"],
            ),
            node(
                func=multi_mahalanobis_detector,
                inputs=[
                    "wide_resnet_model",
                    "train_in_ds",
                    "test_in_ds",
                    "out_ds",
                    "params:batch_size",
                ],
                outputs=["ood_detection_metrics", "ood_detector"],
                name="multi_mahalanobis_detector",
                tags=["model_retrained"],
            ),
        ]
    )
