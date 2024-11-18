"""
This is a boilerplate pipeline 'model_inference'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=predict,
                inputs=["best_model_uri", "normalized_img", "params:device"],
                outputs="prediction",
                name="predict",
                tags=["inference"],
            )
        ]
    )
