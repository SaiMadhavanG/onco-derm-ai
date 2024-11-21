"""
This is a boilerplate pipeline 'inf_postprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import conformal_prediction, log_prediction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=conformal_prediction,
                inputs=["prediction", "cp_predictor"],
                outputs="conformal_prediction",
                name="conformal_prediction_node",
                tags=["inference"],
            ),
            node(
                func=log_prediction,
                inputs="conformal_prediction",
                outputs=None,
                name="log_prediction_node",
                tags=["inference"],
            ),
        ]
    )
