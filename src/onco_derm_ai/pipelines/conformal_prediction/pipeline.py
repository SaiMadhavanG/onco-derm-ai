"""
This is a boilerplate pipeline 'conformal_prediction'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calibrate_predictor, data_prep, evaluate_predictor


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_prep,
                inputs=["params:img_size", "params:normal_mean", "params:normal_std"],
                outputs=["calibration_set", "test_set"],
                name="cp_data_prep_node",
                tags=["model_retrained"],
            ),
            node(
                func=calibrate_predictor,
                inputs=[
                    "calibration_set",
                    "best_model_uri",
                    "params:alpha",
                    "params:penalty",
                    "params:batch_size",
                ],
                outputs="cp_predictor",
                name="cp_calibrate_predictor_node",
                tags=["model_retrained"],
            ),
            node(
                func=evaluate_predictor,
                inputs=["cp_predictor", "test_set", "params:batch_size"],
                outputs="cp_metrics",
                name="cp_evaluate_predictor_node",
                tags=["model_retrained"],
            ),
        ]
    )
