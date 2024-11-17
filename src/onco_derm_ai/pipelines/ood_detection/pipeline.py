"""
This is a boilerplate pipeline 'ood_detection'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_data, rmd_detector


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_data,
                inputs=[
                    "params:ood_detection_out_ds",
                    "params:img_size",
                    "params:normal_mean",
                    "params:normal_std",
                ],
                outputs=["in_ds", "out_ds"],
            ),
            node(
                func=rmd_detector,
                inputs=[
                    "best_model_uri",
                    "in_ds",
                    "out_ds",
                    "params:batch_size",
                    "params:device",
                ],
                outputs=["ood_detection_metrics", "ood_detector"],
            ),
        ]
    )
