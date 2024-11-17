"""
This is a boilerplate pipeline 'inf_data_preprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import normalize_image, resize_image


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=resize_image,
                inputs=["inference_sample", "params:img_size"],
                outputs="resized_img",
                name="resize_image_node_inference",
                tags=["inference"],
            ),
            node(
                func=normalize_image,
                inputs=["resized_img", "params:normal_mean", "params:normal_std"],
                outputs="normalized_img",
                name="normalize_image_node_inference",
                tags=["inference"],
            ),
        ]
    )
