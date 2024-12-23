"""
This is a boilerplate pipeline 'inf_data_preprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import normalize_image, ood_detection, prepare_data_for_ood, resize_image


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
            node(
                func=prepare_data_for_ood,
                inputs="inference_sample",
                outputs="img_for_ood",
                name="prepare_data_for_ood_node",
                tags=["inference"],
            ),
            node(
                func=ood_detection,
                inputs=[
                    "img_for_ood",
                    "ood_detector",
                    "params:ood_threshold",
                    "params:device",
                ],
                outputs=None,
                name="ood_detection_node",
                tags=["inference"],
            ),
        ]
    )
