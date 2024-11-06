"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import normalizing_images, tensoring_resizing


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=normalizing_images,
                inputs="train_raw",
                outputs="train_intermediate",
                name="normalizing_image_node",
            ),
            node(
                func=tensoring_resizing,
                inputs="train_intermediate",
                outputs="pre-processed_train_data",
                name="tensoring_resizing_node",
            ),
        ]
    )
