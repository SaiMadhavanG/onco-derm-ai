"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import class_imbalance, normalizing_images, tensoring_resizing


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=class_imbalance,
                inputs=["train_raw", "params:class_imbalance"],
                outputs="train_balanced",
                name="making_sure_data_is_balanced",
            ),
            node(
                func=normalizing_images,
                inputs="train_balanced",
                outputs="train_intermediate",
                name="normalizing_train_image_node",
            ),
            node(
                func=tensoring_resizing,
                inputs="train_intermediate",
                outputs="pre-processed_train_data",
                name="tensoring_train_resizing_node",
            ),
            node(
                func=normalizing_images,
                inputs="val_raw",
                outputs="val_intermediate",
                name="normalizing_val_image_node",
            ),
            node(
                func=tensoring_resizing,
                inputs="val_intermediate",
                outputs="pre-processed_val_data",
                name="tensoring_val_resizing_node",
            ),
        ]
    )
