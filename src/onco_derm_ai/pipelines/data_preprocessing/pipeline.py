"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_aug, split_data, tensoring_resizing


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs="raw",
                outputs=["train_raw", "val_raw", "test_raw"],
                name="splitting_data",
            ),
            node(
                func=data_aug,
                inputs=[
                    "train_raw",
                    "params:data_augmentation",
                    "params:num_augmented_per_image",
                ],
                outputs="train_augmented",
                name="data_augmentation_node",
            ),
            node(
                func=tensoring_resizing,
                inputs=["train_augmented", "params:img_size"],
                outputs="pre-processed_train_data",
                name="tensoring_train_resizing_node",
            ),
            node(
                func=tensoring_resizing,
                inputs=["val_raw", "params:img_size"],
                outputs="pre-processed_val_data",
                name="tensoring_val_resizing_node",
            ),
        ]
    )
