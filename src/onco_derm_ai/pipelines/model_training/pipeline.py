"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    model_finetune,
    model_select,
    preprocess_data_input,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data_input,
                inputs="pre-processed_train_data",
                outputs="train_dataset",
            ),
            node(
                func=model_select,
                inputs="params:model_name",
                outputs="image_classification_model",
            ),
            node(
                func=model_finetune,
                inputs=[
                    "train_dataset",
                    "image_classification_model",
                    "params:num_epochs",
                ],
                outputs="model_finetuned",
            ),
        ]
    )
