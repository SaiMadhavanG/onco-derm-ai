"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, model_finetune, preprocess_data_input


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data_input,
                inputs="pre-processed_train_data",
                outputs="train_dataset",
            ),
            node(
                func=model_finetune,
                inputs=[
                    "train_dataset",
                    "params:model_name",
                    "params:num_epochs",
                    "params:batch_size",
                    "params:device",
                    "params:lr",
                ],
                outputs="model_finetuned",
            ),
            node(
                func=preprocess_data_input,
                inputs="pre-processed_val_data",
                outputs="val_dataset",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "params:model_name",
                    "model_finetuned",
                    "val_dataset",
                    "params:batch_size",
                    "params:device",
                ],
                outputs="model_metrics",
            ),
        ]
    )
