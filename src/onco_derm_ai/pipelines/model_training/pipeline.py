"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model,
    log_model,
    model_finetune,
    preprocess_data_input,
    set_best_model_uri,
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
                func=preprocess_data_input,
                inputs="pre-processed_val_data",
                outputs="val_dataset",
            ),
            node(
                func=model_finetune,
                inputs=[
                    "train_dataset",
                    "val_dataset",
                    "params:model_name",
                    "params:train_params",
                    "params:device",
                ],
                outputs=["model_finetuned", "training_loss"],
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "params:model_name",
                    "model_finetuned",
                    "val_dataset",
                    "params:eval_batch_size",
                    "params:device",
                ],
                outputs="model_metrics",
            ),
            node(
                func=log_model,
                inputs=[
                    "params:model_name",
                    "model_finetuned",
                    "params:train_params",
                    "model_metrics",
                    "training_loss",
                ],
                outputs="mlflow_uri",
                name="log_model",
            ),
            node(
                func=set_best_model_uri,
                inputs=["params:model_name", "mlflow_uri"],
                outputs=["best_model_uri", "best_model"],
                name="set_best_model_uri",
            ),
        ]
    )
