"""
This is a boilerplate pipeline 'model_eval'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_models, evaluate_model, log_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=[
                    "params:model_name",
                    "model_finetuned",
                    "new_dataset",
                    "params:batch_size",
                    "params:device",
                ],
                outputs="model_new_data_metrics",
                name="evaluate_model_node_new_data",
            ),
            node(
                func=log_model,
                inputs=[
                    "params:model_name",
                    "model_finetuned",
                    "params:train_params",
                    "model_new_data_metrics",
                    "loss_plot",
                ],
                outputs="model_new_data_uri",
                name="log_model_node_new_data",
            ),
            node(
                func=compare_models,
                inputs=["best_model_uri", "model_new_data_uri", "retrain_trigger"],
                outputs=None,
                name="compare_models_node",
            ),
        ]
    )
