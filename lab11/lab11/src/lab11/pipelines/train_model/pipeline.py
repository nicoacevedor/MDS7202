"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_data, evaluate_model, get_best_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["prepared_data", "params:split_params"],
            outputs=["X_train", "X_valid", "X_test", "y_train", "y_valid", "y_test"],
            name="split_data"
        ),
        node(
            func=train_model,
            inputs=["X_train", "X_valid", "y_train", "y_valid"],
            outputs="best_model",
            name="train_model"
        ),
        node(
            func=evaluate_model,
            inputs=["best_model", "X_test", "y_test"],
            outputs=None,
            name="evaluate_model"
        )
    ])
