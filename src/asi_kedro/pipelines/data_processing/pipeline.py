"""Pipeline definition for data processing."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_and_log, preprocess, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess,
                inputs=["airline_raw", "params:target_column", "params:split"],
                outputs="processed_data",
                name="preprocess_node",
            ),
            node(
                func=split_data,
                inputs=["processed_data", "params:target_column", "params:split"],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_and_log,
                inputs=[
                    "trained_model",
                    "X_val",
                    "y_val",
                    "params:model",
                    "params:split",
                ],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )
