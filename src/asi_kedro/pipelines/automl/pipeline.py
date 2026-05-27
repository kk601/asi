"""Pipeline definition for AutoML with AutoGluon."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_automl, train_automl


def create_pipeline(**kwargs) -> Pipeline:
    """Create the AutoML pipeline."""
    return pipeline(
        [
            node(
                func=train_automl,
                inputs=["X_train", "y_train", "params:target_column", "params:automl"],
                outputs="automl_predictor",
                name="train_automl_node",
            ),
            node(
                func=evaluate_automl,
                inputs=[
                    "automl_predictor",
                    "X_val",
                    "y_val",
                    "params:automl",
                ],
                outputs="automl_metrics",
                name="evaluate_automl_node",
            ),
        ]
    )