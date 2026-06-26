"""Pipeline definition for synthetic data generation with SDV."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_synthetic_data, generate_synthetic_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the synthetic data pipeline."""
    return pipeline(
        [
            node(
                func=generate_synthetic_data,
                inputs=["airline_raw", "params:synthetic"],
                outputs="synthetic_data",
                name="generate_synthetic_node",
            ),
            node(
                func=evaluate_synthetic_data,
                inputs=["airline_raw", "synthetic_data", "params:synthetic"],
                outputs="synthetic_scores",
                name="evaluate_synthetic_node",
            ),
        ]
    )
