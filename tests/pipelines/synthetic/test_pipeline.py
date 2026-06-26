"""Tests for the synthetic data pipeline."""

import os

import pandas as pd

os.environ.setdefault("WANDB_MODE", "disabled")

from asi_kedro.pipelines.synthetic.nodes import (  # noqa: E402
    evaluate_synthetic_data,
    generate_synthetic_data,
)


def _sample_df(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": range(n),
            "Gender": ["Male", "Female"] * (n // 2),
            "Age": [30 + (i % 40) for i in range(n)],
            "Class": ["Business", "Economy", "Economy Plus"] * (n // 3),
            "Flight Distance": [500 + i for i in range(n)],
            "Satisfaction": ["Satisfied", "Neutral or Dissatisfied"] * (n // 2),
        }
    )


def test_generate_drops_id_and_returns_n_rows():
    out = generate_synthetic_data(_sample_df(), {"n_samples": 25})
    assert len(out) == 25
    assert "ID" not in out.columns
    assert "Gender" in out.columns


def test_evaluate_returns_scores_in_range():
    df = _sample_df()
    synthetic = generate_synthetic_data(df, {"n_samples": 50})
    scores = evaluate_synthetic_data(
        df, synthetic, {"wandb_project": "test", "wandb_entity": "test"}
    )
    assert set(scores) == {"diagnostic_score", "quality_score"}
    assert 0.0 <= scores["diagnostic_score"] <= 1.0
    assert 0.0 <= scores["quality_score"] <= 1.0


from asi_kedro.pipelines.synthetic import create_pipeline  # noqa: E402


def test_create_pipeline_has_expected_nodes():
    node_names = {node.name for node in create_pipeline().nodes}
    assert node_names == {"generate_synthetic_node", "evaluate_synthetic_node"}


def test_pipeline_inputs_use_airline_raw():
    assert "airline_raw" in create_pipeline().inputs()
    assert "synthetic_data" in create_pipeline().all_outputs()
    assert "synthetic_scores" in create_pipeline().all_outputs()
