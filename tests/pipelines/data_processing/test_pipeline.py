"""Smoke tests for the data_processing pipeline structure."""

from asi_kedro.pipelines.data_processing import create_pipeline


def test_create_pipeline_has_expected_nodes():
    pipeline_obj = create_pipeline()
    node_names = {node.name for node in pipeline_obj.nodes}
    assert node_names == {
        "preprocess_node",
        "split_data_node",
        "train_model_node",
        "evaluate_model_node",
    }


def test_pipeline_inputs_use_data_catalog():
    pipeline_obj = create_pipeline()
    inputs = pipeline_obj.inputs()
    assert "airline_raw" in inputs
    assert "db_path" not in inputs


def test_pipeline_produces_metrics_and_model():
    pipeline_obj = create_pipeline()
    # all_outputs() zwraca wszystkie wyjścia node'ów (także konsumowane wewnętrznie),
    # outputs() zwraca tylko "free outputs" pipeline'u.
    all_outputs = pipeline_obj.all_outputs()
    assert "trained_model" in all_outputs
    assert "metrics" in all_outputs
