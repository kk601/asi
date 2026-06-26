"""Nodes for the synthetic data pipeline using SDV."""

import logging
import os
from typing import Any, Dict

import pandas as pd
import wandb
from dotenv import load_dotenv
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

load_dotenv()

logger = logging.getLogger(__name__)

ID_COLUMNS = ["ID", "id"]


def _drop_id(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe without identifier columns."""
    return data.drop(columns=ID_COLUMNS, errors="ignore")


def generate_synthetic_data(
    real_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Generate synthetic passengers with a GaussianCopula synthesizer.

    Args:
        real_data: Real dataframe from the Data Catalog (``airline_raw``).
            The ``ID`` column is dropped before modelling because a unique
            identifier degrades synthesizer quality.
        parameters: ``params:synthetic`` sub-dict; uses ``n_samples``.

    Returns:
        Synthetic dataframe with the same columns as ``real_data`` minus ``ID``.
    """
    data = _drop_id(real_data)

    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)

    synthetic = synthesizer.sample(num_rows=parameters["n_samples"])
    logger.info("Generated %d synthetic rows", len(synthetic))
    return synthetic


def evaluate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate synthetic data quality and log scores to Weights & Biases.

    Args:
        real_data: Real dataframe; ``ID`` is dropped here too so its columns
            match ``synthetic_data``.
        synthetic_data: Output of :func:`generate_synthetic_data`.
        parameters: ``params:synthetic`` sub-dict; uses ``wandb_project``,
            ``wandb_entity``.

    Returns:
        ``{"diagnostic_score": float, "quality_score": float}``.
    """
    data = _drop_id(real_data)
    metadata = Metadata.detect_from_dataframe(data)

    diagnostic = run_diagnostic(data, synthetic_data, metadata)
    quality = evaluate_quality(data, synthetic_data, metadata)

    scores = {
        "diagnostic_score": float(diagnostic.get_score()),
        "quality_score": float(quality.get_score()),
    }

    with wandb.init(
        project=parameters["wandb_project"],
        entity=parameters["wandb_entity"],
        name="sdv_evaluation",
        job_type="sdv_evaluation",
        config={"n_samples": len(synthetic_data)},
        tags=["sdv", "synthetic"],
    ):
        wandb.log(
            {
                "sdv/diagnostic_score": scores["diagnostic_score"],
                "sdv/quality_score": scores["quality_score"],
            }
        )

    logger.info(
        "SDV scores -> diagnostic=%.4f quality=%.4f",
        scores["diagnostic_score"],
        scores["quality_score"],
    )
    return scores
