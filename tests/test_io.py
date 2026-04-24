"""Unit tests for model persistence helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.io import load_model, save_model
from src.models import make_logistic_regression
from src.preprocess import build_preprocessor
from src.training import fit_and_evaluate


def test_save_and_load_model_round_trip(
    tmp_path,
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """A saved pipeline should reload successfully from disk."""
    df = synthetic_patient_df.copy()
    df["bmi_implausible"] = False
    df["lab_creatinine_negative"] = False

    result = fit_and_evaluate(
        df=df,
        feature_cols=feature_columns["all"],
        preprocessor=build_preprocessor(
            feature_columns["numeric"],
            feature_columns["categorical"],
        ),
        estimator=make_logistic_regression(random_state=42),
        random_state=42,
    )
    model_path = tmp_path / "models" / "pipeline.joblib"

    saved_path = save_model(result.pipeline, model_path)
    reloaded = load_model(saved_path)

    assert saved_path.exists()
    assert saved_path == model_path
    assert hasattr(reloaded, "predict")


def test_save_and_load_model_round_trip_preserves_predictions(
    tmp_path,
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """Reloaded models should produce the same predictions as the original pipeline."""
    df = synthetic_patient_df.copy()
    df["bmi_implausible"] = False
    df["lab_creatinine_negative"] = False

    result = fit_and_evaluate(
        df=df,
        feature_cols=feature_columns["all"],
        preprocessor=build_preprocessor(
            feature_columns["numeric"],
            feature_columns["categorical"],
        ),
        estimator=make_logistic_regression(random_state=42),
        random_state=42,
    )
    model_path = tmp_path / "pipeline.joblib"
    reloaded = load_model(save_model(result.pipeline, model_path))

    original_predictions = result.pipeline.predict(df[feature_columns["all"]])
    reloaded_predictions = reloaded.predict(df[feature_columns["all"]])

    assert np.array_equal(original_predictions, reloaded_predictions)
