"""Regression tests for the reusable training pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from src.models import make_logistic_regression
from src.preprocess import build_preprocessor
from src.training import cross_validate_estimator, fit_and_evaluate
from scripts.train_baseline import main as run_baseline


def test_regression_fit_and_evaluate_splits_before_fitting_preprocessor(
    monkeypatch,
    feature_columns: dict[str, list[str]],
) -> None:
    """Train-time preprocessing should not learn from held-out rows."""
    train_df = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "age": [40, 41, 42, 43],
            "bmi": [25.0, 26.0, 27.0, 28.0],
            "num_prior_admissions": [0, 0, 1, 1],
            "length_of_stay": [2, 2, 3, 3],
            "lab_sodium": [10.0, 12.0, 10.0, 12.0],
            "lab_creatinine": [0.9, 1.0, 1.0, 1.1],
            "bmi_implausible": [False, False, False, False],
            "lab_creatinine_negative": [False, False, False, False],
            "sex": ["M", "M", "F", "F"],
            "diagnosis_code": ["I10", "I10", "E11.9", "E11.9"],
            "hospital_id": ["H01", "H01", "H02", "H02"],
            "readmission_30d": [0, 0, 1, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "patient_id": [3, 3],
            "age": [50, 51],
            "bmi": [29.0, 30.0],
            "num_prior_admissions": [2, 2],
            "length_of_stay": [4, 4],
            "lab_sodium": [1000.0, 1000.0],
            "lab_creatinine": [1.2, 1.3],
            "bmi_implausible": [False, False],
            "lab_creatinine_negative": [False, False],
            "sex": ["other", "other"],
            "diagnosis_code": ["Z99.9", "Z99.9"],
            "hospital_id": ["H99", "H99"],
            "readmission_30d": [1, 1],
        }
    )
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    def fake_split(*args, **kwargs):
        return train_df.copy(), test_df.copy()

    monkeypatch.setattr("src.training.split_by_patient", fake_split)

    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    result = fit_and_evaluate(
        df=full_df,
        feature_cols=feature_columns["all"],
        preprocessor=preprocessor,
        estimator=make_logistic_regression(random_state=42),
        random_state=42,
    )

    numeric_imputer = (
        result.pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    lab_sodium_idx = feature_columns["numeric"].index("lab_sodium")

    assert numeric_imputer.statistics_[lab_sodium_idx] == 11.0
    predictions = result.pipeline.predict(test_df[feature_columns["all"]])
    assert predictions.shape[0] == len(test_df)


def test_regression_cross_validate_estimator_uses_grouped_shuffled_stratified_cv(
    monkeypatch,
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """Cross-validation should pass patient groups into a shuffled grouped splitter."""
    df = synthetic_patient_df.copy()
    df["bmi_implausible"] = False
    df["lab_creatinine_negative"] = False
    captured: dict[str, object] = {}

    def fake_cross_val_score(model, X, y, groups, cv, scoring):
        captured["groups"] = groups
        captured["cv"] = cv
        captured["scoring"] = scoring
        return np.array([0.7, 0.8, 0.9, 0.85, 0.75])

    monkeypatch.setattr("src.training.cross_val_score", fake_cross_val_score)

    result = cross_validate_estimator(
        df=df,
        feature_cols=feature_columns["all"],
        preprocessor=build_preprocessor(
            feature_columns["numeric"],
            feature_columns["categorical"],
        ),
        estimator=make_logistic_regression(random_state=42),
        random_state=42,
    )

    assert result["mean_score"] == np.mean([0.7, 0.8, 0.9, 0.85, 0.75])
    assert np.array_equal(np.asarray(captured["groups"]), df["patient_id"].to_numpy())
    assert isinstance(captured["cv"], StratifiedGroupKFold)
    assert captured["cv"].shuffle is True
    assert captured["scoring"] == "roc_auc"


def test_regression_cross_validate_estimator_handles_ordered_targets_without_collapsing(
    ordered_cv_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """Ordered target blocks should still yield finite grouped CV ROC AUC scores."""
    result = cross_validate_estimator(
        df=ordered_cv_df,
        feature_cols=feature_columns["all"],
        preprocessor=build_preprocessor(
            feature_columns["numeric"],
            feature_columns["categorical"],
        ),
        estimator=make_logistic_regression(random_state=42),
        random_state=42,
    )

    assert len(result["fold_scores"]) == 5
    assert np.isfinite(result["fold_scores"]).all()
    assert np.isfinite(result["mean_score"])


def test_training_pipeline_happy_path_fits_and_predicts_on_tiny_fixture(
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """The shared pipeline should fit and predict without error on tiny data."""
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

    predictions = result.pipeline.predict(df[feature_columns["all"]])
    assert predictions.shape == (len(df),)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert 0.0 <= result.evaluation.accuracy <= 1.0


def test_training_pipeline_happy_path_is_reproducible_for_same_seed(
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """The same seed should produce identical predictions and metrics."""
    df = synthetic_patient_df.copy()
    df["bmi_implausible"] = False
    df["lab_creatinine_negative"] = False

    preprocessor_a = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    preprocessor_b = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )

    result_a = fit_and_evaluate(
        df=df,
        feature_cols=feature_columns["all"],
        preprocessor=preprocessor_a,
        estimator=make_logistic_regression(random_state=42),
        random_state=42,
    )
    result_b = fit_and_evaluate(
        df=df,
        feature_cols=feature_columns["all"],
        preprocessor=preprocessor_b,
        estimator=make_logistic_regression(random_state=42),
        random_state=42,
    )

    preds_a = result_a.pipeline.predict(df[feature_columns["all"]])
    preds_b = result_b.pipeline.predict(df[feature_columns["all"]])

    assert np.array_equal(preds_a, preds_b)
    assert result_a.evaluation.accuracy == result_b.evaluation.accuracy
    assert result_a.evaluation.roc_auc == result_b.evaluation.roc_auc


@pytest.mark.slow
def test_training_pipeline_end_to_end_smoke_test_on_real_csv(tmp_path) -> None:
    """Run the baseline entrypoint on the real CSV and assert sane outputs."""
    model_path = tmp_path / "baseline_smoke.joblib"

    result = run_baseline(
        [
            "--data",
            "data/patients.csv",
            "--cv-folds",
            "5",
            "--random-state",
            "42",
            "--output-model",
            str(model_path),
        ]
    )

    assert set(result["metrics"]) == {"accuracy", "roc_auc"}
    assert 0.55 < result["metrics"]["roc_auc"] < 0.90
    assert model_path.exists()
