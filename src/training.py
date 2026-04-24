"""Training and cross-validation helpers for reusable model workflows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data import split_by_patient
from src.evaluation import EvaluationResult, compute_classification_metrics


@dataclass(frozen=True)
class TrainingResult:
    """Container for a fitted training run."""

    model_name: str
    pipeline: Pipeline
    evaluation: EvaluationResult
    train_rows: int
    test_rows: int


def make_training_pipeline(
    preprocessor: ColumnTransformer,
    estimator,
) -> Pipeline:
    """Create a deterministic sklearn pipeline from preprocessing and model."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def fit_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    preprocessor: ColumnTransformer,
    estimator,
    target_col: str = "readmission_30d",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Split, fit, and evaluate a model using the shared preprocessing path."""
    train_df, test_df = split_by_patient(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = make_training_pipeline(preprocessor, clone(estimator))
    pipeline.fit(train_df[feature_cols], train_df[target_col])

    evaluation = compute_classification_metrics(
        pipeline,
        test_df[feature_cols],
        test_df[target_col],
    )

    return TrainingResult(
        model_name=estimator.__class__.__name__,
        pipeline=pipeline,
        evaluation=evaluation,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )


def cross_validate_estimator(
    df: pd.DataFrame,
    feature_cols: list[str],
    preprocessor: ColumnTransformer,
    estimator,
    target_col: str = "readmission_30d",
    group_col: str = "patient_id",
    cv: int = 5,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> dict[str, float | list[float]]:
    """Run grouped, stratified cross-validation on the shared pipeline."""
    if group_col not in df.columns:
        raise ValueError(f"Grouping column '{group_col}' is missing.")

    splitter = StratifiedGroupKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )
    pipeline = make_training_pipeline(preprocessor, clone(estimator))
    scores = cross_val_score(
        pipeline,
        df[feature_cols],
        df[target_col],
        groups=df[group_col],
        cv=splitter,
        scoring=scoring,
    )
    return {
        "fold_scores": [float(score) for score in scores],
        "mean_score": float(np.mean(scores)),
    }
