"""Evaluation helpers for classification models."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score


@dataclass(frozen=True)
class EvaluationResult:
    """Container for evaluation outputs."""

    accuracy: float
    roc_auc: float
    confusion_matrix: np.ndarray


def _prediction_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return continuous prediction scores when available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_classification_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> EvaluationResult:
    """Compute accuracy, ROC AUC, and confusion matrix for a fitted model."""
    y_pred = model.predict(X)
    y_score = _prediction_scores(model, X)

    accuracy = accuracy_score(y, y_pred)
    roc_auc = (
        float("nan")
        if pd.Series(y).nunique(dropna=False) < 2
        else roc_auc_score(y, y_score)
    )
    cm = confusion_matrix(y, y_pred)

    return EvaluationResult(
        accuracy=float(accuracy),
        roc_auc=float(roc_auc),
        confusion_matrix=cm,
    )


def results_table(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """Convert named evaluation results into a compact dataframe."""
    rows = []
    for model_name, result in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": result.accuracy,
                "roc_auc": result.roc_auc,
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(
    result: EvaluationResult,
    title: str,
) -> None:
    """Plot a confusion matrix from an ``EvaluationResult``."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix=result.confusion_matrix)
    display.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
