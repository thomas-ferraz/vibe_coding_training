"""Regression tests for model evaluation helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_classification_metrics, plot_feature_importance


class ProbabilityAwareDummyModel:
    """Dummy model with label predictions and richer probability scores."""

    def predict(self, X):  # noqa: D401 - sklearn-style signature
        return np.array([0, 0, 0, 1])

    def predict_proba(self, X):  # noqa: D401 - sklearn-style signature
        return np.array(
            [
                [0.90, 0.10],
                [0.60, 0.40],
                [0.55, 0.45],
                [0.10, 0.90],
            ]
        )


def test_regression_compute_classification_metrics_uses_scores_for_auc() -> None:
    """ROC AUC should be computed from probabilities, not hard labels."""
    model = ProbabilityAwareDummyModel()
    X = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])

    result = compute_classification_metrics(model, X, y)

    assert result.accuracy == pytest.approx(0.75)
    assert result.roc_auc == pytest.approx(1.0)


def test_regression_compute_classification_metrics_returns_nan_for_single_class() -> None:
    """Single-class evaluation should not crash ROC AUC computation."""
    model = ProbabilityAwareDummyModel()
    X = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 0, 0])

    result = compute_classification_metrics(model, X, y)

    assert np.isnan(result.roc_auc)
    assert result.confusion_matrix.shape == (2, 2)


def test_regression_plot_feature_importance_supports_linear_models(
    trained_linear_model: tuple,
) -> None:
    """Linear estimators should no longer crash feature-importance plotting."""
    model, X, _ = trained_linear_model

    ax = plot_feature_importance(model, X.columns.tolist(), top_n=3)

    assert ax.get_title() == "Top 3 features"
    plt.close(ax.figure)


def test_regression_plot_feature_importance_supports_tree_models(
    trained_tree_model: tuple,
) -> None:
    """Tree estimators should continue to work for feature-importance plotting."""
    model, X, _ = trained_tree_model

    ax = plot_feature_importance(model, X.columns.tolist(), top_n=3)

    assert ax.get_title() == "Top 3 features"
    plt.close(ax.figure)


def test_regression_plot_feature_importance_validates_feature_name_length(
    trained_linear_model: tuple,
) -> None:
    """Mismatched feature names should raise clearly instead of mislabeling."""
    model, _, _ = trained_linear_model

    with pytest.raises(ValueError, match="feature_names length must match"):
        plot_feature_importance(model, ["f0", "f1"])
