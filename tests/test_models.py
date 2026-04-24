"""Smoke tests for model factories."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models import make_logistic_regression, make_random_forest


def test_make_logistic_regression_returns_expected_estimator() -> None:
    """The factory should expose a deterministic logistic regression model."""
    model = make_logistic_regression(random_state=7, max_iter=123)

    assert isinstance(model, LogisticRegression)
    assert model.random_state == 7
    assert model.max_iter == 123


def test_make_random_forest_returns_expected_estimator() -> None:
    """The factory should expose a deterministic random forest model."""
    model = make_random_forest(random_state=7, n_estimators=31, max_depth=5)

    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 7
    assert model.n_estimators == 31
    assert model.max_depth == 5
