"""Model factory helpers for the patient readmission project."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_logistic_regression(
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Create the baseline logistic regression model."""
    return LogisticRegression(max_iter=max_iter, random_state=random_state)


def make_random_forest(
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 9,
) -> RandomForestClassifier:
    """Create the baseline random forest model."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
