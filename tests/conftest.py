"""Shared pytest fixtures for deterministic synthetic test data."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


NUMERIC_COLS = [
    "age",
    "bmi",
    "num_prior_admissions",
    "length_of_stay",
    "lab_sodium",
    "lab_creatinine",
    "bmi_implausible",
    "lab_creatinine_negative",
]

CATEGORICAL_COLS = ["sex", "diagnosis_code", "hospital_id"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


@pytest.fixture
def feature_columns() -> dict[str, list[str]]:
    """Return the canonical feature column groups used in the project."""
    return {
        "numeric": NUMERIC_COLS,
        "categorical": CATEGORICAL_COLS,
        "all": FEATURE_COLS,
    }


@pytest.fixture
def synthetic_patient_df() -> pd.DataFrame:
    """Build a small deterministic dataframe with repeated patients."""
    rows: list[dict[str, object]] = []
    for patient_idx in range(1, 13):
        patient_id = 10_000 + patient_idx
        target = patient_idx % 2
        diagnosis = "I10" if target == 0 else "E11.9"
        hospital = f"H0{(patient_idx % 3) + 1}"
        sex = "F" if patient_idx % 2 == 0 else "M"
        for visit_idx in range(2):
            rows.append(
                {
                    "patient_id": patient_id,
                    "age": 40 + patient_idx + visit_idx,
                    "sex": sex,
                    "bmi": 24.0 + patient_idx / 10 + visit_idx,
                    "num_prior_admissions": visit_idx,
                    "length_of_stay": 2 + target + visit_idx,
                    "diagnosis_code": diagnosis,
                    "hospital_id": hospital,
                    "lab_sodium": 135.0 + patient_idx,
                    "lab_creatinine": 0.7 + target * 0.2 + visit_idx * 0.05,
                    "admission_date": (
                        f"2024-01-{patient_idx:02d}"
                        if visit_idx == 0
                        else f"{patient_idx:02d}/02/2024"
                    ),
                    "readmission_30d": target,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def dirty_patient_df(synthetic_patient_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with targeted data-quality issues for cleaning tests."""
    df = synthetic_patient_df.copy()
    df.loc[0, "bmi"] = 2.5
    df.loc[1, "bmi"] = 120.0
    df.loc[2, "lab_creatinine"] = -0.4
    df.loc[3, "lab_creatinine"] = -1.1
    df.loc[4, "sex"] = pd.NA
    df.loc[5, "lab_sodium"] = pd.NA
    return df


@pytest.fixture
def ordered_cv_df() -> pd.DataFrame:
    """Create ordered target blocks that would stress bad CV splitting."""
    rows: list[dict[str, object]] = []
    for idx in range(40):
        target = 0 if idx < 20 else 1
        rows.append(
            {
                "patient_id": 20_000 + idx,
                "age": 30 + target * 20,
                "bmi": 22.0 + target * 4,
                "num_prior_admissions": target,
                "length_of_stay": 2 + target,
                "lab_sodium": 136.0 + target,
                "lab_creatinine": 0.8 + target * 0.3,
                "bmi_implausible": False,
                "lab_creatinine_negative": False,
                "sex": "F" if idx % 2 == 0 else "M",
                "diagnosis_code": "I10" if target == 0 else "E11.9",
                "hospital_id": "H01" if idx % 2 == 0 else "H02",
                "admission_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
                "readmission_30d": target,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def write_csv(tmp_path: Path):
    """Return a helper that writes a dataframe to a temporary CSV path."""

    def _write(df: pd.DataFrame, name: str = "patients.csv") -> Path:
        path = tmp_path / name
        df.to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def toy_feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """Return a small numeric dataset for model fitting tests."""
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    target = pd.Series(y, name="target")
    return frame, target


@pytest.fixture
def trained_linear_model(
    toy_feature_matrix: tuple[pd.DataFrame, pd.Series],
) -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """Fit a deterministic logistic regression model on the toy dataset."""
    X, y = toy_feature_matrix
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_tree_model(
    toy_feature_matrix: tuple[pd.DataFrame, pd.Series],
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """Fit a deterministic random forest model on the toy dataset."""
    X, y = toy_feature_matrix
    model = RandomForestClassifier(n_estimators=25, random_state=42)
    model.fit(X, y)
    return model, X, y
