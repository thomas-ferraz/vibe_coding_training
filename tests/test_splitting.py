"""Regression tests for group-aware data splitting."""

import pandas as pd
import pytest

from src.data import split_by_patient


def test_regression_split_by_patient_has_no_patient_overlap(
    synthetic_patient_df: pd.DataFrame,
) -> None:
    """Repeated patients must not leak across train and test."""
    train_df, test_df = split_by_patient(
        synthetic_patient_df,
        test_size=0.25,
        random_state=42,
    )

    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    assert train_patients.isdisjoint(test_patients)


def test_regression_split_by_patient_keeps_all_rows_for_each_patient_together(
    synthetic_patient_df: pd.DataFrame,
) -> None:
    """Every patient group should land entirely in one split."""
    train_df, test_df = split_by_patient(
        synthetic_patient_df,
        test_size=0.25,
        random_state=42,
    )

    train_counts = train_df.groupby("patient_id").size()
    test_counts = test_df.groupby("patient_id").size()
    original_counts = synthetic_patient_df.groupby("patient_id").size()

    for patient_id, count in original_counts.items():
        split_count = train_counts.get(patient_id, 0) + test_counts.get(patient_id, 0)
        assert split_count == count
        assert not (patient_id in train_counts and patient_id in test_counts)


def test_regression_split_by_patient_requires_patient_id() -> None:
    """The grouping column should be validated explicitly."""
    df = pd.DataFrame({"readmission_30d": [0, 1, 0]})

    with pytest.raises(ValueError, match="patient_id"):
        split_by_patient(df)


def test_regression_split_by_patient_is_deterministic_for_same_seed(
    synthetic_patient_df: pd.DataFrame,
) -> None:
    """The same seed should reproduce the same patient assignment."""
    train_a, test_a = split_by_patient(synthetic_patient_df, random_state=7)
    train_b, test_b = split_by_patient(synthetic_patient_df, random_state=7)

    assert train_a["patient_id"].tolist() == train_b["patient_id"].tolist()
    assert test_a["patient_id"].tolist() == test_b["patient_id"].tolist()
