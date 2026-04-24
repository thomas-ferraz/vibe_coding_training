"""Regression tests for preprocessing helpers and shared transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_preprocessor, clean_bmi, clean_creatinine


def test_regression_clean_bmi_flags_implausible_values_without_mutating_input(
    dirty_patient_df: pd.DataFrame,
) -> None:
    """BMI cleaning should preserve the caller dataframe and mark bad rows."""
    original = dirty_patient_df.copy(deep=True)

    result = clean_bmi(dirty_patient_df)

    assert dirty_patient_df.equals(original)
    assert result["bmi_implausible"].sum() == 2
    assert result.loc[[0, 1], "bmi"].isna().all()


def test_regression_clean_creatinine_flags_negative_values_without_mutating_input(
    dirty_patient_df: pd.DataFrame,
) -> None:
    """Creatinine cleaning should preserve the caller dataframe and mark bad rows."""
    original = dirty_patient_df.copy(deep=True)

    result = clean_creatinine(dirty_patient_df)

    assert dirty_patient_df.equals(original)
    assert result["lab_creatinine_negative"].sum() == 2
    assert result.loc[[2, 3], "lab_creatinine"].isna().all()


def test_regression_preprocessor_is_fit_on_train_statistics_only(
    feature_columns: dict[str, list[str]],
) -> None:
    """Train-only fitting should not leak test values into numeric statistics."""
    train_df = pd.DataFrame(
        {
            "age": [40, 41],
            "bmi": [25.0, 26.0],
            "num_prior_admissions": [0, 1],
            "length_of_stay": [2, 3],
            "lab_sodium": [10.0, 12.0],
            "lab_creatinine": [0.9, 1.1],
            "bmi_implausible": [False, False],
            "lab_creatinine_negative": [False, False],
            "sex": ["M", "F"],
            "diagnosis_code": ["I10", "E11.9"],
            "hospital_id": ["H01", "H02"],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [50],
            "bmi": [27.0],
            "num_prior_admissions": [1],
            "length_of_stay": [4],
            "lab_sodium": [1000.0],
            "lab_creatinine": [1.0],
            "bmi_implausible": [False],
            "lab_creatinine_negative": [False],
            "sex": ["M"],
            "diagnosis_code": ["I10"],
            "hospital_id": ["H01"],
        }
    )

    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    preprocessor.fit(train_df)
    transformed_test = preprocessor.transform(test_df)

    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    lab_sodium_idx = feature_columns["numeric"].index("lab_sodium")

    assert imputer.statistics_[lab_sodium_idx] == pytest.approx(11.0)
    assert np.isfinite(np.asarray(transformed_test)).all()


def test_regression_preprocessor_handles_unseen_test_categories_with_stable_shape(
    feature_columns: dict[str, list[str]],
) -> None:
    """Test-time unseen categories should not break the transformed feature schema."""
    train_df = pd.DataFrame(
        {
            "age": [40, 41],
            "bmi": [25.0, 26.0],
            "num_prior_admissions": [0, 1],
            "length_of_stay": [2, 3],
            "lab_sodium": [137.0, 138.0],
            "lab_creatinine": [0.9, 1.1],
            "bmi_implausible": [False, False],
            "lab_creatinine_negative": [False, False],
            "sex": ["M", "F"],
            "diagnosis_code": ["I10", "E11.9"],
            "hospital_id": ["H01", "H02"],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [42],
            "bmi": [27.0],
            "num_prior_admissions": [2],
            "length_of_stay": [5],
            "lab_sodium": [139.0],
            "lab_creatinine": [1.0],
            "bmi_implausible": [False],
            "lab_creatinine_negative": [False],
            "sex": ["other"],
            "diagnosis_code": ["Z99.9"],
            "hospital_id": ["H99"],
        }
    )

    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    preprocessor.fit(train_df)

    train_shape = preprocessor.transform(train_df).shape[1]
    test_shape = preprocessor.transform(test_df).shape[1]

    assert train_shape == test_shape


def test_regression_preprocessor_does_not_create_literal_nan_category_names(
    feature_columns: dict[str, list[str]],
) -> None:
    """Missing categoricals should be imputed, not encoded as a 'nan' category."""
    df = pd.DataFrame(
        {
            "age": [40, 41, 42],
            "bmi": [25.0, 26.0, 27.0],
            "num_prior_admissions": [0, 1, 2],
            "length_of_stay": [2, 3, 4],
            "lab_sodium": [137.0, 138.0, 139.0],
            "lab_creatinine": [0.9, 1.1, 1.0],
            "bmi_implausible": [False, False, False],
            "lab_creatinine_negative": [False, False, False],
            "sex": ["M", pd.NA, "F"],
            "diagnosis_code": ["I10", "I10", pd.NA],
            "hospital_id": ["H01", "H01", "H02"],
        }
    )

    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    preprocessor.fit(df)
    output_names = preprocessor.get_feature_names_out().tolist()

    assert all("nan" not in name.lower() for name in output_names)


def test_regression_preprocessor_does_not_mutate_source_dataframe(
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """Shared preprocessing should leave the caller dataframe untouched."""
    df = synthetic_patient_df.copy()
    df["bmi_implausible"] = False
    df["lab_creatinine_negative"] = False
    original = df.copy(deep=True)

    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    _ = preprocessor.fit_transform(df[feature_columns["all"]])

    assert df.equals(original)


def test_regression_preprocessor_raises_when_configured_columns_are_missing(
    feature_columns: dict[str, list[str]],
) -> None:
    """Missing configured columns should fail instead of being silently ignored."""
    df = pd.DataFrame(
        {
            "age": [40, 41],
            "bmi": [25.0, 26.0],
        }
    )
    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )

    with pytest.raises(ValueError):
        preprocessor.fit(df)


def test_preprocessor_happy_path_preserves_row_count_and_outputs_numeric_features(
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """The shared preprocessor should emit a numeric matrix with one row per input."""
    df = synthetic_patient_df.copy()
    df["bmi_implausible"] = False
    df["lab_creatinine_negative"] = False

    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    transformed = preprocessor.fit_transform(df[feature_columns["all"]])

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= len(feature_columns["numeric"])
    assert np.issubdtype(np.asarray(transformed).dtype, np.number)


def test_preprocessor_happy_path_exposes_expected_feature_names(
    synthetic_patient_df: pd.DataFrame,
    feature_columns: dict[str, list[str]],
) -> None:
    """Feature names should include numeric passthrough names and encoded categoricals."""
    df = synthetic_patient_df.copy()
    df["bmi_implausible"] = False
    df["lab_creatinine_negative"] = False

    preprocessor = build_preprocessor(
        feature_columns["numeric"],
        feature_columns["categorical"],
    )
    preprocessor.fit(df[feature_columns["all"]])
    output_names = preprocessor.get_feature_names_out().tolist()

    assert "numeric__age" in output_names
    assert any(name.startswith("categorical__sex_") for name in output_names)
    assert any(name.startswith("categorical__hospital_id_") for name in output_names)
