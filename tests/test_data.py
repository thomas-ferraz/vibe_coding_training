"""Regression tests for data loading and sanity checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data import basic_sanity_check, load_and_clean
from src.eda import summarize_dataframe


def test_regression_importing_data_module_does_not_import_matplotlib_pyplot() -> None:
    """Importing data utilities should not pull in plotting backends."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.data; print('matplotlib.pyplot' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.stdout.strip() == "False"


def test_regression_load_and_clean_parses_mixed_supported_date_formats(
    synthetic_patient_df: pd.DataFrame,
    write_csv,
) -> None:
    """Supported mixed formats should produce valid timestamps, not NaT."""
    path = write_csv(synthetic_patient_df)

    result = load_and_clean(path)

    assert pd.api.types.is_datetime64_ns_dtype(result["admission_date"])
    assert result["admission_date"].notna().all()


def test_regression_load_and_clean_rejects_invalid_dates(
    synthetic_patient_df: pd.DataFrame,
    write_csv,
) -> None:
    """Bad dates should raise instead of being silently coerced to NaT."""
    df = synthetic_patient_df.copy()
    df.loc[0, "admission_date"] = "2024/31/99"
    path = write_csv(df, "invalid_dates.csv")

    with pytest.raises(ValueError, match="Failed to parse admission_date"):
        load_and_clean(path)


def test_regression_load_and_clean_validates_required_columns(write_csv) -> None:
    """Missing required columns should raise a clear validation error."""
    df = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "readmission_30d": [0, 1],
        }
    )
    path = write_csv(df, "missing_columns.csv")

    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_clean(path)


def test_regression_load_and_clean_rejects_non_binary_targets(
    synthetic_patient_df: pd.DataFrame,
    write_csv,
) -> None:
    """Unexpected target values should fail explicitly."""
    df = synthetic_patient_df.copy()
    df.loc[0, "readmission_30d"] = 2
    path = write_csv(df, "bad_target.csv")

    with pytest.raises(ValueError, match="readmission_30d must contain only 0/1 values"):
        load_and_clean(path)


def test_regression_load_and_clean_drops_rows_with_missing_target(
    synthetic_patient_df: pd.DataFrame,
    write_csv,
) -> None:
    """Rows with missing target values should be removed deterministically."""
    df = synthetic_patient_df.copy()
    df.loc[[0, 5], "readmission_30d"] = pd.NA
    path = write_csv(df, "missing_target.csv")

    result = load_and_clean(path)

    assert len(result) == len(df) - 2
    assert result["readmission_30d"].notna().all()


def test_regression_basic_sanity_check_raises_without_target(
    synthetic_patient_df: pd.DataFrame,
) -> None:
    """Sanity checks should use explicit errors, not removable asserts."""
    df = synthetic_patient_df.drop(columns=["readmission_30d"])

    with pytest.raises(ValueError, match="target column missing"):
        basic_sanity_check(df)


def test_regression_basic_sanity_check_warns_on_implausible_bmi(
    dirty_patient_df: pd.DataFrame,
) -> None:
    """Implausible BMI values should surface as machine-readable warnings."""
    with pytest.warns(UserWarning, match="implausible BMI values present"):
        assert basic_sanity_check(dirty_patient_df) is True


def test_regression_summarize_dataframe_returns_structured_output(
    synthetic_patient_df: pd.DataFrame,
) -> None:
    """EDA summary should return structured data rather than printing only."""
    summary = summarize_dataframe(synthetic_patient_df)

    assert summary["shape"] == synthetic_patient_df.shape
    assert summary["dtypes"]["patient_id"] == synthetic_patient_df["patient_id"].dtype
    assert summary["missing_per_column"]["patient_id"] == 0
