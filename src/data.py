"""Data loading and splitting utilities for the patient readmission project."""

from pathlib import Path
import warnings

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _parse_mixed_admission_dates(date_series: pd.Series) -> pd.Series:
    """Parse supported admission date formats and fail on invalid values."""
    date_text = date_series.astype("string").str.strip()
    iso_mask = date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    slash_mask = date_text.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)

    parsed_dates = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")
    parsed_dates.loc[iso_mask] = pd.to_datetime(
        date_text.loc[iso_mask],
        format="%Y-%m-%d",
        errors="coerce",
    )
    parsed_dates.loc[slash_mask] = pd.to_datetime(
        date_text.loc[slash_mask],
        format="%d/%m/%Y",
        errors="coerce",
    )

    invalid_mask = parsed_dates.isna()
    if invalid_mask.any():
        examples = date_series.loc[invalid_mask].head(5).tolist()
        raise ValueError(
            "Failed to parse admission_date values. "
            "Expected YYYY-MM-DD or DD/MM/YYYY. "
            f"Examples: {examples}"
        )

    return parsed_dates


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load the patient dataset and apply validated base cleaning.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with validated dates and binary target.
    """
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]

    required_cols = {"admission_date", "readmission_30d"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df["admission_date"] = _parse_mixed_admission_dates(df["admission_date"])
    df = df[df["readmission_30d"].notna()].copy()
    df["readmission_30d"] = df["readmission_30d"].astype(int)
    if not df["readmission_30d"].isin([0, 1]).all():
        raise ValueError("readmission_30d must contain only 0/1 values.")

    return df


def split_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train and test sets without patient overlap.

    Parameters
    ----------
    df:
        Input dataframe containing ``patient_id``.
    test_size:
        Fraction of patients assigned to the test split.
    random_state:
        Seed for deterministic splitting.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test dataframes.
    """
    if "patient_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'patient_id' column.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    return train_df, test_df


def basic_sanity_check(df: pd.DataFrame) -> bool:
    """Run a few cheap dataframe sanity checks and warn on implausible BMI."""
    if "readmission_30d" not in df.columns:
        raise ValueError("target column missing")
    if not df["readmission_30d"].isin([0, 1]).all():
        raise ValueError("target not binary")
    if "bmi" in df.columns and ((df["bmi"] < 10) | (df["bmi"] > 80)).any():
        warnings.warn("implausible BMI values present", stacklevel=2)
    return True
