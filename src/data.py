"""Data loading and splitting utilities for the patient readmission project."""

from pathlib import Path

import pandas as pd

from src.ml_utils import load_and_clean as _load_and_clean
from src.ml_utils import train_test_split_by_patient as _train_test_split_by_patient


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
    return _load_and_clean(str(path))


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
    return _train_test_split_by_patient(
        df,
        test_size=test_size,
        random_state=random_state,
    )
