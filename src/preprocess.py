"""Preprocessing helpers for patient readmission modeling."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def normalize_missing_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas missing scalars to numpy nan for sklearn compatibility."""
    return frame.replace({pd.NA: np.nan})


def clean_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and remove medically implausible BMI values.

    Values below 10 or above 80 are treated as medically implausible for this
    adult patient dataset. The function preserves rows, adds a
    ``bmi_implausible`` flag, and sets implausible ``bmi`` values to missing so
    the modeling preprocessor can impute them later.

    Parameters
    ----------
    df:
        DataFrame containing a ``bmi`` column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with ``bmi_implausible`` added and implausible BMI
        values set to ``pd.NA``.
    """
    if "bmi" not in df.columns:
        raise ValueError("DataFrame must contain a 'bmi' column.")

    out = df.copy()
    implausible_mask = out["bmi"].notna() & ((out["bmi"] < 10) | (out["bmi"] > 80))
    out["bmi_implausible"] = implausible_mask
    out.loc[implausible_mask, "bmi"] = pd.NA
    return out


def clean_creatinine(df: pd.DataFrame) -> pd.DataFrame:
    """Handle negative creatinine values as data errors.

    Negative creatinine values are not physiologically valid, so this function
    preserves rows, adds a ``lab_creatinine_negative`` flag, and sets negative
    ``lab_creatinine`` values to missing for later imputation.

    Parameters
    ----------
    df:
        DataFrame containing a ``lab_creatinine`` column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with ``lab_creatinine_negative`` added and negative
        creatinine values set to ``pd.NA``.
    """
    if "lab_creatinine" not in df.columns:
        raise ValueError("DataFrame must contain a 'lab_creatinine' column.")

    out = df.copy()
    negative_mask = out["lab_creatinine"].notna() & (out["lab_creatinine"] < 0)
    out["lab_creatinine_negative"] = negative_mask
    out.loc[negative_mask, "lab_creatinine"] = pd.NA
    return out


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Build a scikit-learn preprocessor for numeric and categorical features.

    Numerics are median-imputed and standard-scaled. Categoricals are imputed
    with the most frequent value and one-hot encoded with unknown categories
    ignored at transform time.

    Parameters
    ----------
    numeric_cols:
        Names of numeric feature columns.
    categorical_cols:
        Names of categorical feature columns.

    Returns
    -------
    ColumnTransformer
        Transformer ready to plug into a scikit-learn ``Pipeline``.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "normalize_missing",
                FunctionTransformer(
                    normalize_missing_categoricals,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ]
    )
