"""
ml_utils.py
-----------

Helper functions for the patient readmission project.
Used from the exploration notebook and from scripts/train_baseline.py.

Author: someone on the team, a while ago
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


DEFAULT_NUMERIC_COLS = [
    "age",
    "bmi",
    "num_prior_admissions",
    "length_of_stay",
    "lab_sodium",
    "lab_creatinine",
]

DEFAULT_CATEGORICAL_COLS = [
    "sex",
    "diagnosis_code",
    "hospital_id",
]


def _parse_mixed_admission_dates(date_series):
    """Parse supported admission date formats and fail on invalid values."""
    date_text = date_series.astype("string").str.strip()
    iso_mask = date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    slash_mask = date_text.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)

    parsed_dates = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")
    parsed_dates.loc[iso_mask] = pd.to_datetime(
        date_text.loc[iso_mask], format="%Y-%m-%d", errors="coerce"
    )
    parsed_dates.loc[slash_mask] = pd.to_datetime(
        date_text.loc[slash_mask], format="%d/%m/%Y", errors="coerce"
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


def load_and_clean(path):
    """
    Load the patients CSV and return a cleaned DataFrame.

    Parses `admission_date` to a proper datetime column and drops
    obviously malformed rows. Safe to call multiple times.

    Parameters
    ----------
    path : str
        Path to patients.csv

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)

    # normalise column names just in case
    df.columns = [c.strip() for c in df.columns]

    required_cols = {"admission_date", "readmission_30d"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df["admission_date"] = _parse_mixed_admission_dates(df["admission_date"])

    # drop rows with no target
    df = df[df["readmission_30d"].notna()].copy()
    df["readmission_30d"] = df["readmission_30d"].astype(int)
    if not df["readmission_30d"].isin([0, 1]).all():
        raise ValueError("readmission_30d must contain only 0/1 values.")

    return df


def impute_numerics(df, cols=None, imputer=None, return_imputer=False):
    """
    Impute missing values in the given numeric columns using the median.

    Returns a copy of the dataframe with imputed values; the original
    frame is not modified. If an imputer is provided, it is used in
    transform-only mode.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str] or None
        Columns to impute. Defaults to the standard numeric list.
    """
    if cols is None:
        cols = DEFAULT_NUMERIC_COLS

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing numeric columns: {missing_cols}")

    out = df.copy()
    fitted_imputer = imputer or SimpleImputer(strategy="median")
    if imputer is None:
        out[cols] = fitted_imputer.fit_transform(out[cols])
    else:
        out[cols] = fitted_imputer.transform(out[cols])

    if return_imputer:
        return out, fitted_imputer
    return out


def encode_categoricals(df, cols=None, category_levels=None, return_levels=False):
    """
    One-hot encode the categorical columns and return a new frame.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str] or None

    Returns
    -------
    pd.DataFrame with the categorical columns replaced by dummies.
    """
    if cols is None:
        cols = DEFAULT_CATEGORICAL_COLS

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing categorical columns: {missing_cols}")

    out = df.copy()
    encoded_parts = []
    fitted_levels = {}

    for col in cols:
        values = out[col].astype("string").fillna("__MISSING__")

        if category_levels is None:
            levels = sorted(values.unique().tolist())
        else:
            levels = list(category_levels[col])

        fitted_levels[col] = levels
        categorical = pd.Series(
            pd.Categorical(values, categories=levels, ordered=False),
            index=out.index,
            name=col,
            dtype=CategoricalDtype(categories=levels),
        )
        encoded_parts.append(pd.get_dummies(categorical, prefix=col))

    encoded = pd.concat(encoded_parts, axis=1)
    result = pd.concat([out.drop(columns=cols), encoded], axis=1)

    if return_levels:
        return result, fitted_levels
    return result


def train_test_split_by_patient(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into train and test, making sure that no patient
    appears in both sets (group-aware split on `patient_id`).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a `patient_id` column.
    test_size : float
    random_state : int

    Returns
    -------
    (train_df, test_df)
    """
    if "patient_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'patient_id' column.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    return train_df, test_df


def build_feature_matrix(df, target_col="readmission_30d"):
    """
    Separate features and target. Drops non-feature columns such as
    `patient_id` and `admission_date`.
    """
    drop_cols = [target_col, "patient_id", "admission_date"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)
    # Keep only numeric columns (after encoding, everything should be numeric)
    X = X.select_dtypes(include=[np.number, "bool"])
    return X, y


def scale_features(X_train, X_test=None):
    """
    Fit a StandardScaler on X_train and transform both sets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fitted classification model and return a dict of metrics.

    Returns a dict with accuracy and roc_auc which are the two standard
    metrics for binary classification.

    Parameters
    ----------
    model : fitted sklearn classifier
    X_test : array-like
    y_test : array-like of 0/1

    Returns
    -------
    dict
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def plot_feature_importance(model, feature_names, top_n=15, figsize=(8, 6)):
    """
    Plot the top N most important features for a fitted model.

    Works with tree-based models and linear models.
    """
    import matplotlib.pyplot as plt

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
    else:
        raise ValueError(
            "Model must expose either feature_importances_ or coef_."
        )

    if len(feature_names) != len(importances):
        raise ValueError("feature_names length must match the number of features.")

    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=figsize)
    plt.barh(range(len(order)), importances[order][::-1])
    plt.yticks(range(len(order)), [feature_names[i] for i in order][::-1])
    plt.xlabel("importance")
    plt.title("Top {} features".format(top_n))
    plt.tight_layout()
    return plt.gca()


def cross_validate_model(model, X, y, cv=5, scoring="roc_auc", groups=None, random_state=42):
    """
    Run k-fold cross validation and return the list of fold scores
    plus the mean.

    Parameters
    ----------
    model : unfitted sklearn estimator
    X : array-like
    y : array-like
    cv : int
        Number of folds (default 5).
    scoring : str

    Returns
    -------
    dict with keys `fold_scores` and `mean_score`.
    """
    if groups is not None:
        splitter = StratifiedGroupKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        scores = cross_val_score(
            model, X, y, cv=splitter, groups=groups, scoring=scoring
        )
    else:
        splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring)
    return {
        "fold_scores": list(scores),
        "mean_score": float(np.mean(scores)),
    }


def summarize_dataframe(df):
    """
    Quick summary helpful for EDA. Prints shape, dtypes, missing counts.
    """
    print("shape:", df.shape)
    print("dtypes:")
    print(df.dtypes)
    print("missing per column:")
    print(df.isna().sum())
    return None


def basic_sanity_check(df):
    """
    A few cheap sanity checks on the dataframe. Raises on obvious issues.
    """
    assert "readmission_30d" in df.columns, "target column missing"
    assert df["readmission_30d"].isin([0, 1]).all(), "target not binary"
    if "bmi" in df.columns:
        # don't be too strict about outliers here, just a heads-up
        if (df["bmi"] < 10).any() or (df["bmi"] > 80).any():
            print("warning: implausible BMI values present")
    return True
