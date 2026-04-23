"""
ml_utils.py
-----------

Helper functions for the patient readmission project.
Used from the exploration notebook and from scripts/train_baseline.py.

Author: someone on the team, a while ago
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


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

    # parse dates — csv has mixed formats, coerce handles everything
    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")

    # drop rows with no target
    df = df[df["readmission_30d"].notna()].copy()
    df["readmission_30d"] = df["readmission_30d"].astype(int)

    return df


def impute_numerics(df, cols=None):
    """
    Impute missing values in the given numeric columns using the median.

    Returns a copy of the dataframe with imputed values; the original
    frame is not modified.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str] or None
        Columns to impute. Defaults to the standard numeric list.
    """
    if cols is None:
        cols = DEFAULT_NUMERIC_COLS

    imputer = SimpleImputer(strategy="median")
    df[cols] = imputer.fit_transform(df[cols])
    return df


def encode_categoricals(df, cols=None):
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

    # Make sure categoricals are strings (NaN -> 'nan' string so dummies don't explode)
    for c in cols:
        df[c] = df[c].astype(str)

    dummies = pd.get_dummies(df[cols], prefix=cols)
    out = pd.concat([df.drop(columns=cols), dummies], axis=1)
    return out


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
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
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

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

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
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=figsize)
    plt.barh(range(len(order)), importances[order][::-1])
    plt.yticks(range(len(order)), [feature_names[i] for i in order][::-1])
    plt.xlabel("importance")
    plt.title("Top {} features".format(top_n))
    plt.tight_layout()
    return plt.gca()


def cross_validate_model(model, X, y, cv=5, scoring="roc_auc"):
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
    kf = KFold(n_splits=cv)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
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
