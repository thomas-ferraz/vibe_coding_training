"""Exploratory data analysis helpers for the patient readmission dataset."""

import math

import matplotlib.pyplot as plt
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load patient data and parse mixed-format admission dates.

    The source file mixes ISO dates (``YYYY-MM-DD``) with European slash dates
    (``DD/MM/YYYY``). This function parses both formats explicitly and raises a
    ``ValueError`` if any value cannot be parsed, so invalid dates do not become
    silent ``NaT`` values.

    Parameters
    ----------
    path:
        Path to the patient CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data with ``admission_date`` converted to datetime64.
    """
    df = pd.read_csv(path)

    if "admission_date" not in df.columns:
        raise ValueError("CSV must contain an 'admission_date' column.")

    raw_dates = df["admission_date"]
    date_text = raw_dates.astype("string").str.strip()

    iso_mask = date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    slash_mask = date_text.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)

    parsed_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
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
        examples = raw_dates.loc[invalid_mask].head(5).tolist()
        count = int(invalid_mask.sum())
        raise ValueError(
            f"Failed to parse {count} admission_date value(s). "
            f"Expected YYYY-MM-DD or DD/MM/YYYY. Examples: {examples}"
        )

    df["admission_date"] = parsed_dates
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages for every column.

    Parameters
    ----------
    df:
        DataFrame to summarize.

    Returns
    -------
    pd.DataFrame
        Report indexed by column name with ``missing_count`` and
        ``missing_percentage``, sorted from most to least missing.
    """
    missing_count = df.isna().sum()
    missing_percentage = missing_count / len(df) * 100 if len(df) else missing_count

    report = pd.DataFrame(
        {
            "missing_count": missing_count.astype(int),
            "missing_percentage": missing_percentage.astype(float),
        }
    )
    return report.sort_values(
        by=["missing_count", "missing_percentage"],
        ascending=False,
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for numeric columns.

    Parameters
    ----------
    df:
        DataFrame containing numeric columns to summarize.

    Returns
    -------
    pd.DataFrame
        One row per numeric column with min, max, mean, median, and standard
        deviation.
    """
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.agg(["min", "max", "mean", "median", "std"]).transpose()


def target_rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    """Calculate the mean 30-day readmission rate grouped by a category.

    Parameters
    ----------
    df:
        DataFrame containing ``readmission_30d`` and the grouping column.
    col:
        Categorical column to group by.

    Returns
    -------
    pd.Series
        Mean ``readmission_30d`` for each category, sorted descending.
    """
    if "readmission_30d" not in df.columns:
        raise ValueError("DataFrame must contain a 'readmission_30d' column.")
    if col not in df.columns:
        raise ValueError(f"DataFrame must contain the grouping column '{col}'.")

    return df.groupby(col, dropna=False)["readmission_30d"].mean().sort_values(
        ascending=False
    )


def plot_numeric_distributions(df: pd.DataFrame, cols: list[str]) -> None:
    """Plot histograms for numeric columns in a compact grid.

    Parameters
    ----------
    df:
        DataFrame containing the columns to plot.
    cols:
        Numeric column names to visualize.

    Returns
    -------
    None
        Displays a matplotlib figure.
    """
    if not cols:
        raise ValueError("At least one column is required.")

    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not found in DataFrame: {missing_cols}")

    n_cols = min(3, len(cols))
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if len(cols) == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    for ax, col in zip(axes_list, cols):
        df[col].dropna().hist(ax=ax, bins=30)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for ax in axes_list[len(cols) :]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_target_rate_by(df: pd.DataFrame, col: str) -> None:
    """Plot the 30-day readmission rate for each category in a column.

    Parameters
    ----------
    df:
        DataFrame containing ``readmission_30d`` and the grouping column.
    col:
        Categorical column to group by.

    Returns
    -------
    None
        Displays a matplotlib bar chart.
    """
    rates = target_rate_by(df, col)

    ax = rates.plot(kind="bar", figsize=(8, 4))
    ax.set_title(f"30-day readmission rate by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Readmission rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
