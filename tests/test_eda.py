"""Meaningful tests for exploratory data analysis helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.eda import (
    load_data,
    missing_value_report,
    numeric_summary,
    plot_numeric_distributions,
    plot_target_rate_by,
    summarize_dataframe,
    target_rate_by,
)


def test_load_data_parses_supported_mixed_date_formats(
    synthetic_patient_df: pd.DataFrame,
    write_csv,
) -> None:
    """Both supported admission-date formats should parse into valid timestamps."""
    path = write_csv(synthetic_patient_df, "eda_supported_dates.csv")

    result = load_data(str(path))

    assert pd.api.types.is_datetime64_ns_dtype(result["admission_date"])
    assert result["admission_date"].notna().all()


def test_load_data_raises_on_invalid_dates(
    synthetic_patient_df: pd.DataFrame,
    write_csv,
) -> None:
    """Invalid dates should raise instead of becoming silent NaT values."""
    df = synthetic_patient_df.copy()
    df.loc[0, "admission_date"] = "31-99-2024"
    path = write_csv(df, "eda_invalid_dates.csv")

    with pytest.raises(ValueError, match="Failed to parse"):
        load_data(str(path))


def test_load_data_requires_admission_date_column(write_csv) -> None:
    """Loading should fail clearly if the admission date column is absent."""
    df = pd.DataFrame({"readmission_30d": [0, 1]})
    path = write_csv(df, "eda_missing_date.csv")

    with pytest.raises(ValueError, match="admission_date"):
        load_data(str(path))


def test_missing_value_report_returns_counts_percentages_and_descending_sort() -> None:
    """Missing-value summaries should be numerically correct and sorted."""
    df = pd.DataFrame(
        {
            "a": [1.0, None, None, None],
            "b": [1.0, 2.0, None, None],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
    )

    report = missing_value_report(df)

    assert report.loc["a", "missing_count"] == 3
    assert report.loc["a", "missing_percentage"] == 75.0
    assert report.index.tolist() == ["a", "b", "c"]


def test_numeric_summary_includes_only_numeric_columns_and_expected_stats() -> None:
    """The numeric summary should ignore non-numeric columns and expose core stats."""
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "int_col": [1, 2, 3],
            "text": ["x", "y", "z"],
        }
    )

    summary = numeric_summary(df)

    assert set(summary.index) == {"num", "int_col"}
    assert set(summary.columns) == {"min", "max", "mean", "median", "std"}
    assert summary.loc["num", "mean"] == 2.0


def test_target_rate_by_computes_sorted_group_means() -> None:
    """Grouped target rates should be computed as descending means."""
    df = pd.DataFrame(
        {
            "hospital_id": ["H01", "H01", "H02", "H02", "H03"],
            "readmission_30d": [0, 1, 1, 1, 0],
        }
    )

    result = target_rate_by(df, "hospital_id")

    assert result.index.tolist() == ["H02", "H01", "H03"]
    assert result.loc["H02"] == 1.0
    assert result.loc["H01"] == 0.5


def test_target_rate_by_validates_required_columns() -> None:
    """Target-rate grouping should fail clearly for missing inputs."""
    df = pd.DataFrame({"hospital_id": ["H01", "H02"]})

    with pytest.raises(ValueError, match="readmission_30d"):
        target_rate_by(df, "hospital_id")

    with pytest.raises(ValueError, match="grouping column 'missing_col'"):
        target_rate_by(
            pd.DataFrame({"readmission_30d": [0, 1]}),
            "missing_col",
        )


def test_plot_numeric_distributions_runs_for_valid_numeric_columns(monkeypatch) -> None:
    """Histogram plotting should run without crashing on valid numeric input."""
    df = pd.DataFrame({"age": [20, 30, 40], "bmi": [22.0, 27.5, 31.0]})
    called = {"show": False}

    def fake_show() -> None:
        called["show"] = True

    monkeypatch.setattr(plt, "show", fake_show)

    plot_numeric_distributions(df, ["age", "bmi"])

    assert called["show"] is True
    plt.close("all")


def test_plot_numeric_distributions_validates_inputs() -> None:
    """Plotting should fail clearly on empty or missing column selections."""
    df = pd.DataFrame({"age": [20, 30, 40]})

    with pytest.raises(ValueError, match="At least one column is required"):
        plot_numeric_distributions(df, [])

    with pytest.raises(ValueError, match="Column\\(s\\) not found"):
        plot_numeric_distributions(df, ["age", "bmi"])


def test_plot_target_rate_by_runs_for_valid_grouping_column(monkeypatch) -> None:
    """Target-rate plotting should produce a figure for a valid categorical column."""
    df = pd.DataFrame(
        {
            "hospital_id": ["H01", "H01", "H02"],
            "readmission_30d": [0, 1, 1],
        }
    )
    called = {"show": False}

    def fake_show() -> None:
        called["show"] = True

    monkeypatch.setattr(plt, "show", fake_show)

    plot_target_rate_by(df, "hospital_id")

    assert called["show"] is True
    plt.close("all")


def test_summarize_dataframe_returns_expected_structure(
    synthetic_patient_df: pd.DataFrame,
) -> None:
    """Dataframe summary should return shape, dtypes, and missing counts."""
    summary = summarize_dataframe(synthetic_patient_df)

    assert summary["shape"] == synthetic_patient_df.shape
    assert "patient_id" in summary["dtypes"].index
    assert summary["missing_per_column"]["patient_id"] == 0
