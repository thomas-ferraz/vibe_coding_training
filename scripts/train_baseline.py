"""
train_baseline.py
-----------------

Baseline training script. Usage:

    python scripts/train_baseline.py --data data/patients.csv --cv-folds 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_and_clean
from src.io import save_model
from src.models import make_logistic_regression
from src.preprocess import build_preprocessor, clean_bmi, clean_creatinine
from src.training import cross_validate_estimator, fit_and_evaluate


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Train baseline readmission model.")
    p.add_argument("--data", type=str, default="data/patients.csv",
                   help="path to patients.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5,
                   help="number of CV folds")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--output-model",
        type=str,
        default="artifacts/baseline_logreg.joblib",
        help="path to save the fitted baseline model pipeline",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)

    df = load_and_clean(args.data)
    df = clean_creatinine(clean_bmi(df))

    numeric_cols = [
        "age",
        "bmi",
        "num_prior_admissions",
        "length_of_stay",
        "lab_sodium",
        "lab_creatinine",
        "bmi_implausible",
        "lab_creatinine_negative",
    ]
    categorical_cols = ["sex", "diagnosis_code", "hospital_id"]
    feature_cols = numeric_cols + categorical_cols
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    estimator = make_logistic_regression(random_state=args.random_state)

    training_result = fit_and_evaluate(
        df=df,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
        estimator=estimator,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = {
        "accuracy": training_result.evaluation.accuracy,
        "roc_auc": training_result.evaluation.roc_auc,
    }
    output_model_path = save_model(training_result.pipeline, args.output_model)
    print("test metrics:", metrics)

    # cross-val
    cv_res = cross_validate_estimator(
        df=df,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
        estimator=estimator,
        cv=args.cv_folds,
        random_state=args.random_state,
    )
    print("cv:", json.dumps(cv_res, default=str))
    print("saved model:", output_model_path)

    return {
        "metrics": metrics,
        "cv": cv_res,
        "model_path": str(output_model_path),
    }


if __name__ == "__main__":
    main()
