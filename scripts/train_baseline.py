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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml_utils import (
    load_and_clean,
    impute_numerics,
    encode_categoricals,
    build_feature_matrix,
    train_test_split_by_patient,
    evaluate_model,
    cross_validate_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline readmission model.")
    p.add_argument("--data", type=str, default="data/patients.csv",
                   help="path to patients.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5,
                   help="number of CV folds")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    df = load_and_clean(args.data)
    train_df, test_df = train_test_split_by_patient(
        df, test_size=args.test_size, random_state=args.random_state
    )
    train_groups = train_df["patient_id"].copy()

    train_df, numeric_imputer = impute_numerics(train_df, return_imputer=True)
    test_df = impute_numerics(test_df, imputer=numeric_imputer)

    train_df, category_levels = encode_categoricals(train_df, return_levels=True)
    test_df = encode_categoricals(test_df, category_levels=category_levels)

    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print("test metrics:", {k: v for k, v in metrics.items() if k != "confusion_matrix"})

    # cross-val
    cv_res = cross_validate_model(
        model,
        X_train,
        y_train,
        cv=args.cv_folds,
        groups=train_groups,
        random_state=args.random_state,
    )
    print("cv:", json.dumps(cv_res, default=str))


if __name__ == "__main__":
    main()
