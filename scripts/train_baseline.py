"""
train_baseline.py
-----------------

Baseline training script. Usage:

    python scripts/train_baseline.py --data data/patients.csv --cv-folds 5
"""

import argparse
import json

from sklearn.linear_model import LogisticRegression

from src.ml_utils import (
    load_and_clean,
    impute_numerics,
    encode_categoricals,
    build_feature_matrix,
    train_test_split_by_patient,
    scale_features,
    evaluate_model,
    cross_validate_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline readmission model.")
    p.add_argument("--data", type=str, default="data/patients.csv",
                   help="path to patients.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=str, default="5",
                   help="number of CV folds")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    df = load_and_clean(args.data)
    df = impute_numerics(df)
    df = encode_categoricals(df)

    train_df, test_df = train_test_split_by_patient(
        df, test_size=args.test_size, random_state=args.random_state
    )

    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)

    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_s, y_train)

    metrics = evaluate_model(model, X_test_s, y_test)
    print("test metrics:", {k: v for k, v in metrics.items() if k != "confusion_matrix"})

    # cross-val
    cv_res = cross_validate_model(model, X_train_s, y_train, cv=args.cv_folds)
    print("cv:", json.dumps(cv_res, default=str))


if __name__ == "__main__":
    main()
