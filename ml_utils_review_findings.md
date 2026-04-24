# `src/ml_utils.py` Review Findings

## Header and Imports

### Low: heavy plotting import at module import time

`matplotlib.pyplot` is imported at the top level, even though only `plot_feature_importance()` needs it.

What could go wrong in production: importing this module for a training or batch scoring job also imports matplotlib. That can add startup latency, trigger font/cache warnings, or fail in minimal/headless environments even when no plotting is requested.

### Low: unused imports

`os` and `warnings` are imported but not used.

What could go wrong in production: low direct risk, but it is a maintenance smell. It suggests the module has accumulated stale code and makes it harder to distinguish intentional dependencies from leftovers.

## `load_and_clean(path)`

### High: mixed date parsing can silently corrupt the date column

The function uses:

```python
pd.to_datetime(df["admission_date"], errors="coerce")
```

The CSV contains both `YYYY-MM-DD` and `DD/MM/YYYY` values. With modern pandas parsing behavior, this can silently convert valid slash-format dates into `NaT` depending on inferred format.

What could go wrong in production: valid admission dates may be lost without warning. If downstream features later use admission month, recency, seasonality, or temporal validation, this silently changes the dataset and can distort model behavior or evaluation.

### High: invalid dates are not rejected or reported

The comment says “coerce handles everything,” but `errors="coerce"` hides parse failures instead of surfacing them.

What could go wrong in production: a feed format change, bad extract, or locale issue could produce many `NaT` values while the job still succeeds. That is a silent data quality failure.

### Medium: no validation for required columns

The function assumes `admission_date` and `readmission_30d` exist.

What could go wrong in production: a schema change or incorrect input file will fail with a raw `KeyError`, not a clear validation error. In an automated pipeline, that makes failures harder to triage.

### Medium: target casting can fail or accept bad assumptions without context

The function casts `readmission_30d` directly to `int` after dropping missing values.

What could go wrong in production: if the target contains unexpected strings, floats, labels, or values other than `0` and `1`, the failure will be generic or the data may be accepted without checking the binary classification assumption.

### Medium: rows with missing target are silently dropped

Rows with missing `readmission_30d` are removed without reporting how many were dropped.

What could go wrong in production: if a source-system issue causes many labels to disappear, the training set could shrink or become biased while the job still completes. Missing labels may also be systematic, not random.

### Low: docstring does not match behavior

The docstring says the function “drops obviously malformed rows,” but the implementation only drops rows with missing `readmission_30d`.

What could go wrong in production: callers may assume malformed rows or invalid dates are removed, but they are not. The mismatch increases the chance that downstream code relies on guarantees this function does not provide.

## `impute_numerics(df, cols=None)`

### High: fitting imputation on the full dataset leaks information across splits

This helper both fits and applies the imputer to whatever dataframe it receives.

What could go wrong in production: if this function is called before train/test splitting, the median values are learned using the full dataset, including evaluation rows. That contaminates model evaluation with information from outside the training split.

### Medium: mutates the input dataframe despite promising a copy

The docstring says “Returns a copy of the dataframe; the original frame is not modified,” but the code assigns directly into `df[cols]`.

What could go wrong in production: callers may reuse the original dataframe assuming it is untouched. That can create hard-to-trace side effects across notebooks, feature experiments, or pipeline steps.

### Medium: no validation that requested columns exist and are numeric

The function assumes every column in `cols` exists and is suitable for median imputation.

What could go wrong in production: schema drift or accidental inclusion of a non-numeric column will fail deep inside sklearn with a less clear error than an explicit validation step.

## `encode_categoricals(df, cols=None)`

### High: encoding before splitting leaks category vocabulary from evaluation data

This helper derives dummy columns from the dataframe it receives, based on all categories present at that moment.

What could go wrong in production: if it is run before the train/test split, the training design matrix is informed by categories that appear only in the test set. That is a form of preprocessing leakage.

### High: no stable feature schema between train and inference

`pd.get_dummies()` creates columns based only on categories present in the current dataframe, and this function does not save or enforce a fixed schema.

What could go wrong in production: train, validation, test, and inference datasets can end up with different columns. That makes model serving brittle and can break or silently misalign features unless every caller manually reconciles columns.

### Medium: missing values are converted to the string `"nan"`

The function forces categoricals to `str`, turning missing values into a literal `"nan"` category.

What could go wrong in production: missingness is no longer distinguishable from a real category string of `"nan"`, and missing data is encoded as if it were an observed category rather than handled explicitly.

### Medium: mutates the input dataframe in place

The categorical columns are converted to string on the original dataframe before the one-hot-encoded output is built.

What could go wrong in production: downstream code that expected the original categorical columns and dtypes may behave differently after this call.

### Low: docstring overstates safety

The comment says the conversion avoids dummies “exploding,” but the real issue is data handling semantics, not runtime safety.

What could go wrong in production: this encourages callers to treat the string cast as harmless when it materially changes how missingness is represented.

## `train_test_split_by_patient(df, test_size=0.2, random_state=42)`

### High: function name and docstring promise a group-aware split, but the code is not group-aware

The implementation calls plain `train_test_split(df, ...)` and does nothing with `patient_id`.

What could go wrong in production: if there are multiple rows per patient, the same patient can appear in both train and test. That creates leakage and inflates evaluation metrics.

### Medium: no validation that `patient_id` exists despite the contract

The docstring says the dataframe “Must contain a `patient_id` column,” but the function never checks it because it never uses it.

What could go wrong in production: callers think patient grouping is enforced when it is not. Missing `patient_id` would go unnoticed, which hides a serious evaluation flaw.

## `build_feature_matrix(df, target_col="readmission_30d")`

### Medium: silently drops all non-numeric features

After dropping a few named columns, the function keeps only numeric and bool dtypes.

What could go wrong in production: any still-unencoded categorical or datetime-derived features disappear silently instead of triggering a validation error. That can make feature loss invisible.

### Medium: assumes the target column exists and is valid

The function accesses `df[target_col]` directly and casts it to `int`.

What could go wrong in production: missing or malformed targets produce generic failures, and non-binary targets are accepted without explicit checking.

### Low: hard-coded removal of `admission_date`

The function always drops `admission_date` if present.

What could go wrong in production: temporal signal is unavailable unless another part of the pipeline has already expanded that column. If a caller expects raw dates to be transformed later, this function discards them prematurely.

## `scale_features(X_train, X_test=None)`

### Medium: easy to misuse in a way that leaks test distribution

This helper fits the scaler immediately on the `X_train` object it receives and can also transform `X_test`.

What could go wrong in production: if a caller passes the full dataset as `X_train`, scaling parameters are learned from all rows. The helper itself does not guard against that misuse.

### Low: loses dataframe metadata

`StandardScaler` returns numpy arrays, not dataframes with column names and indexes.

What could go wrong in production: downstream debugging, feature inspection, and alignment checks become harder because feature names are stripped away.

## `evaluate_model(model, X_test, y_test)`

### High: ROC AUC is computed from hard class predictions instead of scores or probabilities

The function does:

```python
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
```

What could go wrong in production: the reported AUC is not the usual ranking-based ROC AUC. It collapses the model output to 0/1 decisions first, which can materially understate or distort model quality.

### Medium: can fail when the test set contains only one class

`roc_auc_score` requires both classes to be present in `y_test`.

What could go wrong in production: on small or skewed splits, evaluation can raise at runtime instead of returning a controlled result or clearer message.

## `plot_feature_importance(model, feature_names, top_n=15, figsize=(8, 6))`

### High: docstring claims support for linear models, but implementation only works for estimators with `feature_importances_`

The code accesses `model.feature_importances_` directly.

What could go wrong in production: passing a linear model such as logistic regression raises `AttributeError`, despite the docstring explicitly saying linear models are supported.

### Medium: no validation that `feature_names` matches model dimensionality

The function indexes into `feature_names` using the importance ranking without checking lengths.

What could go wrong in production: if feature names and model coefficients/importances are out of sync, labels will be wrong or indexing will fail.

## `cross_validate_model(model, X, y, cv=5, scoring="roc_auc")`

### High: cross-validation is not group-aware, so repeated patients can leak across folds

The function uses plain `KFold`.

What could go wrong in production: if the dataset has multiple rows per patient, information from a patient can appear in both training and validation folds, inflating validation performance.

### High: uses plain `KFold` instead of stratified classification folds

For a binary classification problem, the default here does not preserve class balance across folds.

What could go wrong in production: some folds may be badly imbalanced or even degenerate, making ROC AUC unstable or undefined.

### Medium: `KFold` is unshuffled by default

The function does not set `shuffle=True` or a `random_state`.

What could go wrong in production: if rows are ordered by time, hospital, or extraction order, fold quality depends on file ordering rather than randomized sampling.

## `summarize_dataframe(df)`

### Low: prints output instead of returning structured data

This helper only prints shape, dtypes, and missing counts.

What could go wrong in production: it is hard to test, log consistently, or reuse in dashboards and automated reports because the information is not returned as data.

## `basic_sanity_check(df)`

### Medium: uses `assert` for data validation

Critical checks are implemented with `assert`.

What could go wrong in production: Python can disable assertions with optimization flags, which would remove these checks entirely.

### Low: BMI issue is only printed, not surfaced in a machine-readable way

The function prints a warning for implausible BMI values and still returns `True`.

What could go wrong in production: automated jobs and monitoring systems may miss the warning, so data quality issues can pass unnoticed.
