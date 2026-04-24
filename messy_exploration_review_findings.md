# `notebooks/messy_exploration.ipynb` Review Findings

## High

### Preprocessing is fit on the full dataset before splitting

The notebook imputes, one-hot encodes, and scales the full dataframe in cell 8, then performs `train_test_split` in cell 11.

Why it matters: the train/test boundary is violated. Imputation statistics, category vocabulary, and scaling parameters all learn from evaluation rows.

### The main split is not patient-aware

The first training path uses plain `train_test_split` on already-preprocessed arrays.

Why it matters: if patients have multiple rows, the same patient can land in both train and test. That inflates reported metrics.

### The manual cross-validation loop is broken

The notebook builds `X_all_scaled` once on the full dataset, then in the CV loop it re-fits a `StandardScaler` on already-scaled folds.

Why it matters: the loop is logically inconsistent and still starts from leaked features. The reported scores are not trustworthy.

## Medium

### Three competing preprocessing paths exist in one notebook

There is an initial path using `df_v2`, a second path for random forest using `df_final`, and a later patient split on `df_final`.

Why it matters: there is no single source of truth for modeling. Small differences between paths are easy to miss and hard to test.

### EDA and training are interleaved

Exploration cells, preprocessing cells, model fitting, and reporting are mixed together.

Why it matters: the notebook is hard to read, hard to rerun from top to bottom, and hard to refactor into reusable code.

### Imports are scattered across cells

The notebook imports sklearn, seaborn, `train_test_split`, `KFold`, and `StandardScaler` in multiple places.

Why it matters: execution order becomes stateful. A later cell may fail or behave differently depending on whether a previous import cell was run.

### `drop_duplicates()` is applied ad hoc without a stated reason

The random-forest path reloads the CSV and calls `df_final.drop_duplicates()` directly.

Why it matters: this is silent data loss without a business rule or validation. It may remove legitimate repeated observations.

### Determinism is inconsistent

One split uses `random_state=0`, the RF split in cell 17 has no seed, and the CV loop uses plain `KFold(n_splits=5)`.

Why it matters: results vary depending on execution order and rerun timing, which makes debugging and comparison harder.

### A model is trained but never persisted

The notebook ends with a TODO about saving the model, but no model artifact is actually written.

Why it matters: a successful training run cannot be reused, inspected, or deployed.

## Low

### Temporary variable names hide intent

Variables such as `tmp`, `tmp2`, `df_v2`, and `df_final_final` appear throughout the notebook.

Why it matters: these names do not communicate whether the dataframe is raw, cleaned, train-ready, filtered, or final.

### Dead commented-out code is left in place

Old experiments and commented-out save/export code are kept inline in the execution path.

Why it matters: it adds noise and makes it harder to tell which code path is authoritative.

### Magic numbers are embedded in cells

Examples include `47.3` for an age filter, RF hyperparameters like `137` and `9`, and hard-coded CV settings.

Why it matters: there is no clear place to review or tune modeling assumptions.
