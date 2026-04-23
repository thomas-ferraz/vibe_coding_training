# TP1 — Prompting: generate new code from scratch


**Files you touch:** new files under `src/` and/or a new notebook. **Do NOT modify `src/ml_utils.py` or `notebooks/messy_exploration.ipynb` yet.**

---

## Goal

Get a feel for **prompting discipline**: how the way you phrase a request changes what you get back. You will generate brand-new code for two small tasks, starting from nothing.

---

## Context

You have just been handed the project. Before touching the existing (buggy) code, you want to understand the data and produce a couple of new artefacts from scratch. You will do this with an AI assistant as your pair programmer.

The dataset is `data/patients.csv` (2000 rows, binary target `readmission_30d`). The columns are:

`patient_id, age, sex, bmi, num_prior_admissions, length_of_stay, diagnosis_code, hospital_id, lab_sodium, lab_creatinine, admission_date, readmission_30d`

---

## Tasks

### Task 1.1 — An EDA module

Create **`src/eda.py`** (a new file, don't touch anything else). It must contain **at least** the following functions, all with type hints and docstrings:

- `load_data(path: str) -> pd.DataFrame` — loads the CSV, **correctly parses** `admission_date` even though the file mixes `YYYY-MM-DD` and `DD/MM/YYYY`. No silent `NaT`.
- `missing_value_report(df: pd.DataFrame) -> pd.DataFrame` — returns a per-column report with count and percentage of missing values, sorted descending.
- `numeric_summary(df: pd.DataFrame) -> pd.DataFrame` — min/max/mean/median/std for each numeric column.
- `target_rate_by(df: pd.DataFrame, col: str) -> pd.Series` — mean of `readmission_30d` grouped by a categorical column.
- `plot_numeric_distributions(df: pd.DataFrame, cols: list[str]) -> None` — a grid of histograms, one per column, with title and axis labels.
- `plot_target_rate_by(df: pd.DataFrame, col: str) -> None` — bar chart of target rate per category.

### Task 1.2 — A preprocessing module

Create **`src/preprocess.py`** with:

- A `clean_bmi(df)` function that flags/removes medically implausible values (you decide the threshold; document your choice).
- A `clean_creatinine(df)` function that handles negative values (they are data errors).
- A `build_preprocessor(numeric_cols, categorical_cols)` function that returns a **scikit-learn `ColumnTransformer`**, ready to be plugged into a `Pipeline`, combining:
  - Median imputation + standard scaling for numerics
  - Most-frequent imputation + one-hot encoding (`handle_unknown='ignore'`) for categoricals

Nothing else. Don't do train/test split, don't train a model here.

### Task 1.3 — A small demo notebook

Create `notebooks/eda_demo.ipynb` that calls a few functions from the two modules above, shows one plot of missing values, and one plot of target rate by `hospital_id`.

---

## How to prompt — try all of these

You are **expected** to experiment. Try the same task with different prompting styles and compare outputs.

1. **Zero-shot**: write a one-line request and accept whatever comes back. (e.g., "write an EDA module")
2. **Specified contract**: paste the function signatures you want and ask the AI to implement them.
3. **Role + constraints**: "You are a senior data scientist. Follow PEP 8, add type hints, no external libs beyond pandas/sklearn/matplotlib."
4. **Iterative refinement**: start small, then ask for one improvement at a time.
5. **Attach context**: use `@data/patients.csv` or `@src/ml_utils.py` in Cursor chat to give the model real data/API to match.

Keep the results of style 1 and style 2 side by side. You will compare in the debrief.

---

## Acceptance criteria

- [ ] `src/eda.py` exists, imports cleanly, and every function can be called on `data/patients.csv` without error.
- [ ] `load_data` correctly parses **both** date formats — no `NaT` in `admission_date` for valid rows. Verify with `df['admission_date'].isna().sum()`.
- [ ] `build_preprocessor` returns a `ColumnTransformer` that passes `.fit_transform(X_train)` / `.transform(X_test)` without shape mismatches and without leakage (scaler fit only on train).
- [ ] Every public function has a docstring and type hints.
- [ ] You can explain, in one sentence per function, **why** the AI wrote it that way. If you cannot, ask follow-up prompts until you can.

---

## Deliverables

- `src/eda.py`
- `src/preprocess.py`
- (optional) `notebooks/eda_demo.ipynb`
- A short `tp/TP1_notes.md` where you list:
  - Which prompting style gave the best result, and why you think so
  - One thing the AI got wrong the first time
  - One thing the AI did **better** than you expected

---

## What NOT to do in this TP

- Do not touch `src/ml_utils.py`, `notebooks/messy_exploration.ipynb`, or `scripts/train_baseline.py`. That is TP2 and TP3.
- Do not write tests. That is TP4.
- Do not ask the AI to "build the whole project". This TP is about **generating from scratch in a focused way**, not full automation.

---

Next: [`TP2_debug.md`](./TP2_debug.md)
