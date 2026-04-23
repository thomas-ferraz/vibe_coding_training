# TP4 — Testing: write a regression-proof `pytest` suite

**Duration:** ~1h30
**Tools:** Cursor or Claude Code
**Files you touch:** `tests/` (create), possibly small tweaks in `src/` for testability

---

## Goal

Build a **pytest suite** that protects the codebase against regressions — in particular the bugs you fixed in TP2. If a teammate reintroduces any of those bugs, your tests must fail.

This is the "AI-written tests that actually mean something" TP. AI-generated tests from docstrings alone tend to lock in current behaviour, including bugs. You will learn how to avoid that.

---

## Context

After TP2 and TP3 you have:

- A clean `src/` with small, well-named functions
- A list in `tp/TP2_findings.md` of every bug you fixed

That list is your **test oracle**: each bug should become at least one test that would catch the bug if it came back.

---

## Tasks

### Task 4.1 — Test infrastructure

Set up `tests/` properly:

```
tests/
├── __init__.py
├── conftest.py          # fixtures: tiny synthetic df, trained model, etc.
├── test_data.py
├── test_preprocess.py
├── test_splitting.py
├── test_models.py
├── test_evaluation.py
└── test_training_pipeline.py   # higher-level integration test
```

In `conftest.py`, build **small, hand-crafted fixtures** — 20-50 rows, controlled values, known answers. Do **not** rely on `data/patients.csv` for unit tests (too slow, non-deterministic boundaries). Use the real CSV only in one integration test.

### Task 4.2 — Regression tests from the TP2 bug list

For **every** bug in `tp/TP2_findings.md`, write a test that would have caught the bug. Some examples (without giving away all of them):

- If a date-parsing bug caused silent `NaT` for some format, write a test with a dataframe containing both formats and assert every `admission_date` is a valid `Timestamp`.
- If an imputer used to fit on the full dataset, write a test asserting that the train-imputer does **not** see test statistics (e.g., train has all NaN in a column, test has known values, and after `.fit(train).transform(test)` the test values are still finite).
- If a group-split bug used to leak patients, write a test that creates a df with repeated `patient_id` values and asserts **no overlap** in the resulting split.
- If a metric was computed on labels instead of probabilities, write a test where the same model's accuracy is the same but AUC should differ between a correct and an incorrect implementation — and assert the correct one.
- If a function used to crash on `LogisticRegression`, add a test that calls it on both a tree model and a linear model.
- If CV used to use unshuffled folds on ordered data, write a test where a simple ordered pattern in y would collapse a no-shuffle CV score, and assert shuffled behaviour.

### Task 4.3 — Basic unit tests (positive path)

In addition to regressions, cover the happy path:

- Shapes and dtypes after preprocessing
- Pipeline fits and predicts without error on the tiny fixture
- `save_model` / `load_model` round-trip
- Reproducibility: same seed → same predictions

### Task 4.4 — One end-to-end smoke test

`test_training_pipeline.py` runs the full pipeline on `data/patients.csv` and asserts:

- The script or the main entry function returns a metrics dict with the expected keys
- AUC is in a reasonable range (e.g., `0.55 < auc < 0.90` — bounds chosen on purpose: too low means the model is broken, too high means leakage came back)
- The saved model file exists

Mark it `@pytest.mark.slow` if it takes more than a couple of seconds.

---

## How to use AI effectively here

### Do

- Ask the AI to **generate tests from the bug list**, one bug at a time, with the explicit intent *"write a test that would FAIL on the following buggy version of the function"*. Then verify: revert the fix temporarily, run the test — does it fail? Reapply the fix — does it pass? This is the only way to know your regression test actually works.
- Ask for test **fixtures** to be extracted into `conftest.py`.
- Ask for parametrised tests where relevant (`@pytest.mark.parametrize`).

### Don't

- Don't blindly accept "generate a full test suite for `src/ml_utils.py`". You will get a pile of tests that confirm whatever the code currently does — including any bugs that snuck back.
- Don't test **implementation details** (e.g., "assert we called `SimpleImputer`"). Test **behaviour**.
- Don't write tests against the real CSV that depend on specific row counts or means — they will silently break on regeneration.

---

## Acceptance criteria

- [ ] `pytest` runs from the repo root and all tests pass.
- [ ] There is **at least one test per bug** listed in `tp/TP2_findings.md`. Each such test is annotated with a comment referencing the bug (e.g., `# regression: ml_utils.evaluate_model used labels instead of probabilities for AUC`).
- [ ] You have **actually verified** that each regression test fails on the pre-fix version. Document how you checked (a short note in `tp/TP4_log.md` is enough).
- [ ] The smoke test runs end-to-end on `data/patients.csv` in under ~30 seconds.
- [ ] Coverage of `src/` is at least 70% (`pip install pytest-cov && pytest --cov=src`).

---

## Deliverables

- `tests/` directory populated.
- `tp/TP4_log.md` with your verification notes and a line per regression test explaining which bug it protects against.
- A short CI snippet (optional): a `Makefile` target or a `.github/workflows/tests.yml` that runs `pytest`.

---

## Traps to avoid

- **"AI-written tests that confirm current behaviour."** If you let the AI write tests by reading only the current (post-fix) code, you will test what the code does, not what it **should** do. Always anchor the test against the **bug description**, not the current implementation.
- **Non-deterministic tests.** Set seeds everywhere. No `datetime.now()` in fixtures.
- **Tests that depend on the real dataset.** Only the smoke test should.
- **Brittle assertions** (`assert acc == 0.8421`). Use ranges (`0.65 < acc < 0.95`) or structural assertions (keys present, shape correct).

---

Next: [`TP5_doc.md`](./TP5_doc.md)
