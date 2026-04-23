# TP3 — Refactor: turn the messy notebook into a clean package

**Duration:** ~2h
**Tools:** **half the room uses Cursor, the other half uses Claude Code**. At the end, you compare.
**Files you touch:** `notebooks/messy_exploration.ipynb`, `src/` (new modules), possibly a new `notebooks/` entry.

---

## Goal

Take `notebooks/messy_exploration.ipynb` — a realistically bad data-science notebook — and extract a **clean, testable, reusable** pipeline into `src/`. Keep the notebook only for what a notebook is good at: exploration and storytelling.

By the end, the same model-training flow that lives today in 30 chaotic cells should live in **a few small, well-named functions** in `src/`, plus a thin notebook that calls them.

---

## Starting point

Open `notebooks/messy_exploration.ipynb`. Count the sins: duplicated preprocessing, magic numbers, scaler fit before split, scattered imports, `tmp` / `df_final_final` variables, EDA interleaved with training, dead commented-out code, a broken CV loop, a model trained but never saved.

The ML task underneath is actually simple: load → preprocess → split → train LogReg and RandomForest → evaluate.

---

## Target architecture

Design your own, but it should look roughly like this:

```
src/
├── eda.py              # (from TP1) exploration utilities
├── preprocess.py       # (from TP1) cleaning + ColumnTransformer builder
├── data.py             # load_and_clean, group-aware splitting
├── models.py           # model factory / registry (LogReg, RF)
├── training.py         # fit_and_evaluate, cross_validate
├── evaluation.py       # metrics, plots
└── io.py               # save_model, load_model  (because a model that is
                        # never saved is a model that was never trained)
```

Exact names and boundaries are your call — defend them in the debrief.

The cleaned notebook should be **10-15 cells maximum**, roughly:

1. Imports + config (paths, random state)
2. Load + EDA summary (calling functions from `src/eda.py`)
3. A couple of exploratory plots
4. Define preprocessor (from `src/preprocess.py`)
5. Train + evaluate LogReg (one call to a function from `src/training.py`)
6. Train + evaluate RF
7. Cross-validation summary
8. Save best model (`src/io.py`)
9. Short markdown: what you conclude

---

## Constraints

- **No business logic in the notebook.** If a cell contains a loop, a split, a `fit`, it probably belongs in `src/`.
- **No magic numbers.** Thresholds, seeds, fold counts, hyperparameters go into named constants or function arguments.
- **One preprocessing path.** Not three variants with slight differences.
- **Deterministic.** `random_state` set consistently everywhere it matters.
- **Keep the bugs fixed from TP2.** The refactor must preserve all of your TP2 fixes.
- **No silent data loss.** `drop_duplicates` on the target, silent `NaT` from bad date parsing, etc., should not reappear.
- **Split before preprocess.** Any fitting (imputer, scaler, encoder) happens only on training data, via `Pipeline` or `ColumnTransformer` used inside `fit`.

---

## Suggested two-lane approach

The trainer will split the room.

### Lane A — Cursor IDE

- Work **cell by cell**: read a cell, decide where its logic belongs, move it, test that the notebook still runs.
- Use inline edit (`Ctrl+K` / `Cmd+K`) for small local refactors.
- Use chat (`Ctrl+L` / `Cmd+L`) with file context (`@src/ml_utils.py`) for larger moves.
- Commit after each substantial extraction.

### Lane B — Claude Code (agent)

- Give the agent a clear goal document (you can reuse this TP file).
- Let it propose a plan before writing code. **Read the plan.** Ask for changes before approving.
- Let it make a batch of changes, then **review the full diff** before accepting.
- Iterate: "the `training.py` module is too big, split model definition from training loop".

### Both lanes

Keep a running file `tp/TP3_log.md`: what you asked, what was proposed, what you accepted, what you rejected, why.

---

## Acceptance criteria

- [ ] The refactored notebook executes end-to-end:
  ```bash
  jupyter nbconvert --to notebook --execute notebooks/messy_exploration.ipynb --output /tmp/out.ipynb
  ```
- [ ] The notebook is **≤ 15 cells** and contains **no** `fit` / `split` / `for`-loop logic beyond a one-liner call.
- [ ] `src/` has at least **3 new modules** with a clear responsibility each.
- [ ] `scripts/train_baseline.py` still runs **and** now uses the refactored modules (update the imports as needed).
- [ ] A model is **saved to disk** (e.g., `models/readmission_logreg.joblib`) and loadable.
- [ ] No `tmp`, `tmp2`, `df_final`, `df_final_final` remain anywhere.
- [ ] `git log` is readable: small commits, meaningful messages.

---

## Deliverables

- Reorganised `src/` tree.
- Cleaned `notebooks/messy_exploration.ipynb` (or replace it with a new `notebooks/exploration.ipynb` — your choice).
- Updated `scripts/train_baseline.py`.
- `tp/TP3_log.md` with your prompting log and a short comparison between Cursor and Claude Code (if your pair tried both).

---

## Debrief questions (for the group discussion)

1. Which lane felt faster for the extraction phase?
2. Which lane gave better commit hygiene?
3. Where did the AI "cheat" — e.g., delete code it didn't understand?
4. What did you have to specify explicitly that you expected it to infer?
5. If you had to do a similar refactor tomorrow on another repo, what would you do differently in your first prompt?

---

Next: [`TP4_tests.md`](./TP4_tests.md)
