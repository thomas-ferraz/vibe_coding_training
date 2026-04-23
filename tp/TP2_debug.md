# TP2 — Debug & optimize: fix the codebase with AI

**Files you touch:** `src/ml_utils.py`, `scripts/train_baseline.py`

---

## Goal

Learn to **use AI as a debugging partner without surrendering judgment**. You will:

- Make `scripts/train_baseline.py` runnable end-to-end from the CLI.
- Find and fix the bugs in `src/ml_utils.py`.
- Be able to **justify every fix** — not because the AI suggested it, but because you understand why it is a bug.

The existing code is realistic: it runs or half-runs, produces numbers, and has problems that are **methodological as much as syntactic**. A model that gives 0.95 accuracy is not necessarily a working model.

---

## Warm-up — get the script to run (15 min)

From the repo root:

```bash
cd ai-coding-tp
python scripts/train_baseline.py --data data/patients.csv --cv-folds 5
```

It will fail. There are (at least) **two different problems** preventing it from running end-to-end. One is about imports/project layout, the other is about argument types.

### Tasks

1. Fix both failures. The script must **complete successfully** and print metrics.
2. Do **not** rewrite the whole script — make the minimum changes needed.
3. Keep commits small: one commit per bug fixed.

### How to use AI

- Paste the **exact traceback** into Chat. Do not paraphrase.
- Ask: *"What is the root cause? Give me the smallest possible fix."*
- If the AI suggests rewriting 30 lines, push back: *"Give me a one-line change instead."*

---

## Main task — audit `src/ml_utils.py` (1h)

Read the module. It has **several** non-trivial bugs. Some are leakage, some are misleading docstrings, some are silent data corruption, some produce numerically correct but misleading results.

### Rules

- **Do not fix anything before you have a list.** First, identify bugs; then prioritise; then fix.
- Use AI to **help you think**, not to do the work for you.
- For every bug you fix, update the docstring so it matches the new behaviour.

### Suggested workflow

1. **Open each function one by one.** For each, ask the AI:
   > *Here is a function from an existing codebase. Review it critically. Tell me what could go wrong in production — edge cases, leakage, silent failures, incorrect assumptions. Do not fix anything yet. List issues in order of severity.*

2. **Build a bug list** in a scratch file (`tp/TP2_findings.md`). For each issue, note:
   - The function and line(s)
   - A one-line description
   - Severity (Critical / Major / Minor)
   - How you would test for it (this will feed TP4)

3. **Sanity-check the AI's findings.** Some "issues" the AI flags are stylistic, not real bugs. Some real bugs it will miss. Your job is to filter.

4. **Fix one bug at a time**, each in its own commit:
   ```
   git commit -m "fix(ml_utils): correct AUC computation in evaluate_model"
   ```

5. **Verify** after each fix:
   - `python scripts/train_baseline.py ...` still runs
   - The notebook still executes: `jupyter nbconvert --to notebook --execute notebooks/messy_exploration.ipynb --output /tmp/out.ipynb`
   - Metrics look **sensible** (if AUC drops from 0.92 to 0.78, that may be a sign you removed a leakage, not a regression)

---

## Areas to scrutinise carefully

Without spoiling specifics, pay extra attention to:

- Anything involving **train/test split** and **scaling/imputation** — where is the fit happening?
- **Date parsing** — does every row actually get parsed?
- **Grouping and splitting** — what is the unit of observation? What leaks between splits?
- **Categorical encoding** — what happens if train and test don't see the same categories?
- **Metrics** — what arguments does `roc_auc_score` expect?
- **Feature importance** — do all models expose `.feature_importances_`?
- **Cross-validation** — is the split appropriate for this data?

If `scripts/train_baseline.py` reports an AUC noticeably above ~0.85 on this dataset, you should be **suspicious**, not happy.

---

## Acceptance criteria

- [ ] `scripts/train_baseline.py --data data/patients.csv --cv-folds 5` runs to completion from repo root.
- [ ] `notebooks/messy_exploration.ipynb` **still executes end-to-end** (you may adapt notebook cells if they rely on buggy behaviour, but don't refactor the notebook — that's TP3).
- [ ] You have a file `tp/TP2_findings.md` listing the bugs you identified, with severity and a short rationale.
- [ ] At least **5 distinct bugs** fixed in `src/ml_utils.py`, each in its own commit with a clear message.
- [ ] For every fix, docstrings updated so they no longer lie.
- [ ] You can, without AI help, **explain out loud** why each fix is correct.

---

## Deliverables

- Modified `src/ml_utils.py` and `scripts/train_baseline.py`.
- `tp/TP2_findings.md` (your bug list).
- Clean git history, one fix per commit.

---

## Traps to avoid

- **"The AI said so."** That is not a justification. If you cannot explain a fix, revert it.
- **Fixing style instead of bugs.** Renaming a variable does not fix a leakage.
- **"Fix all bugs in this file."** Vague prompts produce vague, often wrong, large diffs. Go function by function.
- **Accepting a fix that changes metrics drastically without understanding why.** A sudden AUC drop from 0.95 → 0.72 is important information, not a regression to patch over.

---

Next: [`TP3_refactor.md`](./TP3_refactor.md)
