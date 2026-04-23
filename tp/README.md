# AI-Assisted Coding — Hands-on TPs

Welcome. You will work on a **realistic but deliberately imperfect** ML codebase (`ai-coding-tp/`) using AI coding assistants (**Cursor** and **Claude Code**).

The goal is **not** to produce a perfect solution in record time. The goal is to learn how to:

- Prompt effectively
- Spot what the AI gets wrong
- Stay in the driver's seat on technical decisions
- Use AI to move fast **without losing quality**

---

## The project

`ai-coding-tp/` contains a small patient-readmission ML project:

```
ai-coding-tp/
├── README.md
├── requirements.txt
├── data/patients.csv              # ~2000 rows, binary classification target
├── notebooks/messy_exploration.ipynb
├── src/ml_utils.py
├── scripts/train_baseline.py
└── tests/                         # empty — you will fill this in TP4
```

The dataset is synthetic. The code was "written by three different people under deadline pressure": it runs, but it has **real-world problems** — methodological bugs, leakage traps, messy structure, bad practices. **Treat it like production code you just inherited.**

---

## TP schedule

| #   | Topic                | Main skill                         
| --- | -------------------- | ---------------------------------- 
| TP1 | Prompting            | Generating new code from scratch   
| TP2 | Debug & optimize     | Finding and fixing bugs with AI    
| TP3 | Refactor             | Turning a notebook into a package  
| TP4 | Testing              | Writing a regression-proof pytest suite 
| TP5 | Documentation        | Docstrings, README, diagrams       

Each TP has its own instructions file (`TP1_prompting.md`, `TP2_debug.md`, ...).

---

## Ground rules

1. **Read the instructions of each TP before starting.** Know the goal and the acceptance criteria.
2. **Don't blindly apply AI suggestions.** Review every diff before accepting it. Be ready to explain, in your own words, *why* each change is correct.
3. **Keep a short prompt log.** For every significant prompt, note in a scratch file: what you asked, what you got, whether you accepted it. This is how you learn what works.
4. **If the AI makes something worse, revert.** `git` is your friend. Commit often, in small steps.
5. **Prefer small, targeted prompts** over "fix everything in this file". You will get better results and keep control.
6. **No googling for solutions to the bugs.** Use the AI assistant, that's the whole point.

---

## Setup

```bash
cd ai-coding-tp
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If the project is not yet a git repo, initialise one **before** TP2:

```bash
git init
git add .
git commit -m "initial state"
```

You will want to be able to diff and revert throughout the exercises.

---

## A note on the two assistants

- **Cursor** is an IDE. You mainly use Chat (`Ctrl+L` / `Cmd+L`) and inline edits (`Ctrl+K` / `Cmd+K`). You review diffs one by one.
- **Claude Code** is a terminal-based agent. You describe a task, it plans and edits files autonomously. You review the batch of changes at the end.

Both can do most things. Part of the training is noticing where each one shines.

---

When you are ready, open [`TP1_prompting.md`](./TP1_prompting.md).
