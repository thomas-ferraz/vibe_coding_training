# Patient Readmission Project

This project predicts 30-day readmission.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Explore the Data

Start Jupyter and open the EDA notebook:

```bash
jupyter notebook
```

Then open `notebooks/eda_demo.ipynb`.

## Run the Baseline Model

Run the baseline training script from the project root:

```bash
python -m scripts.train_baseline --data data/patients.csv --cv-folds 5
```

This loads `data/patients.csv`, trains the baseline readmission model, prints
test metrics, and runs 5-fold cross-validation.
