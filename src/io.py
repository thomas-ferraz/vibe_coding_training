"""Model persistence helpers."""

from pathlib import Path

import joblib


def save_model(model, path: str | Path) -> Path:
    """Persist a fitted model or pipeline to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path):
    """Load a saved model or pipeline from disk."""
    return joblib.load(Path(path))
