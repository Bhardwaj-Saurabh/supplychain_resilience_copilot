"""Load trained scrc.ml models from local artifacts (production profile).

The XGBoost and Isolation Forest wrappers are richer than plain sklearn
estimators (they own calibration, SHAP, feature names), so they are serialised
with joblib. A deployment materialises the MLflow-registry production versions to
``SCRC_MODEL_DIR`` and these loaders read them behind the tool ports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STOCKOUT_FILENAME = "stockout.joblib"
ANOMALY_FILENAME = "anomaly.joblib"


def save_model(model: Any, path: str | Path) -> None:
    import joblib

    joblib.dump(model, Path(path))


def _load(path: str | Path) -> Any:
    import joblib

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"model artifact not found: {target}")
    return joblib.load(target)


def load_stockout_model(model_dir: str | Path) -> Any:
    return _load(Path(model_dir) / STOCKOUT_FILENAME)


def load_anomaly_model(model_dir: str | Path) -> Any:
    return _load(Path(model_dir) / ANOMALY_FILENAME)
