"""ML serving layer (Layer 2): trained models served behind typed outputs.

Each model produces a ``scrc.contracts`` artefact carrying uncertainty and SHAP
attribution (P2). Models are registered in MLflow; agent tools call only the
alias-resolved production version (:mod:`scrc.ml.registry`).
"""

from __future__ import annotations

from scrc.ml.anomaly import FreightAnomalyDetector
from scrc.ml.classification import StockoutClassifier
from scrc.ml.forecasting import ChronosForecaster, parse_chronos_response
from scrc.ml.registry import (
    load_production_model,
    log_and_register,
    promote_to_production,
)

__all__ = [
    "ChronosForecaster",
    "FreightAnomalyDetector",
    "StockoutClassifier",
    "load_production_model",
    "log_and_register",
    "parse_chronos_response",
    "promote_to_production",
]
