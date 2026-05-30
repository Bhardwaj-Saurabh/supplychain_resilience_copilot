"""Monitoring Agent: drift detection + drift-triggered, gated retraining (§8.2).

Applies the same HITL-gate pattern to the ML lifecycle that the Supervisor
applies to operations. Optuna is optional (grid fallback); MLflow registration
is performed by the caller.
"""

from __future__ import annotations

from scrc.monitoring.agent import (
    DriftInputs,
    MonitoringAgent,
    MonitoringOutcome,
    RetrainData,
)
from scrc.monitoring.drift import (
    DriftReport,
    DriftThresholds,
    assess_drift,
    rolling_f1,
    rolling_mape,
)
from scrc.monitoring.retraining import (
    PromotionMode,
    decide_promotion,
    evaluate_candidate,
    optimise_stockout_params,
    train_candidate,
)

__all__ = [
    "DriftInputs",
    "DriftReport",
    "DriftThresholds",
    "MonitoringAgent",
    "MonitoringOutcome",
    "PromotionMode",
    "RetrainData",
    "assess_drift",
    "decide_promotion",
    "evaluate_candidate",
    "optimise_stockout_params",
    "rolling_f1",
    "rolling_mape",
    "train_candidate",
]
