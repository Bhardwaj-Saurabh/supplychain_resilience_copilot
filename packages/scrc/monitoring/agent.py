"""Monitoring Agent — drift watch + gated retraining (PRD §8.2).

Runs on a schedule: assesses drift; on degradation it retrains a candidate
stockout classifier and decides whether it may be auto-promoted or needs human
approval. Registration to the MLflow registry (and the HITL approval for
``REQUIRES_APPROVAL``) happens at the caller, reusing the same governance gate
the operational flow uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scrc.contracts import SCRCModel
from scrc.monitoring.drift import DriftReport, DriftThresholds, assess_drift
from scrc.monitoring.retraining import (
    PromotionMode,
    decide_promotion,
    evaluate_candidate,
    optimise_stockout_params,
    train_candidate,
)


@dataclass(frozen=True)
class DriftInputs:
    demand_actual: list[float]
    demand_predicted: list[float]
    event_actual: list[int]
    event_predicted: list[int]


@dataclass(frozen=True)
class RetrainData:
    X: pd.DataFrame
    y: pd.Series
    val_X: pd.DataFrame
    val_y: pd.Series
    feature_names: list[str]


class MonitoringOutcome(SCRCModel):
    drift: DriftReport
    retrained: bool
    candidate_auc: float | None = None
    promotion: PromotionMode | None = None
    params: dict[str, object] | None = None


class MonitoringAgent:
    def __init__(self, thresholds: DriftThresholds | None = None, n_trials: int = 15) -> None:
        self._thresholds = thresholds or DriftThresholds()
        self._n_trials = n_trials

    def run(
        self,
        drift_inputs: DriftInputs,
        retrain: RetrainData,
        production_auc: float,
        seed: int = 0,
    ) -> MonitoringOutcome:
        report = assess_drift(
            drift_inputs.demand_actual,
            drift_inputs.demand_predicted,
            drift_inputs.event_actual,
            drift_inputs.event_predicted,
            self._thresholds,
        )
        if not report.degraded:
            return MonitoringOutcome(drift=report, retrained=False)

        params = optimise_stockout_params(
            retrain.X, retrain.y, retrain.feature_names, n_trials=self._n_trials, seed=seed
        )
        candidate = train_candidate(retrain.feature_names, params, retrain.X, retrain.y, seed=seed)
        candidate_auc = evaluate_candidate(candidate, retrain.val_X, retrain.val_y)
        return MonitoringOutcome(
            drift=report,
            retrained=True,
            candidate_auc=candidate_auc,
            promotion=decide_promotion(candidate_auc, production_auc),
            params=params,
        )
