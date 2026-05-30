"""Drift detection for the Monitoring Agent (PRD §8.2).

Pure metric functions over recent predictions vs observed outcomes — rolling
MAPE for the forecaster and rolling F1 for the stockout classifier. Degradation
beyond threshold triggers retraining.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from scrc.contracts import SCRCModel


def rolling_mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error over non-zero actuals (0.0 if none)."""
    pairs = [(a, p) for a, p in zip(actual, predicted, strict=True) if a != 0]
    if not pairs:
        return 0.0
    return sum(abs((a - p) / a) for a, p in pairs) / len(pairs)


def rolling_f1(actual: Sequence[int], predicted: Sequence[int]) -> float:
    """Binary F1 of predicted stockout events vs observed (0.0 if undefined)."""
    tp = sum(1 for a, p in zip(actual, predicted, strict=True) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(actual, predicted, strict=True) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actual, predicted, strict=True) if a == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


class DriftThresholds(SCRCModel):
    max_mape: float = Field(default=0.20, ge=0.0)
    min_f1: float = Field(default=0.70, ge=0.0, le=1.0)


class DriftReport(SCRCModel):
    mape: float
    f1: float
    mape_degraded: bool
    f1_degraded: bool

    @property
    def degraded(self) -> bool:
        return self.mape_degraded or self.f1_degraded


def assess_drift(
    demand_actual: Sequence[float],
    demand_predicted: Sequence[float],
    event_actual: Sequence[int],
    event_predicted: Sequence[int],
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    t = thresholds or DriftThresholds()
    mape = rolling_mape(demand_actual, demand_predicted)
    f1 = rolling_f1(event_actual, event_predicted)
    return DriftReport(
        mape=mape,
        f1=f1,
        mape_degraded=mape > t.max_mape,
        f1_degraded=f1 < t.min_f1,
    )
