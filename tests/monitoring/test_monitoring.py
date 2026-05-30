from __future__ import annotations

import numpy as np
import pandas as pd

from scrc.monitoring import (
    DriftInputs,
    DriftThresholds,
    MonitoringAgent,
    PromotionMode,
    RetrainData,
    assess_drift,
    decide_promotion,
    rolling_f1,
    rolling_mape,
)

FEATURES = ["f0", "f1", "f2"]


# --- drift metrics ---------------------------------------------------------


def test_rolling_mape_skips_zero_actuals() -> None:
    assert rolling_mape([100.0, 0.0, 50.0], [110.0, 5.0, 50.0]) == 0.05  # (0.1 + 0.0)/2


def test_rolling_f1_basic() -> None:
    assert rolling_f1([1, 1, 0, 0], [1, 0, 0, 0]) == 2 / 3  # tp=1, fp=0, fn=1


def test_assess_drift_flags_each_metric() -> None:
    report = assess_drift(
        demand_actual=[100.0, 100.0],
        demand_predicted=[160.0, 160.0],  # 60% MAPE -> degraded
        event_actual=[1, 1, 0],
        event_predicted=[0, 0, 0],  # F1 = 0 -> degraded
        thresholds=DriftThresholds(),
    )
    assert report.mape_degraded is True
    assert report.f1_degraded is True
    assert report.degraded is True


def test_assess_drift_healthy() -> None:
    report = assess_drift([100.0], [101.0], [1, 0], [1, 0])
    assert report.degraded is False


# --- promotion gate --------------------------------------------------------


def test_decide_promotion_auto_vs_approval() -> None:
    assert decide_promotion(0.90, 0.85) is PromotionMode.AUTO_PROMOTE
    assert decide_promotion(0.855, 0.85) is PromotionMode.REQUIRES_APPROVAL  # within margin
    assert decide_promotion(0.80, 0.85) is PromotionMode.REQUIRES_APPROVAL


# --- agent end-to-end ------------------------------------------------------


def _dataset(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATURES})
    y = (X["f0"] + X["f1"] * 0.5 + rng.normal(0, 0.5, n) > 0).astype(int)
    return X, y


def _retrain_data() -> RetrainData:
    X, y = _dataset(seed=1)
    val_X, val_y = _dataset(seed=2)
    return RetrainData(X=X, y=y, val_X=val_X, val_y=val_y, feature_names=FEATURES)


def test_agent_skips_retraining_when_healthy() -> None:
    agent = MonitoringAgent()
    inputs = DriftInputs([100.0], [101.0], [1, 0], [1, 0])
    outcome = agent.run(inputs, _retrain_data(), production_auc=0.8)
    assert outcome.retrained is False
    assert outcome.candidate_auc is None


def test_agent_retrains_and_decides_promotion_on_drift() -> None:
    agent = MonitoringAgent(n_trials=3)  # small; uses grid fallback without optuna
    inputs = DriftInputs([100.0, 100.0], [200.0, 200.0], [1, 1, 0], [0, 0, 0])
    outcome = agent.run(inputs, _retrain_data(), production_auc=0.5)
    assert outcome.retrained is True
    assert outcome.candidate_auc is not None
    assert 0.0 <= outcome.candidate_auc <= 1.0
    assert outcome.promotion in set(PromotionMode)
    assert outcome.params  # chosen hyperparameters recorded for provenance
