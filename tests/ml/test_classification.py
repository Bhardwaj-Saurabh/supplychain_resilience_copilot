from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scrc.contracts import ConfidenceTier, StockoutRiskResult
from scrc.ml.classification import StockoutClassifier

FEATURES = ["lead_time", "demand", "safety_stock"]


def _dataset(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    lead_time = rng.normal(10, 3, n)
    demand = rng.normal(50, 10, n)
    safety = rng.normal(20, 5, n)
    logit = 0.4 * (lead_time - 10) - 0.3 * (safety - 20) + rng.normal(0, 1, n)
    y = (logit > 0).astype(int)
    X = pd.DataFrame({"lead_time": lead_time, "demand": demand, "safety_stock": safety})
    return X, pd.Series(y)


def _trained() -> StockoutClassifier:
    X, y = _dataset()
    clf = StockoutClassifier(FEATURES)
    clf.train(X, y)
    return clf


def test_predict_returns_calibrated_result_with_shap() -> None:
    result = _trained().predict(
        "A", "CA_1", {"lead_time": 18.0, "demand": 55.0, "safety_stock": 10.0}
    )
    assert isinstance(result, StockoutRiskResult)
    assert 0.0 <= result.stockout_probability <= 1.0
    assert result.calibrated is True
    assert result.confidence_tier in set(ConfidenceTier)
    assert 1 <= len(result.shap_values) <= len(FEATURES)


def test_predict_is_deterministic() -> None:
    clf = _trained()
    feats = {"lead_time": 18.0, "demand": 55.0, "safety_stock": 10.0}
    assert clf.predict("A", "CA_1", feats).stockout_probability == pytest.approx(
        clf.predict("A", "CA_1", feats).stockout_probability
    )


def test_higher_risk_inputs_score_higher() -> None:
    clf = _trained()
    high = clf.predict(
        "A", "CA_1", {"lead_time": 22.0, "demand": 60.0, "safety_stock": 5.0}
    ).stockout_probability
    low = clf.predict(
        "A", "CA_1", {"lead_time": 4.0, "demand": 40.0, "safety_stock": 35.0}
    ).stockout_probability
    assert high > low


def test_missing_feature_raises() -> None:
    clf = _trained()
    with pytest.raises(KeyError):
        clf.predict("A", "CA_1", {"lead_time": 10.0, "demand": 50.0})


def test_predict_before_train_raises() -> None:
    with pytest.raises(RuntimeError):
        StockoutClassifier(FEATURES).predict("A", "CA_1", dict.fromkeys(FEATURES, 0.0))
