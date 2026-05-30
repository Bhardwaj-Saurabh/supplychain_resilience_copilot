from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scrc.contracts import AnomalyResult
from scrc.ml.anomaly import FreightAnomalyDetector

FEATURES = ["congestion_index", "dwell_hours", "rolling_zscore"]


def _normal(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "congestion_index": rng.normal(0.3, 0.05, n),
            "dwell_hours": rng.normal(10, 1, n),
            "rolling_zscore": rng.normal(0, 0.5, n),
        }
    )


def _fitted() -> FreightAnomalyDetector:
    det = FreightAnomalyDetector(FEATURES, n_background=30, n_estimators=100)
    det.fit(_normal())
    return det


def test_outlier_scores_higher_than_normal() -> None:
    det = _fitted()
    normal = det.predict(
        ["USLAX"],
        {"congestion_index": 0.3, "dwell_hours": 10.0, "rolling_zscore": 0.0},
        congestion_score=0.3,
        nsamples=50,
    )
    outlier = det.predict(
        ["USLAX"],
        {"congestion_index": 0.95, "dwell_hours": 40.0, "rolling_zscore": 6.0},
        congestion_score=0.95,
        nsamples=50,
    )
    assert isinstance(outlier, AnomalyResult)
    assert outlier.anomaly_score > normal.anomaly_score
    assert 1 <= len(outlier.top_features) <= 3


def test_congestion_score_is_clamped() -> None:
    det = _fitted()
    result = det.predict(
        ["USLAX"],
        {"congestion_index": 0.95, "dwell_hours": 40.0, "rolling_zscore": 6.0},
        congestion_score=1.5,
        nsamples=20,
    )
    assert result.congestion_score == 1.0


def test_predict_before_fit_raises() -> None:
    det = FreightAnomalyDetector(FEATURES)
    with pytest.raises(RuntimeError):
        det.predict(["USLAX"], dict.fromkeys(FEATURES, 0.0), congestion_score=0.1)
