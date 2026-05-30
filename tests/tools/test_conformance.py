"""Proves the concrete scrc.ml models satisfy the tool ports (structural typing)
and flow through the tools to typed contracts — the ML-as-Tool boundary working
end to end without any explicit interface inheritance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scrc.contracts import AnomalyResult, StockoutRiskResult
from scrc.ml.anomaly import FreightAnomalyDetector
from scrc.ml.classification import StockoutClassifier
from scrc.tools import LogisticsTool, StaticFeatureProvider, StockoutTool


def test_real_stockout_classifier_drives_the_tool() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"lead_time": rng.normal(10, 3, n), "safety_stock": rng.normal(20, 5, n)})
    y = (X["lead_time"] > X["lead_time"].median()).astype(int)
    clf = StockoutClassifier(["lead_time", "safety_stock"])
    clf.train(X, y)

    tool = StockoutTool(clf)  # real model satisfies StockoutModel protocol
    result = tool.classify_stockout_risk("A", "CA_1", {"lead_time": 18.0, "safety_stock": 10.0})
    assert isinstance(result, StockoutRiskResult)
    assert result.shap_values  # attribution flows through


def test_real_anomaly_detector_drives_the_tool() -> None:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {"congestion_index": rng.normal(0.3, 0.05, n), "dwell_hours": rng.normal(10, 1, n)}
    )
    det = FreightAnomalyDetector(
        ["congestion_index", "dwell_hours"], n_background=30, n_estimators=80
    )
    det.fit(X)

    provider = StaticFeatureProvider(
        logistics={"USLAX": {"congestion_index": 0.95, "dwell_hours": 40.0}}
    )
    tool = LogisticsTool(det, provider)  # real model satisfies AnomalyModel protocol
    result = tool.detect_freight_anomaly("USLAX")
    assert isinstance(result, AnomalyResult)
    assert result.port_ids == ["USLAX"]
