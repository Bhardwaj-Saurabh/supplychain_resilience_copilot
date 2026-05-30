"""Stockout-Risk Classifier Agent (Layer 4).

Assembles the **joint feature vector** from the three upstream agent outputs
(demand, logistics, macro) plus base SKU features from the feature store, then
calls the stockout tool. This conjoint assembly — reasoning across all signals
at once — is the gap G2 the system exists to close (PRD §2.2).
"""

from __future__ import annotations

from scrc.contracts import (
    AnomalyResult,
    DecisionRequest,
    MacroSignals,
    QuantileForecastResult,
    RegimeLabel,
    StockoutRiskResult,
)
from scrc.tools import FeatureProvider, StockoutTool


def assemble_features(
    forecast: QuantileForecastResult,
    anomaly: AnomalyResult,
    macro: MacroSignals,
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build the joint feature vector from the upstream typed outputs.

    Uncertainty and anomaly attribution are carried in as features, never
    collapsed to a point estimate (P2).
    """
    derived = {
        "demand_p50": forecast.p50,
        "demand_interval_width": forecast.interval_width,
        "congestion_score": anomaly.congestion_score,
        "anomaly_score": anomaly.anomaly_score,
        "anomaly_flag": float(anomaly.anomaly_flag),
        "macro_shock": 1.0 if macro.regime_label is RegimeLabel.SHOCK else 0.0,
        "macro_confidence": macro.regime_confidence,
    }
    return {**(base or {}), **derived}


class StockoutAgent:
    def __init__(self, tool: StockoutTool, features: FeatureProvider | None = None) -> None:
        self._tool = tool
        self._features = features

    def run(
        self,
        request: DecisionRequest,
        forecast: QuantileForecastResult,
        anomaly: AnomalyResult,
        macro: MacroSignals,
    ) -> StockoutRiskResult:
        base = (
            self._features.stockout_features(request.sku_id, request.store_id)
            if self._features is not None
            else {}
        )
        features = assemble_features(forecast, anomaly, macro, base)
        return self._tool.classify_stockout_risk(request.sku_id, request.store_id, features)
