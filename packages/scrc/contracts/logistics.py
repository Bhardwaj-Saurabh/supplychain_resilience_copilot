"""Logistics Risk Agent output contracts (Isolation Forest + KernelSHAP)."""

from __future__ import annotations

from pydantic import Field

from scrc.contracts.common import SCHEMA_VERSION, Probability, SCRCModel


class ShapFeature(SCRCModel):
    """A single per-feature anomaly attribution (KernelSHAP)."""

    feature: str = Field(min_length=1)
    shap_value: float


class CongestionMetrics(SCRCModel):
    """Output of ``get_port_congestion`` — current port congestion snapshot."""

    schema_version: str = SCHEMA_VERSION
    port_ids: list[str]
    congestion_score: Probability


class AnomalyResult(SCRCModel):
    """Output of ``detect_freight_anomaly`` — anomaly score with attribution.

    ``top_features`` carries the SHAP attribution through to the Supervisor (P2).
    """

    schema_version: str = SCHEMA_VERSION
    port_ids: list[str]
    congestion_score: Probability
    anomaly_flag: bool
    anomaly_score: float
    top_features: list[ShapFeature] = Field(default_factory=list)
