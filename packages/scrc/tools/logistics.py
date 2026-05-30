"""Logistics risk tools (Layer 3)."""

from __future__ import annotations

from collections.abc import Sequence

from scrc.contracts import AnomalyResult, CongestionMetrics
from scrc.tools.ports import AnomalyModel, FeatureProvider


class LogisticsTool:
    """The ``get_port_congestion`` and ``detect_freight_anomaly`` tools."""

    def __init__(self, model: AnomalyModel, features: FeatureProvider):
        self._model = model
        self._features = features

    def get_port_congestion(self, port_ids: Sequence[str]) -> CongestionMetrics:
        scores = [
            self._features.logistics_features(p).get("congestion_index", 0.0) for p in port_ids
        ]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return CongestionMetrics(port_ids=list(port_ids), congestion_score=mean_score)

    def detect_freight_anomaly(self, port_id: str) -> AnomalyResult:
        features = self._features.logistics_features(port_id)
        congestion = features.get("congestion_index", 0.0)
        return self._model.predict([port_id], features, congestion_score=congestion)
