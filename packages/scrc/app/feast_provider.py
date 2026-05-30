"""Feast-backed FeatureProvider for the production profile (ADR-0006).

Implements the ``scrc.tools.ports.FeatureProvider`` protocol against the Feast
**online** store (low-latency reads at inference time). Point-in-time features
come from ``get_online_features``; the demand context the forecaster needs is a
materialised rolling array feature (``recent_unit_sales``).

The ``FeatureStore`` is injected for testability; ``from_repo`` lazily imports
Feast so the rest of the system loads without it. Feature refs mirror
``pipelines/feast/features.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_DEMAND_BASE = [
    "demand_features:sell_price",
    "demand_features:lag_7",
    "demand_features:lag_28",
    "demand_features:rolling_mean_28",
]
_LOGISTICS = [
    "logistics_features:congestion_index",
    "logistics_features:dwell_hours",
    "logistics_features:rolling_zscore",
]


class FeastFeatureProvider:
    def __init__(self, store: Any) -> None:
        self._store = store

    @classmethod
    def from_repo(cls, repo_path: str) -> FeastFeatureProvider:
        from feast import FeatureStore

        return cls(FeatureStore(repo_path=repo_path))

    def _online(self, refs: Sequence[str], entity_row: dict[str, Any]) -> dict[str, Any]:
        names = [ref.split(":")[-1] for ref in refs]
        resp = self._store.get_online_features(
            features=list(refs), entity_rows=[entity_row]
        ).to_dict()
        return {name: resp[name][0] for name in names if name in resp}

    def demand_context(self, sku_id: str, store_id: str, lookback: int) -> list[float]:
        row = self._online(
            ["demand_features:recent_unit_sales"], {"sku_id": sku_id, "store_id": store_id}
        )
        series = row.get("recent_unit_sales") or []
        return [float(x) for x in series][-lookback:]

    def stockout_features(self, sku_id: str, store_id: str) -> dict[str, float]:
        row = self._online(_DEMAND_BASE, {"sku_id": sku_id, "store_id": store_id})
        return {k: float(v) for k, v in row.items() if v is not None}

    def logistics_features(self, port_id: str) -> dict[str, float]:
        row = self._online(_LOGISTICS, {"port_id": port_id})
        return {k: float(v) for k, v in row.items() if v is not None}

    def macro_latest(self, series_ids: Sequence[str]) -> dict[str, float]:
        refs = [f"macro_features:{s}" for s in series_ids]
        row = self._online(refs, {"market_id": "global"})
        return {k: float(v) for k, v in row.items() if v is not None}
