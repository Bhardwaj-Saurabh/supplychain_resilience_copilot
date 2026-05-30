"""Concrete feature providers.

``StaticFeatureProvider`` is an in-memory implementation of the
:class:`~scrc.tools.ports.FeatureProvider` port — handy for local runs, tests,
and replay scenarios. The Feast-backed provider (online store reads) is added
when the store is wired; it will implement the same protocol, so nothing above
this layer changes.
"""

from __future__ import annotations

from collections.abc import Sequence


class StaticFeatureProvider:
    """Dict-backed feature provider satisfying the ``FeatureProvider`` protocol."""

    def __init__(
        self,
        demand_contexts: dict[tuple[str, str], list[float]] | None = None,
        stockout: dict[tuple[str, str], dict[str, float]] | None = None,
        logistics: dict[str, dict[str, float]] | None = None,
        macro: dict[str, float] | None = None,
    ) -> None:
        self._demand = demand_contexts or {}
        self._stockout = stockout or {}
        self._logistics = logistics or {}
        self._macro = macro or {}

    def demand_context(self, sku_id: str, store_id: str, lookback: int) -> list[float]:
        return self._demand.get((sku_id, store_id), [])[-lookback:]

    def stockout_features(self, sku_id: str, store_id: str) -> dict[str, float]:
        return dict(self._stockout.get((sku_id, store_id), {}))

    def logistics_features(self, port_id: str) -> dict[str, float]:
        return dict(self._logistics.get(port_id, {}))

    def macro_latest(self, series_ids: Sequence[str]) -> dict[str, float]:
        return {s: self._macro[s] for s in series_ids if s in self._macro}
