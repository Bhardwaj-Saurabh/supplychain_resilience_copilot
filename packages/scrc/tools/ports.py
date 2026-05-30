"""Ports for the tool / capability layer (Layer 3).

These ``Protocol``s are the seams of the hexagonal design (ADR-0003). Tools
orchestrate a *model* port and a *feature* port into a typed contract, but they
import neither ``scrc.ml`` nor ``scrc.data`` — the concrete models and feature
store satisfy these protocols **structurally** and are wired at a composition
root. This keeps the tool layer (and the agents above it) decoupled from the
serving implementation. The decoupling is enforced in `.importlinter`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from scrc.contracts import (
    AnomalyResult,
    QuantileForecastResult,
    StockoutRiskResult,
)

# --- Model ports (satisfied by scrc.ml.*) ----------------------------------


class Forecaster(Protocol):
    def forecast(
        self,
        sku_id: str,
        store_id: str,
        context: Sequence[float],
        horizon_days: int,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult: ...


class StockoutModel(Protocol):
    def predict(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult: ...


class AnomalyModel(Protocol):
    def predict(
        self, port_ids: Sequence[str], features: Mapping[str, float], congestion_score: float
    ) -> AnomalyResult: ...


# --- Feature port (satisfied by a Feast-backed provider, later) ------------


class FeatureProvider(Protocol):
    """Low-latency feature reads for inference (Feast online store in prod)."""

    def demand_context(self, sku_id: str, store_id: str, lookback: int) -> list[float]: ...
    def stockout_features(self, sku_id: str, store_id: str) -> dict[str, float]: ...
    def logistics_features(self, port_id: str) -> dict[str, float]: ...
    def macro_latest(self, series_ids: Sequence[str]) -> dict[str, float]: ...
