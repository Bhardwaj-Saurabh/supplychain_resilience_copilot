"""Demand forecasting tool (Layer 3)."""

from __future__ import annotations

from collections.abc import Sequence

from scrc.contracts import QuantileForecastResult
from scrc.tools.ports import FeatureProvider, Forecaster


class ForecastTool:
    """Reads the demand context from the feature store, calls the forecaster,
    returns a typed ``QuantileForecastResult`` (the ``chronos_forecast`` tool)."""

    def __init__(self, forecaster: Forecaster, features: FeatureProvider, lookback: int = 90):
        self._forecaster = forecaster
        self._features = features
        self._lookback = lookback

    def chronos_forecast(
        self,
        sku_id: str,
        store_id: str,
        horizon_days: int,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult:
        context = self._features.demand_context(sku_id, store_id, self._lookback)
        return self._forecaster.forecast(sku_id, store_id, context, horizon_days, covariates)
