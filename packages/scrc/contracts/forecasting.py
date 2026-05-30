"""Demand Forecasting Agent output contract (Chronos-2)."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator

from scrc.contracts.common import SCHEMA_VERSION, NonNegFloat, SCRCModel


class QuantileForecastResult(SCRCModel):
    """Chronos-2 quantile forecast for one SKU-store-horizon.

    ``interval_width`` (P90 - P10) is the confidence signal that flows into the
    escalation logic — never discard it for a point estimate (P2;
    architecture.md §8). It is derived automatically when omitted.
    """

    schema_version: str = SCHEMA_VERSION
    sku_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    horizon_days: int = Field(gt=0)
    p10: float
    p50: float
    p90: float
    interval_width: NonNegFloat
    covariate_flags_used: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _default_interval_width(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and "interval_width" not in data
            and "p10" in data
            and "p90" in data
        ):
            return {**data, "interval_width": data["p90"] - data["p10"]}
        return data

    @model_validator(mode="after")
    def _check_quantile_order(self) -> Self:
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError("quantiles must satisfy p10 <= p50 <= p90")
        return self
