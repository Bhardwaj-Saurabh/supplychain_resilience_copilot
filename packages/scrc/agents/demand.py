"""Demand Forecasting Agent (Layer 4).

Framework-agnostic: depends on the tool layer and contracts, not on LangGraph
(P5). The orchestration adapter wraps ``run`` as a graph node.
"""

from __future__ import annotations

from collections.abc import Sequence

from scrc.contracts import DecisionRequest, QuantileForecastResult
from scrc.tools import ForecastTool


class DemandAgent:
    def __init__(self, tool: ForecastTool) -> None:
        self._tool = tool

    def run(
        self,
        request: DecisionRequest,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult:
        return self._tool.chronos_forecast(
            request.sku_id, request.store_id, request.horizon_days, covariates
        )
