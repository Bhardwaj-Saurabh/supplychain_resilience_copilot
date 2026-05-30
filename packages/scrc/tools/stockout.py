"""Stockout-risk classification tool (Layer 3).

The joint feature vector is assembled by the stockout *agent* from the other
three agents' outputs (PRD §6.2); this tool is the typed boundary over the
calibrated model. ``classify_stockout_risk`` returns the contract unchanged —
turning SHAP into a planner brief is the agent/LLM's job (ADR-0001).
"""

from __future__ import annotations

from collections.abc import Mapping

from scrc.contracts import StockoutRiskResult
from scrc.tools.ports import StockoutModel


class StockoutTool:
    def __init__(self, model: StockoutModel):
        self._model = model

    def classify_stockout_risk(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult:
        return self._model.predict(sku_id, store_id, features)
