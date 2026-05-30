"""LangGraph state schema for the decision graph (orchestration adapter).

The graph state carries the request and each agent's typed output. ``errors``
uses an additive reducer because the three specialist nodes run in parallel and
may each append — every other key is written by exactly one node.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from scrc.contracts import (
    AnomalyResult,
    DecisionRequest,
    MacroSignals,
    QuantileForecastResult,
    ReviewRequest,
    StockoutRiskResult,
    SupervisorDecision,
)


class GraphState(TypedDict, total=False):
    request: DecisionRequest
    forecast: QuantileForecastResult | None
    anomaly: AnomalyResult | None
    macro: MacroSignals | None
    stockout: StockoutRiskResult | None
    decision: SupervisorDecision
    audit_id: str | None
    rollback_entry_ids: list[str]
    review: ReviewRequest
    human_outcome: dict[str, object]
    errors: Annotated[list[str], operator.add]
