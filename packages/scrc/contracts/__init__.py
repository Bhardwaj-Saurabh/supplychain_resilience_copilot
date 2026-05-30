"""SCRC typed contracts — the stable interface between layers (ADR-0003).

Import from this package, not the submodules:

    from scrc.contracts import QuantileForecastResult, EscalationTier
"""

from __future__ import annotations

from scrc.contracts.common import SCHEMA_VERSION, NonNegFloat, Probability, SCRCModel
from scrc.contracts.decision import (
    ActionRecommendation,
    ActionType,
    EscalationTier,
    ReviewRequest,
    SupervisorDecision,
)
from scrc.contracts.forecasting import QuantileForecastResult
from scrc.contracts.hitl import HumanDecision
from scrc.contracts.logistics import AnomalyResult, CongestionMetrics, ShapFeature
from scrc.contracts.macro import MacroSignals, RegimeLabel
from scrc.contracts.provenance import DecisionProvenance
from scrc.contracts.request import DecisionRequest
from scrc.contracts.stockout import ConfidenceTier, ShapValue, StockoutRiskResult

__all__ = [
    "SCHEMA_VERSION",
    "ActionRecommendation",
    "ActionType",
    "AnomalyResult",
    "ConfidenceTier",
    "CongestionMetrics",
    "DecisionProvenance",
    "DecisionRequest",
    "EscalationTier",
    "HumanDecision",
    "MacroSignals",
    "NonNegFloat",
    "Probability",
    "QuantileForecastResult",
    "RegimeLabel",
    "ReviewRequest",
    "SCRCModel",
    "ShapFeature",
    "ShapValue",
    "StockoutRiskResult",
    "SupervisorDecision",
]
