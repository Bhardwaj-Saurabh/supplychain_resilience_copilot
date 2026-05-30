"""Supervisor decision and HITL contracts.

The escalation tier and the ``autonomous`` flag are recorded here, but the
*logic* that assigns them lives in the governance layer, not in this contract —
tiering is deterministic code, never the LLM (ADR-0001).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field

from scrc.contracts.common import SCHEMA_VERSION, Probability, SCRCModel
from scrc.contracts.provenance import DecisionProvenance


class EscalationTier(StrEnum):
    """Four-tier escalation model (PRD §6.3)."""

    ROUTINE = "routine"
    MONITOR = "monitor"
    REVIEW = "review"
    CRITICAL = "critical"


class ActionType(StrEnum):
    """Ranked action classes the Supervisor may recommend (PRD §6.2)."""

    EXPEDITE = "expedite"
    REROUTE = "reroute"
    SAFETY_STOCK_TRANSFER = "safety_stock_transfer"
    SUBSTITUTE = "substitute"


class ActionRecommendation(SCRCModel):
    """One ranked, rationale-bearing action. ``reversible`` gates rollback
    registration before autonomous execution (PRD §7.4)."""

    action_type: ActionType
    rank: int = Field(ge=1)
    rationale: str = Field(min_length=1)
    reversible: bool


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SupervisorDecision(SCRCModel):
    """The synthesised, tiered, audited decision artefact."""

    schema_version: str = SCHEMA_VERSION
    decision_id: str = Field(min_length=1)
    sku_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    tier: EscalationTier
    stockout_probability: Probability
    autonomous: bool
    recommended_actions: list[ActionRecommendation] = Field(default_factory=list)
    provenance: DecisionProvenance
    created_at: datetime = Field(default_factory=_utcnow)


class ReviewRequest(SCRCModel):
    """Structured HITL brief emitted via ``interrupt()`` (PRD §6.2; architecture.md §7).

    Not a bare approval prompt — it carries the full explanation the planner
    needs to act in seconds.
    """

    schema_version: str = SCHEMA_VERSION
    decision_id: str = Field(min_length=1)
    tier: EscalationTier
    brief: str = Field(min_length=1)
    recommended_actions: list[ActionRecommendation] = Field(default_factory=list)
    counterfactual: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
