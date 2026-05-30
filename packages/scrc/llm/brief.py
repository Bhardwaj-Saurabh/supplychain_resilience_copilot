"""SHAP-to-brief translation (capability layer).

Turns the typed decision artefacts into a planner-readable brief. The factual
block is composed deterministically from the model outputs (no invented
numbers); the LLM only *narrates* it. When no LLM is available the factual block
is itself returned as the brief — the "LLM down -> raw SHAP" resilience path
(architecture.md §17/§19).
"""

from __future__ import annotations

from scrc.contracts import StockoutRiskResult, SupervisorDecision
from scrc.llm.ports import LLMClient

BRIEF_SYSTEM_PROMPT = (
    "You translate supply-chain ML model outputs into a concise, factual brief "
    "for a planner. Restate and explain ONLY the provided numbers and signals. "
    "Never invent figures and never make your own prediction — the models have "
    "already predicted. Keep it under 150 words."
)


def compose_brief_facts(decision: SupervisorDecision, stockout: StockoutRiskResult) -> str:
    """Deterministic factual block: the ground truth the brief must rest on."""
    lines = [
        f"SKU {decision.sku_id} @ store {decision.store_id}",
        f"Escalation tier: {decision.tier.value.upper()} (autonomous={decision.autonomous})",
        f"Calibrated stockout probability: {stockout.stockout_probability:.0%} "
        f"(confidence: {stockout.confidence_tier.value})",
    ]
    if stockout.shap_values:
        drivers = ", ".join(
            f"{s.feature} ({s.contribution:+.3f})" for s in stockout.shap_values[:5]
        )
        lines.append(f"Top risk drivers (SHAP): {drivers}")
    if decision.recommended_actions:
        actions = "; ".join(
            f"{a.rank}. {a.action_type.value} ({'reversible' if a.reversible else 'irreversible'})"
            for a in decision.recommended_actions
        )
        lines.append(f"Recommended actions: {actions}")
    return "\n".join(lines)


class BriefWriter:
    """Narrates the factual block via an LLM (P1: narration only)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def write(self, decision: SupervisorDecision, stockout: StockoutRiskResult) -> str:
        facts = compose_brief_facts(decision, stockout)
        return self._llm.complete(BRIEF_SYSTEM_PROMPT, facts)
