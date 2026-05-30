"""Deterministic escalation policy (governance, cross-cutting).

This is the core control flow (PRD §6.3) and it is **pure deterministic code,
never the LLM** (ADR-0001) — which is what makes routing reproducible (≥95% on
identical inputs) and audit-defensible. The Supervisor computes
:class:`EscalationSignals` from the agent outputs and calls
:func:`evaluate_escalation`; the LLM never assigns a tier.

Governance depends on ``scrc.contracts`` only (enforced in .importlinter).
"""

from __future__ import annotations

from pydantic import Field

from scrc.contracts import EscalationTier, SCRCModel

# Tunable thresholds. In production these come from the per-class autonomy
# policy in PostgreSQL (PRD §7.4); the version is stamped into provenance.
ROUTINE_MAX_PROB = 0.30
REVIEW_MIN_PROB = 0.55
CRITICAL_MIN_PROB = 0.75
NARROW_INTERVAL_RATIO = 0.20
HIGH_UNCERTAINTY_RATIO = 0.50


class EscalationSignals(SCRCModel):
    """The conjoint signals the tier is a pure function of."""

    stockout_probability: float = Field(ge=0.0, le=1.0)
    uncertainty_ratio: float = Field(ge=0.0)  # forecast interval width / P50
    anomaly_flag: bool = False
    macro_shock: bool = False
    #: Count of elevated independent signals (for multi-signal conjunction).
    signal_count: int = Field(default=0, ge=0)
    #: Set when a required upstream agent output was missing/timed out.
    missing_signal: bool = False

    @property
    def narrow_interval(self) -> bool:
        return self.uncertainty_ratio < NARROW_INTERVAL_RATIO

    @property
    def high_uncertainty(self) -> bool:
        return self.uncertainty_ratio > HIGH_UNCERTAINTY_RATIO


class EscalationOutcome(SCRCModel):
    """The tiering result. ``autonomous`` is true only for ROUTINE/MONITOR."""

    tier: EscalationTier
    autonomous: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_escalation(signals: EscalationSignals) -> EscalationOutcome:
    """Assign an escalation tier from conjoint signals (PRD §6.3).

    Precedence is CRITICAL → REVIEW → MONITOR → ROUTINE, with a conservative
    MONITOR default. A missing/timed-out upstream signal is treated as maximum
    uncertainty and forced to CRITICAL — never hallucinated past (PRD §7.4).
    """
    p = signals.stockout_probability
    reasons: list[str] = []

    if signals.missing_signal:
        return EscalationOutcome(
            tier=EscalationTier.CRITICAL,
            autonomous=False,
            reasons=["missing/timed-out agent output → treated as maximum uncertainty"],
        )

    if p > CRITICAL_MIN_PROB:
        reasons.append(f"stockout probability {p:.2f} > {CRITICAL_MIN_PROB}")
        tier = EscalationTier.CRITICAL
    elif signals.anomaly_flag and signals.macro_shock and signals.high_uncertainty:
        reasons.append("anomaly + macro shock + high demand uncertainty")
        tier = EscalationTier.CRITICAL
    elif REVIEW_MIN_PROB <= p <= CRITICAL_MIN_PROB:
        reasons.append(f"stockout probability {p:.2f} in review band")
        tier = EscalationTier.REVIEW
    elif signals.signal_count >= 2:
        reasons.append(f"multi-signal conjunction (signal_count={signals.signal_count})")
        tier = EscalationTier.REVIEW
    elif ROUTINE_MAX_PROB <= p < REVIEW_MIN_PROB:
        reasons.append(f"stockout probability {p:.2f} in monitor band")
        tier = EscalationTier.MONITOR
    elif signals.anomaly_flag:
        reasons.append("single anomaly")
        tier = EscalationTier.MONITOR
    elif p < ROUTINE_MAX_PROB and signals.narrow_interval and not signals.anomaly_flag:
        reasons.append("low probability, narrow interval, no anomaly")
        tier = EscalationTier.ROUTINE
    else:
        reasons.append("conservative default")
        tier = EscalationTier.MONITOR

    autonomous = tier in (EscalationTier.ROUTINE, EscalationTier.MONITOR)
    return EscalationOutcome(tier=tier, autonomous=autonomous, reasons=reasons)
