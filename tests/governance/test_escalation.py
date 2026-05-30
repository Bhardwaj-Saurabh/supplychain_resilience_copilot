from __future__ import annotations

from scrc.contracts import EscalationTier
from scrc.governance import EscalationSignals, evaluate_escalation


def _signals(**over: object) -> EscalationSignals:
    base: dict[str, object] = {"stockout_probability": 0.1, "uncertainty_ratio": 0.1}
    base.update(over)
    return EscalationSignals(**base)  # type: ignore[arg-type]


def test_routine_low_prob_narrow_no_anomaly() -> None:
    outcome = evaluate_escalation(_signals(stockout_probability=0.1, uncertainty_ratio=0.1))
    assert outcome.tier is EscalationTier.ROUTINE
    assert outcome.autonomous is True


def test_monitor_band_probability() -> None:
    assert evaluate_escalation(_signals(stockout_probability=0.4)).tier is EscalationTier.MONITOR


def test_monitor_single_anomaly_even_at_low_prob() -> None:
    outcome = evaluate_escalation(_signals(stockout_probability=0.1, anomaly_flag=True))
    assert outcome.tier is EscalationTier.MONITOR
    assert outcome.autonomous is True


def test_review_band_probability() -> None:
    outcome = evaluate_escalation(_signals(stockout_probability=0.6))
    assert outcome.tier is EscalationTier.REVIEW
    assert outcome.autonomous is False


def test_review_multi_signal_conjunction_at_low_prob() -> None:
    outcome = evaluate_escalation(_signals(stockout_probability=0.2, signal_count=2))
    assert outcome.tier is EscalationTier.REVIEW


def test_critical_high_probability() -> None:
    outcome = evaluate_escalation(_signals(stockout_probability=0.9))
    assert outcome.tier is EscalationTier.CRITICAL
    assert outcome.autonomous is False


def test_critical_triple_conjunction_at_low_prob() -> None:
    outcome = evaluate_escalation(
        _signals(
            stockout_probability=0.2,
            uncertainty_ratio=0.6,  # high uncertainty
            anomaly_flag=True,
            macro_shock=True,
            signal_count=3,
        )
    )
    assert outcome.tier is EscalationTier.CRITICAL


def test_missing_signal_forces_critical() -> None:
    outcome = evaluate_escalation(_signals(stockout_probability=0.05, missing_signal=True))
    assert outcome.tier is EscalationTier.CRITICAL
    assert outcome.autonomous is False
    assert "maximum uncertainty" in outcome.reasons[0]
