from __future__ import annotations

from scrc.contracts import EscalationTier
from scrc.eval import (
    EvalThresholds,
    accuracy_score,
    disruption_cases,
    passes_gate,
    reproducibility_score,
    run_routing_eval,
    scenario_decision_fn,
)
from scrc.eval.datasets import DisruptionScenario


def test_reproducibility_and_accuracy_scores() -> None:
    assert reproducibility_score([True, True, True]) == 1.0
    assert reproducibility_score([True, False]) == 0.5
    assert reproducibility_score([]) == 1.0
    assert accuracy_score([(EscalationTier.REVIEW, EscalationTier.REVIEW)]) == 1.0
    assert accuracy_score([(EscalationTier.ROUTINE, EscalationTier.CRITICAL)]) == 0.0


def test_disruption_replay_hits_expected_tiers() -> None:
    report = run_routing_eval(scenario_decision_fn, disruption_cases())
    # The governed pipeline reproduces each historical disruption's tier exactly.
    assert report.accuracy == 1.0
    assert report.reproducibility == 1.0
    assert passes_gate(report)
    by_name = {c.name: c for c in report.cases}
    assert by_name["COVID-19 demand shock"].observed is EscalationTier.CRITICAL
    assert by_name["Suez Canal 2021 blockage"].observed is EscalationTier.REVIEW
    assert by_name["Baseline calm market"].observed is EscalationTier.ROUTINE


def test_scenario_tiers_individually() -> None:
    from scrc.eval.datasets import EvalCase

    def tier(scenario: DisruptionScenario) -> EscalationTier:
        case = next(c for c in disruption_cases() if c.scenario is scenario)
        return scenario_decision_fn(case).tier

    assert tier(DisruptionScenario.RED_SEA_2024) is EscalationTier.CRITICAL
    assert tier(DisruptionScenario.BASELINE) is EscalationTier.ROUTINE
    _ = EvalCase  # imported type is part of the public surface


def test_gate_fails_below_threshold() -> None:
    report = run_routing_eval(scenario_decision_fn, disruption_cases())
    strict = EvalThresholds(min_reproducibility=0.95, min_accuracy=1.0)
    assert passes_gate(report, strict)
    impossible = EvalThresholds(min_reproducibility=1.01, min_accuracy=1.0)
    assert not passes_gate(report, impossible)


def test_routing_is_reproducible_under_repeats() -> None:
    # The ≥95% reproducibility target (PRD §11.1): deterministic tiering -> 100%.
    report = run_routing_eval(scenario_decision_fn, disruption_cases(), repeats=10)
    assert report.reproducibility == 1.0
