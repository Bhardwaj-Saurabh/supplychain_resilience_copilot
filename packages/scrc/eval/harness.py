"""Evaluation harness and the release gate (ADR-0007).

Runs each case multiple times to measure Supervisor routing **reproducibility**
(identical inputs -> identical tier, target ≥95%, PRD §11.1) and **accuracy**
against the disruption-replay golden tiers. ``passes_gate`` is the block/allow
decision wired into CI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from scrc.contracts import EscalationTier, SCRCModel, SupervisorDecision
from scrc.eval.datasets import EvalCase
from scrc.eval.metrics import accuracy_score, reproducibility_score

CaseDecisionFn = Callable[[EvalCase], SupervisorDecision]


class CaseResult(SCRCModel):
    name: str
    expected: EscalationTier
    observed: EscalationTier
    reproducible: bool

    @property
    def matched(self) -> bool:
        return self.observed is self.expected


class EvaluationReport(SCRCModel):
    reproducibility: float
    accuracy: float
    cases: list[CaseResult]


class EvalThresholds(SCRCModel):
    min_reproducibility: float = 0.95
    min_accuracy: float = 1.0  # disruption replay: no regression against the baseline


def run_routing_eval(
    decision_fn: CaseDecisionFn, cases: Sequence[EvalCase], repeats: int = 5
) -> EvaluationReport:
    results: list[CaseResult] = []
    for case in cases:
        tiers = [decision_fn(case).tier for _ in range(repeats)]
        results.append(
            CaseResult(
                name=case.name,
                expected=case.expected_tier,
                observed=tiers[0],
                reproducible=len(set(tiers)) == 1,
            )
        )
    return EvaluationReport(
        reproducibility=reproducibility_score([r.reproducible for r in results]),
        accuracy=accuracy_score([(r.observed, r.expected) for r in results]),
        cases=results,
    )


def passes_gate(report: EvaluationReport, thresholds: EvalThresholds | None = None) -> bool:
    t = thresholds or EvalThresholds()
    return report.reproducibility >= t.min_reproducibility and report.accuracy >= t.min_accuracy
