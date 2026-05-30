"""Run the routing evaluation as a release gate (ADR-0007).

    uv run python -m scrc.eval

Exits non-zero if reproducibility or disruption-replay accuracy regress below
threshold — wire this into CI to block releases.
"""

from __future__ import annotations

from scrc.eval.datasets import disruption_cases
from scrc.eval.harness import EvalThresholds, passes_gate, run_routing_eval
from scrc.eval.opik_adapter import log_evaluation
from scrc.eval.scenarios import scenario_decision_fn


def main() -> int:
    report = run_routing_eval(scenario_decision_fn, disruption_cases())
    thresholds = EvalThresholds()
    ok = passes_gate(report, thresholds)

    print("Supply Chain Resilience Co-Pilot — routing evaluation")
    print(
        f"  reproducibility: {report.reproducibility:.2%} "
        f"(min {thresholds.min_reproducibility:.0%})"
    )
    print(f"  accuracy:        {report.accuracy:.2%} (min {thresholds.min_accuracy:.0%})")
    for case in report.cases:
        mark = "ok" if case.matched else "MISS"
        repro = "stable" if case.reproducible else "FLAKY"
        print(
            f"    [{mark:>4}] {case.name}: {case.observed.value} "
            f"(expected {case.expected.value}, {repro})"
        )
    logged = log_evaluation(report)
    print(f"  opik: {'logged' if logged else 'unavailable (no-op)'}")
    print(f"  GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
