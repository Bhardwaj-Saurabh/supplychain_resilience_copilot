"""Evaluation harness (Opik-backed): routing reproducibility + disruption replay.

The release gate (ADR-0007): identical-input routing reproducibility ≥95% and no
disruption-replay regression. Runs the real agents end-to-end via scenario
bundles; Opik logging is optional (degrades to a no-op).
"""

from __future__ import annotations

from scrc.eval.datasets import DisruptionScenario, EvalCase, disruption_cases
from scrc.eval.harness import (
    CaseResult,
    EvalThresholds,
    EvaluationReport,
    passes_gate,
    run_routing_eval,
)
from scrc.eval.metrics import accuracy_score, reproducibility_score
from scrc.eval.opik_adapter import log_evaluation
from scrc.eval.scenarios import build_scenario_bundle, scenario_decision_fn

__all__ = [
    "CaseResult",
    "DisruptionScenario",
    "EvalCase",
    "EvalThresholds",
    "EvaluationReport",
    "accuracy_score",
    "build_scenario_bundle",
    "disruption_cases",
    "log_evaluation",
    "passes_gate",
    "reproducibility_score",
    "run_routing_eval",
    "scenario_decision_fn",
]
