"""Optional Opik logging for evaluation runs (ADR-0005).

Opik is the tracing **and** evaluation home, but the harness must run without it
(e.g. in CI without a Comet/Opik backend). ``log_evaluation`` lazily imports
Opik and degrades to a no-op (returning ``False``) when it's unavailable or
misconfigured, so the gate never depends on the observability stack.

Install to enable:  uv pip install opik
"""

from __future__ import annotations

from scrc.eval.harness import EvaluationReport


def log_evaluation(report: EvaluationReport, project_name: str = "scrc") -> bool:
    """Log the evaluation report to Opik. Returns True if logged, False if Opik
    is unavailable (no-op)."""
    try:
        import opik

        client = opik.Opik(project_name=project_name)
        trace = client.trace(
            name="routing-evaluation",
            input={"cases": [c.name for c in report.cases]},
            output={
                "reproducibility": report.reproducibility,
                "accuracy": report.accuracy,
            },
        )
        for case in report.cases:
            trace.span(
                name=case.name,
                input={"expected": case.expected.value},
                output={"observed": case.observed.value, "reproducible": case.reproducible},
            )
        trace.end()
        return True
    except Exception:
        return False
