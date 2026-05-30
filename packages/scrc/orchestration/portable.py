"""Framework-agnostic pipeline runner.

Runs the decision pipeline with no orchestrator at all — plain Python over the
same agents the LangGraph graph and the MAF port wire up. This is the concrete
proof of P5/ADR-0003: orchestration frameworks add durability, HITL, and
observability *around* this logic; they do not own it.

It deliberately omits the cross-cutting concerns (checkpointing, interrupt-based
HITL, audit gate) that the LangGraph adapter provides — use it for tests,
batch/offline replay, and as the reference the MAF port is compared against.
"""

from __future__ import annotations

from scrc.contracts import DecisionRequest, SupervisorDecision
from scrc.orchestration.bundle import AgentBundle


def run_pipeline(bundle: AgentBundle, request: DecisionRequest) -> SupervisorDecision:
    """Demand/Logistics/Macro -> Stockout -> Supervisor, returning the decision.

    A failure in any specialist is treated as a missing signal and escalated to
    CRITICAL (PRD §7.4), mirroring the graph's conservative join.
    """
    try:
        forecast = bundle.demand.run(request)
        anomaly = bundle.logistics.run(request)
        macro = bundle.macro.run()
        stockout = bundle.stockout.run(request, forecast, anomaly, macro)
    except Exception as exc:
        return bundle.supervisor.escalate_missing(request, f"specialist failure: {exc}")
    return bundle.supervisor.synthesise(request, forecast, anomaly, macro, stockout)
