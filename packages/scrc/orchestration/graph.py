"""LangGraph StateGraph wiring (orchestration adapter, Layer 4).

This is the *only* module that imports LangGraph. The agents it wires are
framework-agnostic (P5), so the MAF port (Module 6) reuses them and swaps this
adapter. Topology (architecture.md §5):

    START ─▶ demand ─┐                                   autonomous ─▶ END
            logistics ┼─▶ stockout ─▶ supervisor ─▶ audit ─┤
            macro ────┘                                   escalate ─▶ review ─(interrupt)─▶ END

The three specialists fan out in parallel; ``stockout`` joins (runs once after
all three) because it needs the joint feature vector. Each specialist node is
wrapped so a failure records an error and leaves its output ``None`` — the
Supervisor then escalates to CRITICAL rather than acting on a partial picture.

The ``audit`` node logs every decision and enforces no-audit-no-autonomy
(ADR-0002): an unauditable autonomous decision is downgraded to HITL; otherwise
reversible actions are registered for rollback before execution. Routing then
sends ROUTINE/MONITOR to END and REVIEW/CRITICAL to the review node, which
``interrupt()``s to a human and resumes from the checkpoint (PRD §6.3).
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from scrc.governance import (
    AuditLog,
    InMemoryAuditLog,
    InMemoryRollbackRegistry,
    RollbackRegistry,
    enforce_audit,
    register_reversible_actions,
)
from scrc.observability import decision_span, record_audit_downgrade, record_decision
from scrc.orchestration.bundle import AgentBundle
from scrc.orchestration.state import GraphState


def build_graph(
    bundle: AgentBundle,
    checkpointer: Any = None,
    audit: AuditLog | None = None,
    rollback: RollbackRegistry | None = None,
) -> Any:
    """Compile the decision StateGraph for the given agent bundle.

    Pass a ``PostgresSaver`` checkpointer, an ``MlflowAuditLog``, and a durable
    rollback registry in production; all default to in-process implementations
    (sufficient for tests and single-process runs).
    """
    audit_log = audit or InMemoryAuditLog()
    rollback_registry = rollback or InMemoryRollbackRegistry()

    def demand_node(state: GraphState) -> GraphState:
        try:
            return {"forecast": bundle.demand.run(state["request"])}
        except Exception as exc:
            return {"forecast": None, "errors": [f"demand: {exc}"]}

    def logistics_node(state: GraphState) -> GraphState:
        try:
            return {"anomaly": bundle.logistics.run(state["request"])}
        except Exception as exc:
            return {"anomaly": None, "errors": [f"logistics: {exc}"]}

    def macro_node(state: GraphState) -> GraphState:
        try:
            return {"macro": bundle.macro.run()}
        except Exception as exc:
            return {"macro": None, "errors": [f"macro: {exc}"]}

    def stockout_node(state: GraphState) -> GraphState:
        forecast = state.get("forecast")
        anomaly = state.get("anomaly")
        macro = state.get("macro")
        if forecast is None or anomaly is None or macro is None:
            return {"stockout": None}
        try:
            return {"stockout": bundle.stockout.run(state["request"], forecast, anomaly, macro)}
        except Exception as exc:
            return {"stockout": None, "errors": [f"stockout: {exc}"]}

    def supervisor_node(state: GraphState) -> GraphState:
        request = state["request"]
        forecast = state.get("forecast")
        anomaly = state.get("anomaly")
        macro = state.get("macro")
        stockout = state.get("stockout")
        if forecast is None or anomaly is None or macro is None or stockout is None:
            reason = "; ".join(state.get("errors", [])) or "missing upstream signal"
            return {"decision": bundle.supervisor.escalate_missing(request, reason)}
        return {
            "decision": bundle.supervisor.synthesise(request, forecast, anomaly, macro, stockout)
        }

    def audit_node(state: GraphState) -> GraphState:
        # Logs every decision and enforces no-audit-no-autonomy (ADR-0002): if an
        # autonomous decision cannot be audited, it is downgraded to HITL. For
        # decisions that remain autonomous, reversible actions are registered in
        # the rollback registry BEFORE execution (PRD §7.4).
        decision = state["decision"]
        with decision_span(
            "decision",
            tier=decision.tier.value,
            autonomous=decision.autonomous,
            input_hash=decision.provenance.input_hash,
        ):
            outcome = enforce_audit(decision, audit_log)
        record_decision(outcome.decision.tier.value, outcome.decision.autonomous)

        updates: GraphState = {"decision": outcome.decision, "audit_id": outcome.audit_id}
        if outcome.downgraded:
            record_audit_downgrade()
            updates["errors"] = ["audit unavailable: autonomy downgraded to review (ADR-0002)"]
        if outcome.decision.autonomous:
            entries = register_reversible_actions(outcome.decision, rollback_registry)
            updates["rollback_entry_ids"] = [e.entry_id for e in entries]
        return updates

    def review_node(state: GraphState) -> GraphState:
        # Reached only for non-autonomous (REVIEW/CRITICAL) decisions. interrupt()
        # durably pauses the graph and surfaces the ReviewRequest to the planner;
        # the graph resumes from this exact checkpoint with the human outcome
        # (PRD §6.2, architecture.md §7). HITL is enforced in code, not a prompt.
        decision = state["decision"]
        review = bundle.supervisor.build_review_request(decision, state.get("stockout"))
        human_outcome: dict[str, object] = interrupt(review.model_dump(mode="json"))
        return {"review": review, "human_outcome": human_outcome}

    def route_after_audit(state: GraphState) -> str:
        # Routes on the possibly-downgraded decision (no-audit-no-autonomy).
        return "end" if state["decision"].autonomous else "review"

    graph = StateGraph(GraphState)
    graph.add_node("demand", demand_node)
    graph.add_node("logistics", logistics_node)
    graph.add_node("macro", macro_node)
    graph.add_node("stockout", stockout_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("audit", audit_node)
    graph.add_node("review", review_node)

    for specialist in ("demand", "logistics", "macro"):
        graph.add_edge(START, specialist)
        graph.add_edge(specialist, "stockout")
    graph.add_edge("stockout", "supervisor")
    graph.add_edge("supervisor", "audit")
    graph.add_conditional_edges("audit", route_after_audit, {"review": "review", "end": END})
    graph.add_edge("review", END)

    # A checkpointer is required for interrupt()/resume; MemorySaver suffices
    # in-process, PostgresSaver is wired for production (ADR-0006).
    return graph.compile(checkpointer=checkpointer or MemorySaver())
