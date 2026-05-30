"""LangGraph StateGraph wiring (orchestration adapter, Layer 4).

This is the *only* module that imports LangGraph. The agents it wires are
framework-agnostic (P5), so the MAF port (Module 6) reuses them and swaps this
adapter. Topology (architecture.md §5):

    START ─▶ demand ─┐                          autonomous ─▶ END
            logistics ┼─▶ stockout ─▶ supervisor ─┤
            macro ────┘                          escalate ─▶ review ─(interrupt)─▶ END

The three specialists fan out in parallel; ``stockout`` joins (runs once after
all three) because it needs the joint feature vector. Each specialist node is
wrapped so a failure records an error and leaves its output ``None`` — the
Supervisor then escalates to CRITICAL rather than acting on a partial picture.

ROUTINE/MONITOR decisions are autonomous and end the run; REVIEW/CRITICAL route
to the review node, which ``interrupt()``s to a human and resumes from the
checkpoint with their outcome (PRD §6.3). Requires a checkpointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    StockoutAgent,
    SupervisorAgent,
)
from scrc.orchestration.state import GraphState


@dataclass(frozen=True)
class AgentBundle:
    """The five agents, constructed (with real tools) at a composition root."""

    demand: DemandAgent
    logistics: LogisticsAgent
    macro: MacroAgent
    stockout: StockoutAgent
    supervisor: SupervisorAgent


def build_graph(bundle: AgentBundle, checkpointer: Any = None) -> Any:
    """Compile the decision StateGraph for the given agent bundle.

    Pass a ``PostgresSaver`` in production; defaults to an in-process
    ``MemorySaver`` (sufficient for tests and single-process runs).
    """

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

    def review_node(state: GraphState) -> GraphState:
        # Reached only for non-autonomous (REVIEW/CRITICAL) decisions. interrupt()
        # durably pauses the graph and surfaces the ReviewRequest to the planner;
        # the graph resumes from this exact checkpoint with the human outcome
        # (PRD §6.2, architecture.md §7). HITL is enforced in code, not a prompt.
        decision = state["decision"]
        review = bundle.supervisor.build_review_request(decision, state.get("stockout"))
        human_outcome: dict[str, object] = interrupt(review.model_dump(mode="json"))
        return {"review": review, "human_outcome": human_outcome}

    def route_after_supervisor(state: GraphState) -> str:
        return "end" if state["decision"].autonomous else "review"

    graph = StateGraph(GraphState)
    graph.add_node("demand", demand_node)
    graph.add_node("logistics", logistics_node)
    graph.add_node("macro", macro_node)
    graph.add_node("stockout", stockout_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("review", review_node)

    for specialist in ("demand", "logistics", "macro"):
        graph.add_edge(START, specialist)
        graph.add_edge(specialist, "stockout")
    graph.add_edge("stockout", "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_after_supervisor, {"review": "review", "end": END}
    )
    graph.add_edge("review", END)

    # A checkpointer is required for interrupt()/resume; MemorySaver suffices
    # in-process, PostgresSaver is wired for production (ADR-0006).
    return graph.compile(checkpointer=checkpointer or MemorySaver())
