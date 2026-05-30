"""LangGraph StateGraph wiring (orchestration adapter, Layer 4).

This is the *only* module that imports LangGraph. The agents it wires are
framework-agnostic (P5), so the MAF port (Module 6) reuses them and swaps this
adapter. Topology (architecture.md §5):

    START ─▶ demand ─┐
            logistics ┼─▶ stockout ─▶ supervisor ─▶ END
            macro ────┘

The three specialists fan out in parallel; ``stockout`` joins (runs once after
all three) because it needs the joint feature vector. Each specialist node is
wrapped so a failure records an error and leaves its output ``None`` — the
Supervisor then escalates to CRITICAL rather than acting on a partial picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

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


def build_graph(bundle: AgentBundle) -> Any:
    """Compile the decision StateGraph for the given agent bundle."""

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

    graph = StateGraph(GraphState)
    graph.add_node("demand", demand_node)
    graph.add_node("logistics", logistics_node)
    graph.add_node("macro", macro_node)
    graph.add_node("stockout", stockout_node)
    graph.add_node("supervisor", supervisor_node)

    for specialist in ("demand", "logistics", "macro"):
        graph.add_edge(START, specialist)
        graph.add_edge(specialist, "stockout")
    graph.add_edge("stockout", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile()
