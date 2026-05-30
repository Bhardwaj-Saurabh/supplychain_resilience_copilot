"""Microsoft Agent Framework 1.0 port of the decision workflow (Module 6).

This is the framework-portability teaching deliverable. It wires the **same**
framework-agnostic agents as the LangGraph graph (P5, ADR-0003) — only the
orchestration primitives change. Mapping:

    LangGraph (scrc.orchestration.graph)   MAF 1.0 (this module)
    ------------------------------------   ---------------------
    StateGraph node (def)                  Executor + @handler (async)
    add_edge / START fan-out               add_fan_out_edges
    join via multiple incoming edges       add_fan_in_edges (list delivered)
    shared TypedDict GraphState            ctx shared state + typed messages
    interrupt() / Command(resume=...)      ctx.request_info + @response_handler
    MemorySaver / PostgresSaver            DurableWorkflows (Azure-backed)

``agent_framework`` is imported lazily (install the ``maf`` extra), so this
module imports cleanly without the SDK and the rest of the system never depends
on it. The framework-agnostic reference is ``scrc.orchestration.run_pipeline``.
"""

from __future__ import annotations

from typing import Any

from scrc.contracts import DecisionRequest
from scrc.orchestration.bundle import AgentBundle


def build_maf_workflow(bundle: AgentBundle) -> Any:
    """Build the MAF ``Workflow`` equivalent of the LangGraph decision graph.

    Topology mirrors the graph: ingest fans out to demand/logistics/macro,
    which fan in to stockout (joint feature vector), then supervisor yields the
    decision. The audit gate and HITL are added with ``request_info`` in the
    same way the LangGraph version uses ``interrupt()`` — left as the documented
    extension point so the core port stays legible.
    """
    from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

    class IngestExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="ingest")

        @handler
        async def ingest(
            self, request: DecisionRequest, ctx: WorkflowContext[DecisionRequest]
        ) -> None:
            ctx.set_state("request", request)  # shared state ≈ GraphState["request"]
            await ctx.send_message(request)

    class DemandExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="demand")

        @handler
        async def run(self, request: DecisionRequest, ctx: WorkflowContext[Any]) -> None:
            await ctx.send_message(("forecast", bundle.demand.run(request)))

    class LogisticsExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="logistics")

        @handler
        async def run(self, request: DecisionRequest, ctx: WorkflowContext[Any]) -> None:
            await ctx.send_message(("anomaly", bundle.logistics.run(request)))

    class MacroExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="macro")

        @handler
        async def run(self, request: DecisionRequest, ctx: WorkflowContext[Any]) -> None:
            await ctx.send_message(("macro", bundle.macro.run()))

    class StockoutExecutor(Executor):
        """Fan-in join: receives the three specialist outputs as a list."""

        def __init__(self) -> None:
            super().__init__(id="stockout")

        @handler
        async def run(self, signals: list[tuple[str, Any]], ctx: WorkflowContext[Any]) -> None:
            data = dict(signals)
            request: DecisionRequest = ctx.get_state("request")
            stockout = bundle.stockout.run(
                request, data["forecast"], data["anomaly"], data["macro"]
            )
            await ctx.send_message({**data, "stockout": stockout})

    class SupervisorExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="supervisor")

        @handler
        async def run(self, data: dict[str, Any], ctx: WorkflowContext[Any, Any]) -> None:
            request: DecisionRequest = ctx.get_state("request")
            decision = bundle.supervisor.synthesise(
                request, data["forecast"], data["anomaly"], data["macro"], data["stockout"]
            )
            await ctx.yield_output(decision)

    ingest = IngestExecutor()
    demand = DemandExecutor()
    logistics = LogisticsExecutor()
    macro = MacroExecutor()
    stockout = StockoutExecutor()
    supervisor = SupervisorExecutor()

    return (
        WorkflowBuilder(start_executor=ingest)
        .add_fan_out_edges(ingest, [demand, logistics, macro])
        .add_fan_in_edges([demand, logistics, macro], stockout)
        .add_edge(stockout, supervisor)
        .build()
    )
