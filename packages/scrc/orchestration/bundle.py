"""The agent bundle — orchestrator-neutral.

Defined apart from any orchestrator so both the LangGraph adapter and the MAF
port (and the framework-agnostic runner) reuse the *same* agents (P5, ADR-0003).
"""

from __future__ import annotations

from dataclasses import dataclass

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    StockoutAgent,
    SupervisorAgent,
)


@dataclass(frozen=True)
class AgentBundle:
    """The five agents, constructed (with real tools) at a composition root."""

    demand: DemandAgent
    logistics: LogisticsAgent
    macro: MacroAgent
    stockout: StockoutAgent
    supervisor: SupervisorAgent
