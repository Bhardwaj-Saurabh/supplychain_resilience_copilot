"""Agent layer (Layer 4): five framework-agnostic specialist/supervisor agents.

Agents depend on the tool layer and contracts (and governance, for the
Supervisor) — never on LangGraph, ``scrc.ml``, or ``scrc.data`` (enforced). The
orchestration adapter wraps each ``run``/``synthesise`` as a graph node, so the
same agents drive both the LangGraph graph and the MAF port (P5).
"""

from __future__ import annotations

from scrc.agents.demand import DemandAgent
from scrc.agents.logistics import LogisticsAgent
from scrc.agents.macro import MacroAgent
from scrc.agents.stockout import StockoutAgent, assemble_features
from scrc.agents.supervisor import ProvenanceContext, SupervisorAgent

__all__ = [
    "DemandAgent",
    "LogisticsAgent",
    "MacroAgent",
    "ProvenanceContext",
    "StockoutAgent",
    "SupervisorAgent",
    "assemble_features",
]
