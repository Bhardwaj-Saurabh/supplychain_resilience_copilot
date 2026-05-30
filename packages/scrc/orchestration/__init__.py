"""Orchestration layer (Layer 4): the LangGraph adapter.

The only place LangGraph is imported. Wires the framework-agnostic agents into a
checkpointable StateGraph; swapping this module for the MAF port (Module 6)
leaves agents, tools, governance, and contracts untouched (P5, ADR-0003).
"""

from __future__ import annotations

from scrc.orchestration.graph import AgentBundle, build_graph
from scrc.orchestration.state import GraphState

__all__ = ["AgentBundle", "GraphState", "build_graph"]
