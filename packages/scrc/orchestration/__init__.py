"""Orchestration layer (Layer 4): the LangGraph adapter + portability seams.

LangGraph is imported only in ``graph.py``. The agents it wires are
framework-agnostic, so ``run_pipeline`` (no orchestrator) and the MAF port reuse
them unchanged (P5, ADR-0003).
"""

from __future__ import annotations

from scrc.orchestration.bundle import AgentBundle
from scrc.orchestration.graph import build_graph
from scrc.orchestration.portable import run_pipeline
from scrc.orchestration.runner import (
    interrupt_payload,
    is_review_required,
    resume_decision,
    start_decision,
)
from scrc.orchestration.state import GraphState

__all__ = [
    "AgentBundle",
    "GraphState",
    "build_graph",
    "interrupt_payload",
    "is_review_required",
    "resume_decision",
    "run_pipeline",
    "start_decision",
]
