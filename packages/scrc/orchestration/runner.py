"""Run/resume helpers over a compiled decision graph.

Keeps LangGraph's ``Command``/thread-config mechanics in the orchestration layer
so callers (e.g. the FastAPI surface) drive the graph through a small typed API
without importing LangGraph.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.types import Command

from scrc.contracts import DecisionRequest


def start_decision(app: Any, request: DecisionRequest) -> tuple[str, dict[str, Any]]:
    """Run a fresh decision on a new thread; returns (thread_id, result)."""
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"request": request, "errors": []}, config)
    return thread_id, result


def resume_decision(app: Any, thread_id: str, human_outcome: dict[str, Any]) -> dict[str, Any]:
    """Resume an interrupted decision with the planner's outcome."""
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume=human_outcome), config)


def is_review_required(result: dict[str, Any]) -> bool:
    """True when the graph paused for human review (interrupt)."""
    return "__interrupt__" in result


def interrupt_payload(result: dict[str, Any]) -> Any:
    """The ReviewRequest payload surfaced by the interrupt, if any."""
    interrupts = result.get("__interrupt__")
    return interrupts[0].value if interrupts else None
