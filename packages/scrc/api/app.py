"""FastAPI surface (Layer 6): the system's only inbound interface.

Exposes the decision flow and the **HITL approval webhook** (PRD §6.2, §12 — the
integration seam in lieu of pre-built ERP connectors). Decision logic lives
below this layer; the API only translates HTTP to graph runs and back.

``create_app`` takes a *compiled decision graph* — composition (wiring real
tools/models/audit) happens at a deployment entrypoint, keeping this module
testable with a fake-backed graph.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Response, status

from scrc.contracts import DecisionRequest, HumanDecision
from scrc.orchestration import (
    interrupt_payload,
    is_review_required,
    resume_decision,
    start_decision,
)


def create_app(decision_graph: Any) -> FastAPI:
    api = FastAPI(title="Supply Chain Resilience Co-Pilot", version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/decisions")
    def create_decision(request: DecisionRequest, response: Response) -> dict[str, Any]:
        """Run a decision. Autonomous tiers complete; escalated tiers return 202
        with the ReviewRequest for a planner to resolve via the resume webhook."""
        thread_id, result = start_decision(decision_graph, request)
        if is_review_required(result):
            response.status_code = status.HTTP_202_ACCEPTED
            return {
                "thread_id": thread_id,
                "status": "review_required",
                "review": interrupt_payload(result),
            }
        return {
            "thread_id": thread_id,
            "status": "completed",
            "decision": result["decision"].model_dump(mode="json"),
        }

    @api.post("/decisions/{thread_id}/resume")
    def resume(thread_id: str, decision: HumanDecision) -> dict[str, Any]:
        """HITL approval webhook: resume an interrupted decision with the
        planner's outcome and return the resolved decision."""
        result = resume_decision(decision_graph, thread_id, decision.model_dump(mode="json"))
        return {
            "thread_id": thread_id,
            "status": "resolved",
            "human_outcome": result.get("human_outcome"),
            "decision": result["decision"].model_dump(mode="json"),
        }

    return api
