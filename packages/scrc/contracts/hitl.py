"""Human-in-the-loop decision payload (the HITL webhook body)."""

from __future__ import annotations

from pydantic import Field

from scrc.contracts.common import SCRCModel
from scrc.contracts.decision import ActionType


class HumanDecision(SCRCModel):
    """A planner's response to a ``ReviewRequest`` (PRD §6.2).

    Resumes the interrupted graph. ``override_action`` lets the planner replace
    the top recommended action; ``approved=False`` rejects the recommendation.
    """

    approved: bool
    reviewer_id: str = Field(min_length=1)
    note: str | None = None
    override_action: ActionType | None = None
