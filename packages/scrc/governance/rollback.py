"""Rollback registry (PRD §7.4).

Reversible autonomous actions are registered **before** execution so the
Monitoring Agent can reverse them within a window if a later observation
contradicts the action. The registry is a port; the in-memory implementation is
the default for tests/local.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Protocol

from scrc.contracts import ActionType, SCRCModel, SupervisorDecision

DEFAULT_WINDOW = timedelta(hours=24)


class RollbackEntry(SCRCModel):
    entry_id: str
    decision_id: str
    action_type: ActionType
    registered_at: datetime
    window_expires: datetime
    rolled_back: bool = False


class RollbackRegistry(Protocol):
    def register(self, entry: RollbackEntry) -> None: ...
    def pending(self) -> list[RollbackEntry]: ...
    def mark_rolled_back(self, entry_id: str) -> None: ...


class InMemoryRollbackRegistry:
    """In-process rollback registry. Implements ``RollbackRegistry``."""

    def __init__(self) -> None:
        self._entries: dict[str, RollbackEntry] = {}

    def register(self, entry: RollbackEntry) -> None:
        self._entries[entry.entry_id] = entry

    def pending(self) -> list[RollbackEntry]:
        return [e for e in self._entries.values() if not e.rolled_back]

    def mark_rolled_back(self, entry_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry is not None:
            self._entries[entry_id] = entry.model_copy(update={"rolled_back": True})


def register_reversible_actions(
    decision: SupervisorDecision,
    registry: RollbackRegistry,
    window: timedelta = DEFAULT_WINDOW,
) -> list[RollbackEntry]:
    """Register every reversible recommended action before autonomous execution."""
    now = datetime.now(UTC)
    entries: list[RollbackEntry] = []
    for action in decision.recommended_actions:
        if not action.reversible:
            continue
        entry = RollbackEntry(
            entry_id=uuid.uuid4().hex,
            decision_id=decision.decision_id,
            action_type=action.action_type,
            registered_at=now,
            window_expires=now + window,
        )
        registry.register(entry)
        entries.append(entry)
    return entries
