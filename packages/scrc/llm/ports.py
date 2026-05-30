"""LLM port (capability layer).

The LLM is a *reasoning/narration* utility, never a predictor (P1, ADR-0001).
Callers depend on this protocol; the Azure adapter and any fake implement it.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...
