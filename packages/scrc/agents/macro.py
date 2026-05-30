"""Macro Signal Agent (Layer 4)."""

from __future__ import annotations

from scrc.contracts import MacroSignals
from scrc.tools import MacroTool


class MacroAgent:
    def __init__(self, tool: MacroTool) -> None:
        self._tool = tool

    def run(self) -> MacroSignals:
        return self._tool.assess_macro()
