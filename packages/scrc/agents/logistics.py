"""Logistics Risk Agent (Layer 4)."""

from __future__ import annotations

from scrc.contracts import AnomalyResult, DecisionRequest
from scrc.tools import LogisticsTool


class LogisticsAgent:
    def __init__(self, tool: LogisticsTool) -> None:
        self._tool = tool

    def run(self, request: DecisionRequest) -> AnomalyResult:
        if not request.port_ids:
            raise ValueError("DecisionRequest.port_ids is required for the logistics agent")
        # BTS FAF is the primary signal; assess the request's primary port.
        return self._tool.detect_freight_anomaly(request.port_ids[0])
