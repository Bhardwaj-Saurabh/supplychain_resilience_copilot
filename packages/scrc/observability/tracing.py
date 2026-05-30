"""OpenTelemetry tracing helpers (emission layer; ADR-0005).

OTEL spans are emitted and exported to Opik. These helpers **degrade to no-ops**
when OpenTelemetry isn't installed, so the core code paths and tests never
depend on the observability stack being present.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def decision_span(name: str, **attributes: Any) -> Iterator[None]:
    """Open an OTEL span carrying the standard decision attributes (§7.1).

    No-op if OpenTelemetry is unavailable.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        yield
        return

    tracer = trace.get_tracer("scrc")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield
