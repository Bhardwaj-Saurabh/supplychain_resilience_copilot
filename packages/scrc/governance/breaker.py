"""Circuit breaker and timeout helpers (PRD §7.4).

An agent that fails repeatedly or exceeds its timeout trips the breaker; the
orchestration treats a tripped/timed-out call as a missing signal and routes
conservatively (governance §escalation) — never hallucinating a value.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TypeVar

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the breaker is open."""


def call_with_timeout(fn: Callable[[], T], timeout: float) -> T:
    """Run ``fn`` with a wall-clock timeout, raising ``TimeoutError`` if exceeded.

    Note: the worker thread is abandoned (not killed) on timeout — acceptable for
    bounding agent latency in this reference architecture.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise TimeoutError(f"call exceeded {timeout}s timeout") from exc


class CircuitBreaker:
    """Trips open after ``failure_threshold`` consecutive failures; half-opens
    again after ``reset_timeout`` seconds."""

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30.0) -> None:
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        # Half-open (allow a trial call) once the reset window has elapsed.
        return (time.monotonic() - self._opened_at) < self._reset_timeout

    def call(self, fn: Callable[[], T]) -> T:
        if self.is_open:
            raise CircuitOpenError("circuit is open")
        try:
            result = fn()
        except Exception:
            self._record_failure()
            raise
        self._reset()
        return result

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()

    def _reset(self) -> None:
        self._failures = 0
        self._opened_at = None
