from __future__ import annotations

import time

import pytest

from scrc.governance import CircuitBreaker, CircuitOpenError, call_with_timeout


def test_call_with_timeout_returns_fast_result() -> None:
    assert call_with_timeout(lambda: 21 * 2, timeout=1.0) == 42


def test_call_with_timeout_raises_on_slow_call() -> None:
    def slow() -> int:
        time.sleep(1.0)
        return 1

    with pytest.raises(TimeoutError):
        call_with_timeout(slow, timeout=0.05)


def test_breaker_opens_after_threshold_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=30.0)

    def boom() -> int:
        raise ValueError("fail")

    for _ in range(2):
        with pytest.raises(ValueError):
            breaker.call(boom)

    assert breaker.is_open is True
    with pytest.raises(CircuitOpenError):
        breaker.call(boom)


def test_breaker_half_opens_after_reset() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError()))

    # reset_timeout=0 -> immediately half-open; a success closes it again.
    assert breaker.is_open is False
    assert breaker.call(lambda: 7) == 7


def test_breaker_success_resets_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=2)
    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError()))
    assert breaker.call(lambda: 1) == 1  # success resets the counter
    assert breaker.is_open is False
