"""Pure scoring functions for the evaluation harness."""

from __future__ import annotations

from collections.abc import Sequence

from scrc.contracts import EscalationTier


def reproducibility_score(reproducible_flags: Sequence[bool]) -> float:
    """Fraction of cases whose tier was identical across repeated runs (≥0.95 target)."""
    if not reproducible_flags:
        return 1.0
    return sum(reproducible_flags) / len(reproducible_flags)


def accuracy_score(pairs: Sequence[tuple[EscalationTier, EscalationTier]]) -> float:
    """Fraction of (observed, expected) tier pairs that match."""
    if not pairs:
        return 1.0
    return sum(1 for observed, expected in pairs if observed is expected) / len(pairs)
