# ADR-0001: Tiering is deterministic code; the LLM never assigns a tier

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §6.3, §13 · architecture.md §17 · ADR-0002

## Context

The system must hit a **≥ 95% routing reproducibility** target: identical inputs must yield identical escalation routing (PRD §11.1). The four-tier escalation model (`ROUTINE` / `MONITOR` / `REVIEW` / `CRITICAL`, PRD §6.3) decides whether an action runs autonomously or is blocked for human review — it is the single most safety-critical control in the system.

LLMs are non-deterministic by default and can be steered by injected content in upstream data. If an LLM assigned the tier, reproducibility would be probabilistic, the highest-impact decisions would be exposed to prompt injection, and audit defensibility would be weak ("the model decided"). At the same time, we still want the LLM's strengths: ranking candidate actions and producing planner-readable explanations.

## Decision

We will make **tier assignment and the autonomy gate pure deterministic functions** of numeric inputs (calibrated stockout probability, quantile interval width, anomaly flags, macro regime), implemented in the governance layer's scoring matrix.

The LLM is confined to: ranking/justifying candidate actions, translating SHAP values into a plain-language brief, and narrative synthesis for the reviewer. **The LLM may propose, but it cannot change a tier.** The `CRITICAL` tier is fully rule-based with no LLM in its path. LLM calls used for narration run at temperature 0 with strict structured output validated against Pydantic schemas.

## Consequences

### Positive
- Reproducibility becomes a property of code, not generation — the ≥95% target is achievable and testable via input-hash memoisation.
- The highest-impact path (`CRITICAL`) has zero LLM attack surface.
- Audit records can point to a deterministic rule + provenance tuple (ADR-0002, architecture.md §20), not "the model's judgement."

### Negative / costs
- The scoring matrix must be explicitly designed and tuned; it cannot rely on LLM "common sense" for edge cases.
- Threshold configuration (per SKU class, $ impact, reversibility) becomes a first-class artefact requiring governance and versioning.

### Neutral / follow-ups
- Thresholds live in PostgreSQL and are read at runtime; changes are version-stamped into decision provenance.
- The reproducibility test in CI (architecture.md §22) gates releases on this invariant holding.

## Alternatives considered

- **LLM assigns tier with structured output + low temperature** — rejected: still non-deterministic at the margins, exposes the safety-critical control to injection, and weakens audit defensibility.
- **LLM assigns tier, validated against rules post-hoc** — rejected: adds complexity for no benefit; if rules are authoritative, let them decide directly.
