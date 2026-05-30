# ADR-0003: Hexagonal layering; the orchestrator is an adapter

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §4.3 (LangGraph rationale), §10 (MAF port) · architecture.md §3, §4, §14

## Context

The project is both a production system and a teaching reference, and it explicitly commits to porting one workflow from LangGraph to Microsoft Agent Framework 1.0 (PRD §10, module 6). If agent reasoning, contracts, and governance were entangled with LangGraph primitives (`StateGraph`, `interrupt()`, `Command`), the MAF port would mean rewriting business logic, and the "framework-agnostic architecture" teaching contribution (PRD §10.2) would not hold.

Separately, the system must remain testable: agents should be unit-testable without spinning up models, databases, or an orchestrator.

## Decision

We will adopt a **hexagonal (ports-and-adapters) architecture** with a strict inward dependency rule:

- **Domain core** (typed contracts + reasoning/escalation policies) depends on nothing.
- **Ports** (tool, feature, audit, HITL interfaces) are defined in the core.
- **Adapters** (ML inference, Feast, MLflow, FastAPI, and the **orchestrator itself**) implement those ports at the edge.
- LangGraph — and the MAF port — are **orchestration adapters**, not the home of business logic. Swapping the orchestrator must not touch domain, tool, or governance code.

Package boundaries enforce the direction: `contracts ← tools ← agents ← orchestration`; `governance` and `observability` are cross-cutting.

## Consequences

### Positive
- The MAF port reuses contracts, reasoning, tools, and governance unchanged; only the orchestration adapter is rewritten — validating the framework-portability thesis.
- Agents are unit-testable against fake tools; the Supervisor is testable against fixture agent outputs (architecture.md §15).
- Infrastructure (model serving, DB, LLM provider) can be swapped behind ports without rippling into domain code.

### Negative / costs
- More upfront structure: interfaces and adapters add indirection and boilerplate versus calling frameworks directly.
- Requires discipline to keep orchestrator-specific concepts out of the core (enforced via package/import linting).

### Neutral / follow-ups
- Import-direction rules should be enforced in CI (e.g. import-linter) so the boundary cannot erode silently.
- The MAF↔LangGraph mapping table (architecture.md §14) is itself a teaching deliverable.

## Alternatives considered

- **Build directly on LangGraph idioms** — rejected: fastest to first demo, but makes the MAF port a rewrite and couples logic to a framework.
- **Full clean/onion architecture with use-case interactors** — rejected as over-engineered for this scope; hexagonal gives the needed portability without the ceremony.
