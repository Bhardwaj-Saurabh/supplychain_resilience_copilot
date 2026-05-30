# ADR-0006: Single PostgreSQL, schema-separated, over per-service databases

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §4.3 (PostgreSQL rationale), §9 · architecture.md §12, §13

## Context

Four subsystems need durable relational storage: the LangGraph checkpointer (graph state / HITL resume), MLflow metadata + audit, the Airflow scheduler, and Feast (offline + online feature stores). The PRD targets a one-command `docker-compose up` cold start on a developer laptop and a cloud deployment on Azure Database for PostgreSQL.

Running four separate database engines maximises isolation but multiplies operational surface, memory footprint, and Compose complexity — working against the laptop cold-start goal and the teaching clarity objective.

## Decision

We will run a **single PostgreSQL instance with separate schemas** per concern (`langgraph_checkpoints`, `mlflow`, `airflow`, `feast`). Logical separation by schema preserves clear ownership boundaries and per-schema access scoping, while sharing one engine for operational simplicity and transactional consistency on feature reads/writes. In cloud deployment this maps to a single Azure Database for PostgreSQL Flexible Server.

## Consequences

### Positive
- Simple, fast local cold start; fewer containers; lower resource footprint.
- One backup/restore, one connection-pool story, one engine to operate and teach.
- Transactional consistency available across feature reads/writes within one engine.

### Negative / costs
- Shared blast radius: a database-level outage affects all four concerns simultaneously (mitigated by the resilience rules in architecture.md §19 — e.g. checkpointer-down rejects new runs).
- Noisy-neighbour risk: heavy Airflow or Feast load could affect checkpointer latency; requires connection-pool and resource governance.
- Per-schema scoping must be actively configured to preserve least-privilege (the audit schema is governance-write-only — ADR-0002).

### Neutral / follow-ups
- If a single concern outgrows the shared instance (likely Feast online at scale), split it out via a new ADR; the schema separation makes extraction low-risk.
- Apply per-schema roles so application identities cannot cross schema boundaries.

## Alternatives considered

- **One database engine per service (four engines)** — rejected: strongest isolation but heaviest ops/footprint, hurting laptop cold start and teaching clarity.
- **Single database, single shared schema** — rejected: loses ownership boundaries and per-schema access control, making the audit-write-only invariant hard to enforce.
