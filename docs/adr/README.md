# Architecture Decision Records

This directory records the significant architectural decisions for the Supply Chain Resilience Co-Pilot, using lightweight [Nygard-style ADRs](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions). Each record captures the context, the decision, and its consequences at the time it was made.

**Conventions**
- One decision per file, named `NNNN-kebab-title.md`.
- Numbers are immutable; a superseded ADR keeps its number and is marked `Superseded by ADR-NNNN`.
- To change a past decision, write a *new* ADR that supersedes it — do not rewrite history.
- Status values: `Proposed` · `Accepted` · `Superseded` · `Deprecated`.
- Start from [`template.md`](template.md).

| ADR | Title | Status | Date |
|---|---|---|---|
| [0001](0001-deterministic-tiering.md) | Tiering is deterministic code; the LLM never assigns a tier | Accepted | 2026-05-30 |
| [0002](0002-no-audit-no-autonomy.md) | No-audit-no-autonomy invariant | Accepted | 2026-05-30 |
| [0003](0003-hexagonal-orchestrator-adapter.md) | Hexagonal layering; the orchestrator is an adapter | Accepted | 2026-05-30 |
| [0004](0004-chronos2-over-timegen.md) | Retain Chronos-2 (Azure-hosted, open-source) over TimeGEN-1 | Accepted | 2026-05-30 |
| [0005](0005-opik-observability-eval.md) | Opik for agent tracing + evaluation; Prometheus/Grafana for infra only | Accepted | 2026-05-30 |
| [0006](0006-single-postgres-schema-separated.md) | Single PostgreSQL, schema-separated, over per-service databases | Accepted | 2026-05-30 |
| [0007](0007-eval-as-release-gate.md) | Evaluation-as-release-gate; model promotion decoupled from app deploy | Accepted | 2026-05-30 |
| [0008](0008-azure-openai-via-foundry.md) | Azure OpenAI via Azure AI Foundry; single-provider deployment failover | Accepted | 2026-05-30 |

See [../architecture.md](../architecture.md) for the full design these decisions support.
