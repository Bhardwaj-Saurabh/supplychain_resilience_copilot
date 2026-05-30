# ADR-0002: No-audit-no-autonomy invariant

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §7.3, §7.4 · architecture.md §19 · ADR-0001

## Context

The PRD's core thesis is that autonomous action is only acceptable when it is *governed, explainable, and auditable* (PRD §2.3, §7). Gartner attributes 40%+ of agentic AI cancellations to inadequate risk controls. The MLflow audit trail is not an afterthought — it is the mechanism by which the organisation can trust the system to act without a human in the loop.

This raises a failure-mode question Part I left unanswered: **what should the system do when the audit store is unavailable but a decision is otherwise eligible for autonomous execution?** Acting-and-logging-later trades governance for availability; the wrong default here undermines the whole trust model.

## Decision

We will treat **traceability as a precondition for autonomy**. If a decision cannot be durably persisted to the append-only audit store *before* the action is taken, the system **must not act autonomously**. It degrades to **escalate-only**: the decision is routed to a human via the HITL gate, and the action context is buffered to a local write-ahead log for later reconciliation.

Autonomy is a privilege contingent on the ability to record what was done and why.

## Consequences

### Positive
- The audit trail is guaranteed complete for every autonomous action — there is no "acted but unlogged" state.
- Aligns directly with the enterprise-trust requirement that distinguishes this system from prototypes.
- Simple, defensible rule for reviewers and auditors.

### Negative / costs
- Reduced availability of the autonomous path: an audit-store outage converts autonomous decisions into human-review load.
- Requires a durable local WAL and a reconciliation process for buffered context.

### Neutral / follow-ups
- The audit store is append-only/WORM and writable only by the governance layer (PRD §7.4); no agent can write it.
- Health of the audit store is a monitored dependency (architecture.md §19); sustained outages should page.
- A future ADR may revisit the trade-off if an "act + WAL + async reconcile" mode is needed for high-availability deployments.

## Alternatives considered

- **Act autonomously, write audit asynchronously (best-effort)** — rejected as the default: creates a window of unlogged autonomous actions, violating the trust thesis. Retained as a possible future opt-in mode behind an explicit ADR.
- **Block all decisions (including escalation) when audit is down** — rejected: unnecessarily strict; human-reviewed actions are inherently logged by the review workflow and can proceed.
