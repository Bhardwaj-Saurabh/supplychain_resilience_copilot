# ADR-0007: Evaluation-as-release-gate; model promotion decoupled from app deploy

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §8, §11.1 · architecture.md §11, §22 · ADR-0001

## Context

The system has hard behavioural targets: routing reproducibility ≥ 95%, stockout AUC ≥ 0.85 / ECE ≤ 0.05, IsoForest F1 ≥ 0.70, and no regression on disruption-replay scenarios (COVID, Suez 2021, Red Sea 2024). In Part I these existed as runtime/eval concerns but had no enforcement point — nothing prevented a regression from shipping.

Two lifecycles also move at different cadences: **application code** (agents, orchestration, governance) and **ML models** (retrained on drift, PRD §8.2). Coupling model promotion to application deploys would force a full redeploy for every model update and complicate rollback.

## Decision

We will make **evaluation a release gate**: the CI pipeline runs lint/type/unit → consumer-driven contract tests → integration on an ephemeral Compose stack → an **Opik-backed eval gate** that blocks the release if any behavioural threshold regresses against versioned golden baselines.

Separately, **model promotion is decoupled from application deployment**: a model version is promoted by flipping its MLflow Model Registry production alias after it passes its own evaluation. The application resolves the production alias at call time (architecture.md §8), so a model update — and its rollback — is an alias flip, not an app redeploy.

## Consequences

### Positive
- Behavioural regressions cannot reach production silently; the disruption-replay and reproducibility tests have a real enforcement point.
- Model updates and app releases move independently; model rollback is instant (alias flip).
- Golden datasets make "no regression" objective and versioned.

### Negative / costs
- CI must stand up an ephemeral Compose stack and run the eval suite — slower, heavier pipelines than unit tests alone.
- Golden baselines must be curated and maintained as scenarios evolve; a stale baseline can mask or cause false regressions.
- Two release mechanisms (app deploy vs registry alias) require clear runbooks to avoid confusion.

### Neutral / follow-ups
- Eval thresholds are defined alongside the datasets in `eval/`; changes to thresholds are themselves reviewed.
- Drift-triggered retraining (PRD §8.2) feeds candidates into the same eval gate before alias promotion, with HITL approval when the auto-promotion margin is not met.

## Alternatives considered

- **Evaluation as a dashboard only (no gate)** — rejected: provides visibility but does not prevent regressions from shipping.
- **Promote models by redeploying the app with pinned versions** — rejected: couples lifecycles, slows model updates, and makes rollback a redeploy instead of an alias flip.
