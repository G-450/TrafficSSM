# Project documentation — source of truth

This directory is the authoritative documentation for **Uncertainty-Aware Time-Series Forecasting via Deep Spatial-Temporal State Space Models (ST-DSSM)**. It defines the project as it is now, the decisions that govern it, and the work still required. When a legacy document disagrees with this directory, this directory wins.

## Start here

| Need | Canonical document |
|---|---|
| The finished study, with evidence links | [Final report](FINAL_REPORT.md) |
| What and why we are building | [PRD](PRD.md) |
| What exists today | [Current state](CURRENT_STATE.md) |
| Technical design | [Architecture](ARCHITECTURE.md) |
| Data rules and interfaces | [Data contract](DATA_CONTRACT.md) |
| Experiments and scientific claims | [Research and experiments](RESEARCH_AND_EXPERIMENTS.md) |
| What the results do **not** support | [Limitations and claim reconciliation](LIMITATIONS.md) |
| Delivery sequence | [Implementation plan](IMPLEMENTATION_PLAN.md) |
| Quality, reproducibility, and testing | [Quality and reproducibility](QUALITY_AND_REPRODUCIBILITY.md) |
| Git, commits, pull requests, and merging | [Development workflow](DEVELOPMENT_WORKFLOW.md) |
| Required directory layout and placement rules | [Repository structure](REPOSITORY_STRUCTURE.md) |
| Binding design decisions | [Decision records](DECISIONS/README.md) |
| Questions that must not be guessed | [Open questions](OPEN_QUESTIONS.md) |

## Documentation rules

1. Use the words **implemented**, **verified**, **planned**, and **proposed** precisely. A planned feature is never evidence that it works.
2. Every material ambiguity must become either an open question or a decision record before implementation depends on it.
3. Update the relevant canonical document in the same pull request as any behavior, interface, experiment, or policy change.
4. Preserve evidence: configuration, code revision, data provenance, random seeds, and results belong with every reported experiment.
5. The compact files in [`.agent-context/`](../.agent-context/README.md) are the entry point for AI agents. They summarize these documents; they do not override them.

## Legacy material

`REPORT.md`, `prob.md`, and `PHASE1_SUMMARY.md` were reviewed as historical inputs. They are not canonical because they contain future-state descriptions and, in places, statements that conflict with the checked implementation. The canonical write-up of the finished study is [FINAL_REPORT.md](FINAL_REPORT.md), whose every number is traceable to a locked deliverable; the legacy `REPORT.md` is not its predecessor and should not be read as one. Their useful requirements have been reconciled here; unresolved differences are listed in [open questions](OPEN_QUESTIONS.md).
