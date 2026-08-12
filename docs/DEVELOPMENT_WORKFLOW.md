# Development workflow and Git discipline

## Branches

- `main` (or the repository's protected integration branch): stable, reviewed work only.
- `codex/<area>-<short-description>`: all new AI-agent and contributor work. Example: `codex/data-cadence-validation`.
- Keep a branch focused on one cohesive outcome. Do not combine refactors, data-policy changes, and model features unless they are inseparable.

The current branch is `phase-1`; it is the clean starting baseline for Phase 1, not evidence that Phase 1 work is already implemented and not a signal that future work should bypass review.

## Repository placement

Follow the binding [repository structure](REPOSITORY_STRUCTURE.md). In particular, reusable code goes under `src/st_dssm/`, named configurations under `configs/`, tests under `tests/`, and generated model artefacts under ignored `artifacts/` paths.

## Commits

Make small, descriptive commits using this format:

```text
<type>(<scope>): <imperative summary>
```

Allowed types: `docs`, `feat`, `fix`, `test`, `refactor`, `build`, `ci`, `chore`. Examples:

```text
docs(architecture): define masked-input interface
fix(data): enforce five-minute timestamp cadence
test(metrics): cover Gaussian CRPS reference cases
feat(masking): add reproducible sensor-node masks
```

Commit only files belonging to the change. Never commit raw data, generated tensors, checkpoints, or secrets. Each commit must leave the repository in a coherent state where practical.

## Pull requests

Open a PR from a focused branch into the integration branch. Its description must include:

1. Problem and intended outcome.
2. Scope and explicitly excluded work.
3. Linked ADR/open question, if applicable.
4. Tests/checks run and results.
5. Data/model/experiment impact, including backward compatibility.
6. Documentation changes.
7. Evidence artefacts or a statement that none were generated.

PR title follows the same conventional format as commits. An AI agent must not self-approve a PR; a human owner reviews scientific-design changes, data-policy changes, and final-result claims.

Use the repository [pull-request template](../.github/PULL_REQUEST_TEMPLATE.md). Bug reports and research/design decisions use the corresponding issue templates under `.github/ISSUE_TEMPLATE/`.

## Review checklist

- Does the code match the documented interface and decision records?
- Are input/target/mask paths free of leakage?
- Are units, shapes, seeds, and output metadata explicit?
- Do tests cover the new behavior and failure path?
- Are claims limited to evidence?
- Is generated/large/sensitive material excluded from Git?

## Merge policy

Use squash merge for a PR that represents one cohesive outcome; its final commit message uses the approved PR title. Rebase or update the branch before merging if it has diverged materially. Do not merge with failing required checks, unresolved review comments, or undocumented design changes. Tag only a reviewed, reproducible milestone.

## Decision and issue discipline

Create an ADR before implementing a decision that changes architecture, data semantics, experimental protocol, or reproducibility policy. Create an issue/task for defects and research questions; link it from the PR. If an AI agent finds ambiguity, it must update [open questions](OPEN_QUESTIONS.md) or propose an ADR rather than guessing.
