# Current focus

**The project is released as v1.0.0. All fourteen phases are complete and locked,
and there is no in-flight phase.** Do not start experimental work without a new
authorization; the deliverables below are frozen evidence.

Read [`docs/FINAL_REPORT.md`](../docs/FINAL_REPORT.md) first — it is the finished
study with every number linked to the deliverable and command that reproduces it
— and [`docs/LIMITATIONS.md`](../docs/LIMITATIONS.md) before repeating any claim,
because several intuitive readings of these results are wrong and are documented
as such.

## The three locked deliverables

| Deliverable | File | Verify with |
|---|---|---|
| Normal-condition study (7 manifests) | `experiments/normal/phase10_results.md` | `python scripts/aggregate_phase10.py --check` |
| Robustness condition matrix (30 manifests) | `experiments/missingness/phase12_results.md` | `python scripts/aggregate_phase12.py --check` |
| Calibration, error analysis, latency | `experiments/calibration/phase13_results.md` | `python scripts/aggregate_phase13.py --check` |

Never hand-edit a locked results file; it is generated from checksummed
manifests, and `--check` will fail. The Phase 13 aggregator recomputes from the
git-ignored `*_predictions.npz` artifacts (~160 MB each), so it needs them on
disk and takes ~25 minutes.

## Traps to avoid when citing these results

- **Aggregate interval ECE improves under missingness while NLL triples.** That
  is an artifact of bulk over-dispersion, not robustness. Cite the per-group
  tables and NLL instead.
- **The masked-training ablation is a different model** from a primary condition
  run (ADR-0011) and must never be averaged into a primary cell. It exists at 20%
  only, for ST-DSSM only, and assumes the outage pattern is known at training
  time.
- **No claim that the model is calibrated after temperature scaling** is
  supportable: the fitted scalar lowers NLL while raising interval ECE.
- **The latent state is not claimed to confer or to harm robustness.** The Phase
  12 seed variance turned out to be a train/test representation mismatch, which
  the ablation demonstrated.

## If more work is authorized

The highest-value extensions, in order, are in `docs/FINAL_REPORT.md` §11: a
heavier-tailed predictive family (the shape mismatch is present with no masking
at all), and conditioning the predictive scale on the observation mask so
uncertainty localizes where evidence was lost. Lower priority: masked-training at
10%/30% (~4.5 GPU-hours per run), per-condition temperatures (outside the locked
ADR-0008 protocol), correlated outage mechanisms, and a second sensor network.

See `docs/IMPLEMENTATION_PLAN.md` for the phase record and gates, and consult the
accepted decision records in `docs/DECISIONS/` before making a material design
choice. Record any newly discovered ambiguity in `docs/OPEN_QUESTIONS.md`.
