# Report — Task 27: Drop friction γ (reference-impl style) — mildly negative

**Date:** 2026-05-13
**Status:** ⚠️ Disabling friction γ (using direct drift target as in `/home/anaderi/FlowMatching/DM`'s `drift_loss.py`) yields match@20 = **23.2%** — slightly worse than friction-version repel=1.0 (26.6%). Friction is **not a waste** — it acts as a trust-region warm-up that benefits convergence.

## Hypothesis under test

From `findings.md`:
> Friction γ=t/(MAX_ITER-1) linear 0→1 means second half of training has γ>0.5 → V signal weakens to near-zero. The second 25k iters are "wasted". Reference impl uses direct drift_loss without γ. Removing friction should free up the second half of training.

## Setup

- Default arch (no FiLM), K=1, default lat prior, default V temperatures
- `--repulsion 1.0` (proven best) + **`--no_friction`** (new): γ hardcoded to 0.0
- Direct loss: `target = x_gen + 1·V` (instead of `target = x_gen + (1-γ)·V`)
- 50k iters @ 51.4 it/s — ~16 min train
- Final training loss = **0.37** (vs friction version: 0.0 by construction). Loss is now a real MSE on V-drift target, not trivially zero.

## Results

| Variant | NFE | match@1 | match@20 | RMSE@20 |
|---|---:|---:|---:|---:|
| Default K=1 friction | 1 | 3.0% | 10.6% | 0.34 |
| **repel=1.0 friction** | **1** | **3.4%** | **26.6%** | 0.39 |
| repel=1.0 + **no_friction** | 1 | 2.8% | 23.2% | 0.39 |
| DiffCSP-mp_csp | 1000 | 56.6% | 84.4% | 0.046 |

**Removing friction hurts** match@20 by ~3 pp (10% relative).

## Per-N

| N | repel=1.0 friction | **no_friction** | DiffCSP |
|---:|---:|---:|---:|
| 2 | 79% | 57% | 86% |
| 3 | 86% | 86% | 100% |
| **4** | 80% | **83%** | 99% |
| 5 | 70% | 55% | 95% |
| 6 | 51% | 44% | 84% |
| 7 | 26% | 11% | 79% |
| 8 | 22% | 12% | 84% |
| 9+ | 0% | 0% | 70-90% |

N=4 marginally better without friction (+3 pp). N=2, 5-8 noticeably worse. Net negative for aggregate.

## Interpretation: friction is a trust-region schedule

The naive expectation was: "γ → 1 in second half = wasted training". But the **actual role** of friction is the opposite:

- **Early training** (γ ≈ 0): V signal is at full strength. The model is randomly initialized; V drift directions are large and noisy. Without friction, gradients are dominated by raw V → unstable optimization, possibly overshoot.
- **Late training** (γ → 1): V signal is attenuated. By now the model has learned to produce sensible outputs; small refinements are needed. Friction prevents over-correction.

This is structurally a **gradient warm-up + cooling schedule**. Removing it = full-strength V from iteration 0 = the model spends early iters fighting overconfident drift signals.

The reference impl works around this via **scale normalization inside `drift_loss`** (`scale = weighted_dist.mean() / weight.mean()`). They don't need friction because their loss is already scale-invariant in output magnitude. Our friction-shaped loss does similar (scale modulation) but via the γ schedule.

**Implication**: to drop friction *without* losing quality, we'd also need to port their scale-normalized kernel. Task 28 (scale-normalized V) would be the right composition.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | match@20 ≥ friction version's 26.6% | ❌ 23.2% |
| S2 | Training loss remains meaningful (not trivially 0) | ✅ 0.37 final |
| S3 | At least one per-N stratum improves | ✅ N=4 +3pp to 83% |
| S4 | Output distribution similar to friction version (no degenerate cells) | ✅ lengths 0.71-15.67 |

2 of 4 pass. Net negative on aggregate match-rate; positive on training-signal meaningfulness.

## Key takeaway

**Friction γ is not just a "fade-out for free" trick.** It's a structurally-important schedule that warms up V's influence. The toy notebook's authors got the schedule right despite the "second-half wasted" appearance. Removing it requires compensating elsewhere (scale normalization).

The reference impl's design choice (no friction + scale-normalized loss + class memory bank) is one valid combination. Our (friction + raw kernel + same-N batching) is another. They achieve similar function via different mechanisms.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_repel10_nofric_50k.pt` — ckpt (49.7 MB)
- `~/miad/dmf_poc/cache/csp_metrics_nofric_limit500.json`
- `~/miad/dmf_poc/cache/nofric/eval_gen.pt`, `metrics_partial.json` (gen-metrics computing)
- `train_dmf_mp20.py` now supports `--no_friction`

## Next

- Try **scale-normalized V** (Task 28) — same direction as reference impl, may unlock no_friction's potential
- **Full 9046 CSP eval** on repel=1.0 (friction) — paper-grade CI
- **Diversity analysis** — count unique structural clusters per composition (Task 24)
