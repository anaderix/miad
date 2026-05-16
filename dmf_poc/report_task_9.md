# Report — Task 9: V kernel temperature ablation (negative result)

**Date:** 2026-05-12
**Status:** ⚠️ Hypothesis from Task 8 only weakly confirmed; validity collapses. Together with Task 7+8 this paints a consistent picture: simple hyperparameter tuning of the DMF setup moves the model along a **validity-vs-density Pareto frontier** but cannot escape it. The structural ceiling appears to be the kernel-mean-shift framework itself with same-N batching.

## Hypothesis under test

From Task 8's deeper finding:
> Higher τ → softer kernel → V toward batch mean instead of nearest neighbor → wider equilibrium output distribution.

Concrete prediction: increase `temperatures_L` from `[0.5, 1.0, 2.0]` to `[5.0, 10.0, 20.0]`, retrain, expect (a) wider output lattice distribution, (b) lower `wdist_density`, (c) hopefully maintained or improved `struct_valid` (since the model trained for this).

## Setup

- Same `CSPNetDMF`, same 50k iters, same B=64, default lat prior (`m=5, s=1`)
- ONLY change: `--temps_L 5.0,10.0,20.0` during training (10-20× higher than default)
- Same temps at inference (implicitly via the new model's learned mapping)

## Results

| Metric | default | hightempL | Δ% | Direction |
|---|---:|---:|---:|---|
| `wdist_density` (↓) | 4.376 | 4.272 | **−2.4%** | weak ✅ |
| `struct_valid` (↑) | 0.364 | **0.199** | **−45%** | ❌ |
| `valid` (↑) | 0.253 | 0.147 | −42% | ❌ |
| `cov_recall` (↑) | 0.370 | 0.319 | −14% | ❌ |
| `wdist_num_elems` (↓) | 0.234 | 0.339 | +45% | ❌ |
| `amsd_recall` (↓) | 0.560 | 0.529 | −6% | weak ✅ |
| `cov_precision` (↑) | 0.882 | 0.882 | = | = |
| `comp_valid` (↑) | 0.630 | 0.630 | = | = (composition is input) |
| `amcd_*` | — | — | = | = |

**Output lattice range:** 1.78–9.49 Å (default: 1.96–8.27 Å). Slight widening, as predicted. But the win on density-wdist is tiny (-2.4%) and validity collapsed.

## Interpretation

Task 8 predicted higher τ would widen output and improve density. Output DID widen (lower lat_min, higher lat_max) but **density-wdist barely moved**. This means the wider lattice range is not where the density distribution mismatch lives — it must be in *which densities are over- and under-represented*, not in raw spread.

The validity collapse (-45% on `struct_valid`) is consistent with the Task 7 pattern: any move toward wider outputs makes more structures physically problematic. The validity-density trade-off has the same shape regardless of which knob we turn (prior, kernel temperature). **This is the Pareto frontier the model is stuck on.**

## The three negative results so far make a pattern

| Experiment | Lattice range | density | validity | Result |
|---|---|---:|---:|---|
| Task 6 (default DMF) | 1.96–8.27 Å | 4.38 | 0.364 | baseline |
| Task 7 (inference prior swap) | 1.18–11.79 Å | **0.91** | 0.214 | density ✅ validity ❌ |
| Task 8 (retrain w/ wider prior) | 2.33–8.00 Å | 6.06 | 0.278 | ❌ both |
| Task 9 (retrain w/ higher τ) | 1.78–9.49 Å | 4.27 | **0.199** | density ≈ validity ❌ |

Always: widening lattice spread (in any way) helps density at most marginally and hurts validity substantially. Density-wdist of ~4.4 appears to be a **structural ceiling** of the same-N kernel V approach with B=64 on MP-20.

## Success criteria for Task 9

| # | Criterion | Result |
|---|---|---|
| S1 | `wdist_density` reduced by ≥20% | ❌ only −2.4% |
| S2 | `struct_valid` maintained within 10% of default | ❌ dropped 45% |
| S3 | Output lattice range widened toward MP-20's 2-15 Å | ✅ widened mildly (1.78-9.49) |
| S4 | At least one ↑-better metric improved | ❌ only mild wins on amsd_recall and cov_precision (both ~rounding-error) |

1 of 4 criteria pass. Net: hypothesis from Task 8 falsified at the level of *practical improvement*.

## What this means for the parent verdict

After three attempts to improve DMF on MP-20 (prior at inference, retrain with wider prior, retrain with higher V τ), the model's metrics on this dataset appear to be **structurally limited** by:
1. The choice of V kernel (mean-shift) — bounds output distribution to batch-target distribution
2. Same-N batching at fixed composition — limits how "diverse" the batches can be
3. The fundamental one-shot architecture — without iterative refinement, multi-atom coordination problem is hard

Further hyperparameter tuning is unlikely to break the validity/density trade-off. The path forward requires architectural changes:

1. **FiLM-style composition conditioning** (predicted Task 3) — composition gates feature modulation in CSPLayer
2. **Per-atom z** instead of per-graph z — more diversity per atom
3. **Modified V with repulsion** — explicit "spread the output" term, not just attraction
4. **Iterative DMF** — drop the "one-shot" constraint, apply k forward steps with friction-like decay (2-5 steps would likely capture most of diffusion's quality at fraction of compute)

Option 4 is particularly promising: it gives up some of the throughput win (NFE=5 instead of 1, still 200× faster than diffusion's NFE=1000) for likely substantial quality gains.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_hightempL_50k.pt` — retrained checkpoint
- `~/miad/dmf_poc/cache/hightempL/eval_gen.pt`, `metrics_partial.json`
- `/tmp/dmf-train-hightempL.log` — training log
