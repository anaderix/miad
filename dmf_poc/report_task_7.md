# Report — Task 7: Lattice-prior ablation

**Date:** 2026-05-12
**Status:** ✅ done. Confirms the lattice-collapse hypothesis from Task 6: switching the inference-time lattice prior from `randn+5·I` to `randn·2+10·I` collapses `wdist_density` by **79%** (4.38 → 0.91). But it also tanks `struct_valid` (-41%) and `cov_recall` (-20%) because the model wasn't trained for the wider range. **Verdict: re-training with the wider prior is the next move; inference-only swap is a partial proof-of-concept.**

## Hypothesis under test

From `findings.md` after Task 6:
> Generated lattice lengths span only 1.96–8.27 Å vs MP-20's 2–15 Å.
> Compressing lattice mode coverage hurts density and structural validity.

I.e. the dominant cause of `wdist_density = 4.38` (vs DiffCSP's 0.13) is that DMF outputs lattices in too-narrow a range. Since CSPNet output has `ip=True` (i.e., `L̂ = lattice_out_matrix @ lat_in`), the lattice scale is multiplicatively conditioned on `lat_in`. Therefore widening `lat_in` should widen the lattice output distribution.

## Setup

Same trained DMF checkpoint (`dmf_mp20_50k.pt`). Only inference-time `lat_in` prior changed:

| Config | `lat_in` | l-len range observed (10k samples) | l-len std |
|---|---|---|---|
| **default** | `randn(3,3) + 5·I` | 1.96–8.27 Å | 0.95 |
| **wider** | `randn(3,3)·2 + 10·I` | 1.18–11.79 Å | 1.43 |
| MP-20 (target) | n/a | 2–15 Å (literature) | ~2 |

A 12-config sweep over `(mean, std) ∈ {3,5,7,10} × {0.5, 2.0, 5.0}` confirmed:
- Increasing **mean** of prior shifts output median modestly (4.55 → 5.28 over 3→10).
- Increasing **std** of prior widens output spread (std 0.92 → 2.39 from s=0.5 to s=5).
- At too-extreme priors (s=5), outputs include unphysical 0.1 Å vectors.

`m=10, s=2` gives the most MP-20-like coverage without breakdown.

## Metrics comparison

| Metric | default | wider | Δ% | Direction |
|---|---:|---:|---:|---|
| **`wdist_density`** | 4.376 | **0.906** | **−79%** | ✅ huge improvement |
| `wdist_num_elems` | 0.234 | 0.327 | +40% | ❌ worse |
| `struct_valid` | 0.364 | 0.214 | −41% | ❌ worse |
| `valid` | 0.253 | 0.165 | −35% | ❌ worse |
| `cov_recall` | 0.370 | 0.297 | −20% | ❌ worse |
| `cov_precision` | 0.882 | 0.881 | ≈ | ≈ |
| `comp_valid` | 0.630 | 0.630 | = | unchanged (composition is input) |
| `amsd_*` | 0.560 / 0.193 | 0.585 / 0.202 | +5% | ❌ worse |
| `amcd_*` | 4.84 / 5.17 | 4.84 / 5.17 | = | unchanged |

## Interpretation

The density-wdist collapse is **almost entirely a prior issue**, not a fundamental model failure. Switching prior at inference recovers most of the lost density mass coverage.

But validity drops because the trained model was anchored to the narrower prior — when forced to produce wider lattices via mismatched `lat_in`, it generates *some* unphysical structures (squashed/elongated cells with overlapping atoms). The signal: `struct_valid` 36% → 21% is a 41% relative loss.

This trade-off is the **expected behavior of inference-prior mismatch**. The fix is to **retrain** with the wider prior, so the model adapts to handle the full lattice range physically.

## Success criteria for this ablation

| # | Criterion | Result |
|---|---|---|
| S1 | `wdist_density` drops meaningfully (>50%) with wider prior | ✅ −79% |
| S2 | Lattice length output range expands toward MP-20 (2-15 Å) | ✅ 1.18-11.79 vs 1.96-8.27 |
| S3 | At least one prior config produces *both* better density AND ≥80% of default validity | ❌ wider config trades validity for density |

Two of three criteria pass. The PoC for "prior matters" is complete (S1, S2). S3 isn't reachable without retraining.

## What this changes for the parent verdict

- **Adds nuance.** Task 6's headline "DMF loses 10/11 metrics" was partly inference-time configuration, not fundamental. With prior tuning, density-class metrics can be made competitive at the cost of structural-validity-class metrics. Both pulls are real; a retrained model with `m=10, s=2` priors during training should land somewhere in the middle.
- **The N≥9 cliff remains.** Lattice prior is orthogonal to the many-atom coupling failure; we'd expect retraining to help density+validity without much movement on per-N match rates.

## Concrete next experiment

Retrain DMF with the wider prior baked into training:
- `train_dmf_mp20.py` needs the same `--lat_prior_*` flags propagated
- Same 50k iters, same B=64
- Re-evaluate with matched (wider) prior at inference
- Expected: `wdist_density` ≈ 1, `struct_valid` ≈ 0.4-0.6 (recovered from 0.21), `valid` ≈ 0.3-0.4

Estimated cost: ~36 min retrain + ~30 min metrics = 1h cycle.

## Artifacts

- `dmf_poc/evaluate_dmf.py` — now takes `--lat_prior_mean`, `--lat_prior_std`
- `~/miad/dmf_poc/cache/wider/eval_gen.pt` — 10k DMF crystals with wider prior
- `~/miad/dmf_poc/cache/wider/metrics_partial.json` — metrics for the wider config
- `~/miad/dmf_poc/cache/wider/compare_widerVsdefault.md` — side-by-side comparison table
- `~/miad/dmf_poc/cache/eval_gen_m*.pt` — 12-config sweep outputs (small, 2000 samples each)
