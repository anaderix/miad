# Report — Task 14: FiLM composition conditioning (negative; "best RMSE, worst match")

**Date:** 2026-05-12
**Status:** ⚠️ FiLM hurts match@20 vs default K=1 (10.6% → 4.6%) but achieves the **best RMSE among matches** (0.224 vs 0.34) and **best amsd_recall** (closeness to GT) of all DMF variants. Pattern: **stronger composition conditioning → tighter convergence to per-composition mean → fewer per-composition diverse samples → fewer matches at K=20**.

## Hypothesis under test

From `findings.md`:
> FiLM-style composition conditioning — per-CSPLayer modulation by atom_types. Strengthens what's currently weak.

Implementation: pool atom_types one-hot per crystal → MLP → per-CSPLayer `(γ, β)`. Inside each layer's forward: `node_output = γ_per_atom * node_output + β_per_atom`. Each CSPLayer therefore sees a fresh composition signal in addition to whatever propagates through node features.

Added 3.2 M parameters (12.3 M → 15.5 M, +26%).

## Setup

- `--film` flag on `train_dmf_mp20.py` and `evaluate_dmf*.py`
- 50k iters @ 49.4 it/s (~10% slower than K=1 default due to extra capacity) → 17 min train
- Default lat prior, K=1, per-graph z

## Results (CSP-strict eval, limit=500)

| DMF variant | match@1 | match@20 | RMSE@20 (Å) |
|---|---:|---:|---:|
| Default K=1 | **3.0%** | **10.6%** | 0.34 |
| K=2 retrain | 2.4% | 8.4% | 0.32 |
| K=5 retrain | 2.2% | 6.2% | 0.31 |
| Per-atom z | 2.0% | 3.0% | 0.26 |
| **FiLM** | 2.6% | **4.6%** | **0.224** |

**FiLM achieves the lowest RMSE among matches (0.224 Å) but the second-worst match-rate (after per-atom z).** Pattern across all variants: stronger conditioning correlates with tighter RMSE among matches and lower match-rate-at-K.

## Per-N (FiLM, limit=500)

| N | n | FiLM @20 | Default @20 |
|---:|---:|---:|---:|
| 2 | 14 | 50.0% | 64.3% |
| 3 | 14 | 14.3% | 35.7% |
| 4 | 70 | 11.4% | 25.7% |
| 5 | 20 | 5.0% | 20.0% |
| 6-8 | 112 | mostly 0-5% | 12-21% |
| 9+ | 268 | 0% | 0% |

Same N≥9 cliff. Small-N performance degraded across the board.

## Interpretation: "the conditioning-diversity trade-off"

The whole extension-arc data points to a single phenomenon:

```
Stronger conditioning (composition signal in more layers, longer iterative refinement, 
per-atom diversification) → model converges more tightly to per-composition mean 
→ RMSE-among-matches improves (closer to GT on average) 
→ but per-composition spread of 20 generations shrinks 
→ match-rate-at-K drops because no individual sample falls in StructureMatcher's tolerance ball
```

The V kernel mean-shift loss optimizes for *being close to the data distribution mean*. It doesn't reward sample-level diversity. Adding more conditioning capacity (FiLM, K-iteration, etc.) gives the model more ability to find that mean *precisely* — but at the cost of the variation needed for match-rate-at-K.

This is now firmly seen in 4 separate experiments:
- K=1 → K=2: match@20 drops, RMSE@matches improves
- K=2 → K=5: match@20 drops, RMSE@matches improves
- K=1 default → FiLM: match@20 drops, RMSE@matches improves
- Per-atom z: an extreme version (per-atom decorrelation actually hurts everything, but the same direction of "more capacity → mode-seeking" pattern)

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | match@20 ≥ default's 10.6% | ❌ 4.6% |
| S2 | N≥9 match rate > 0% | ❌ still 0% |
| S3 | Lowest RMSE among matches | ✅ **0.224** (best of all DMF) |
| S4 | gen-metric improvement (cov_recall, density wdist) | (gen-metrics still computing) |

1 of 3 testable pass — and that one is a "consolation prize" given match-rate is what matters for CSP.

## What this reveals about V-loss DMF in general

**The architecture is not the bottleneck. The loss is.** Kernel-mean-shift V loss rewards mean-seeking. Stronger architectures find that mean faster — but mean-seeking is the wrong objective for CSP, which is fundamentally a coverage problem (must produce *the right* structure, not the *average* structure).

The cure has to come from the loss side:
- **Score-matching loss** (instead of friction-V) — rewards covering the data manifold including tails
- **V with explicit repulsion** — adds anti-mode-seeking term
- **Adversarial / GAN-style** — diversity-aware discriminator
- **Multi-modal diffusion-style** — DDPM-on-lattice + wrapped-normal-frac + D3PM-types (cf. DiffCSP). This isn't really "DMF" anymore, but captures the diversity diffusion gets for free.

None of these is a simple tweak.

## Appendix — FiLM gen-metrics (10k samples)

| Metric | Default | K=2 retrain | K=5 retrain | Per-atom z | **FiLM** |
|---|---:|---:|---:|---:|---:|
| `valid` (↑) | 0.253 | 0.252 | 0.220 | 0.185 | **0.172** worst |
| `struct_valid` (↑) | 0.364 | 0.347 | 0.280 | 0.228 | **0.209** worst |
| `cov_recall` (↑) | 0.370 | 0.389 | 0.400 | 0.240 | **0.512** best |
| `amsd_recall` (↓) | 0.560 | 0.508 | 0.496 | 0.596 | **0.451** best |
| `amsd_precision` (↓) | 0.193 | 0.194 | 0.221 | 0.205 | **0.188** best |
| `wdist_density` (↓) | 4.376 | 6.035 | 5.112 | 2.805 | 4.229 |
| `wdist_num_elems` (↓) | 0.234 | 0.263 | 0.336 | 0.329 | 0.366 |
| `cov_precision` (↑) | 0.882 | 0.882 | 0.882 | 0.882 | 0.882 |

**FiLM is the Pareto-best variant on coverage/distance** (best `cov_recall`, `amsd_recall`, `amsd_precision`) but worst on validity. The model places its outputs *very close* to the GT distribution in aggregate, but many individual outputs aren't physically valid (atom overlaps, bad angles).

This further consolidates the diagnosis: **stronger conditioning ⇒ better aggregate-distribution-fit ⇒ worse individual structural validity**. For practical CSP use this is the wrong direction; for distribution-matching use (e.g. comparing material families' typical lattice ranges), FiLM is the right tool.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_film_50k.pt` — FiLM retrained ckpt (62 MB)
- `~/miad/dmf_poc/cache/csp_metrics_film_limit500.json` — match-rate
- `~/miad/dmf_poc/cache/film/eval_gen.pt` — 10k generated
- `~/miad/dmf_poc/cache/film/metrics_partial.json` — gen metrics (still computing)
- `cspnet_dmf.py`, `train_dmf_mp20.py`, `evaluate_dmf*.py` now support `--film`
