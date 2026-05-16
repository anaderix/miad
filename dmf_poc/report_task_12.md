# Report — Task 12: K=5 iterative training (negative; mode-collapse with K)

**Date:** 2026-05-12
**Status:** ⚠️ K=5 retrain hurts match-rate vs K=2 retrain, continuing the monotonic decline from K=1. **Refinement-with-V-loss converges to a learned mode** that maximizes V's drift target but reduces structure-space diversity per composition.

## Hypothesis under test

From `findings.md` after Task 10b:
> K=3 retrain (extrapolate K=2's mild improvements). Cost +50% (1h 45min train).

Direct test: train at K=5 with 30k iters (the friction schedule scales fine to fewer iters since γ=t/(T-1) hits 1.0 at the end either way). Check whether more refinement steps continue to improve metrics.

## Setup

- Same `CSPNetDMF`, default lat prior, default V temps, default per-graph z
- ONLY change: `--n_steps 5` during train AND eval
- 30k iters @ 11.5 it/s (5× slower than K=1's 53) = ~43 min train

## Results (CSP-strict eval, limit=500)

| K | match@1 | match@20 | RMSE@20 | Lat length range |
|---:|---:|---:|---:|---|
| **1** (default) | **3.0%** | **10.6%** | 0.34 | 1.96-8.27 Å |
| 2 retrain | 2.4% | 8.4% | 0.32 | 2.15-8.09 Å |
| **5 retrain** | 2.2% | **6.2%** | 0.31 | **2.90-7.83 Å** |

**Match-rate falls monotonically with K. Output lattice range narrows monotonically.** RMSE among matches improves marginally (0.34 → 0.31) but matches are fewer.

## Interpretation

Each iterative refinement step pushes the network toward the V kernel's drift target (the kernel-mean of MP-20 lattices for similar compositions). With K=5, after 5 refinements the output is close to this mean **for every sample**, regardless of z. Diversity collapses.

This is a real phenomenon distinct from the inference-only attractor of Task 10a:
- Task 10a (inference K=5 on K=1-trained): output **shrinks** toward singular cubic cell due to eigenvalue-<1 contraction.
- Task 12 (training K=5 baked in): output **clusters** toward MP-20 mean per composition. Doesn't shrink absolutely — it converges to a learned class-mean.

The shared structural issue: **V kernel's drift target is the data mean, not the data distribution**. The friction-MSE loss pushes the model toward this mean. More refinement = more mean-seeking = less diversity = lower match-rate-at-K (which needs diverse coverage).

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | match@20 ≥ K=2 retrain's 8.4% | ❌ 6.2% |
| S2 | Output lattice range stays ≥ default's 1.96-8.27 | ❌ narrowed to 2.90-7.83 |
| S3 | Any per-N improvement | (gen-metrics still computing) |
| S4 | NFE-normalized match rate (match@20 ÷ K) ≥ default | ❌ default: 10.6, K=2: 4.2, K=5: 1.24 — strict monotone decrease |

0 of 3 testable pass. **Seventh negative result in the extension arc.**

## What this strengthens

After 7 of 7 extension experiments yielding net-negative or zero-net-positive results, the structural ceiling of the (kernel-mean-shift V) + (one-shot or K-shot CSPNet) + (same-N batching) framework is **firmly established** as the operating point we get. No knob tested moves match-rate up.

**The fundamental issue**: V kernel pulls toward batch mean. This is great for guiding a *direction* of training but is the *wrong target* for sample-level match-rate. To improve match-rate, you need:
- A loss that rewards individual sample diversity, not just mean-seeking (e.g. score-matching, or V with repulsion)
- OR enough iterative steps with stochastic sampling that diversity is preserved (diffusion-style)

Both are major code rewrites beyond simple V-temperature or K tweaks.

## Final pattern across iterative training

| Aspect | K=1 | K=2 | K=5 | Trend |
|---|---:|---:|---:|---|
| match@20 | 10.6% | 8.4% | 6.2% | ↓ monotone |
| RMSE@20 | 0.34 | 0.32 | 0.31 | ↓ tiny |
| Lat range | wide | mid | narrow | ↓ monotone |
| NFE per gen | 1 | 2 | 5 | ↑ |
| Effective DiffCSP speedup | 220× | 110× | 44× | ↓ |

**No K is a Pareto improvement.** K=1 dominates K=2 and K=5 on every relevant axis except RMSE-among-matched (where it's marginal). The whole iterative-training arc is a wash.

## Appendix — Gen metrics for K=1, K=2-retrain, K=5-retrain

| Metric | K=1 (default) | K=2 retrain | K=5 retrain | trend |
|---|---:|---:|---:|---|
| `valid` (↑) | 0.253 | 0.252 | **0.220** | ↓ monotone |
| `struct_valid` (↑) | 0.364 | 0.347 | **0.280** | ↓ monotone |
| `cov_recall` (↑) | 0.370 | 0.389 | **0.400** | ↑ monotone |
| `amsd_recall` (↓) | 0.560 | 0.508 | **0.496** | ↓ monotone (better) |
| `wdist_density` (↓) | 4.376 | 6.035 | 5.112 | mixed |
| `cov_precision` (↑) | 0.882 | 0.882 | 0.882 | flat |
| `amsd_precision` (↓) | 0.193 | 0.194 | 0.221 | ↑ (worse) |
| `comp_valid`, `amcd_*` | — | — | — | flat |

**The K-iteration trade-off, cleanly visible:**
- **Coverage and structural-closeness improve** with K (cov_recall, amsd_recall monotonically better).
- **Validity and density distribution worsen** with K (valid, struct_valid drop; wdist_density swings).
- **Match@20 drops** because validity dominates: of the K=5 samples, fewer are valid crystals → fewer can match any GT.

In a sentence: *K-step iterative DMF training pushes the model toward the V-kernel batch mean (good for "average distance to GT") at the cost of physical structural validity (bad for "this is a real crystal that could match a real GT structure"). The two effects cancel for match-rate.*

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_K5_30k.pt` — K=5 retrained ckpt
- `~/miad/dmf_poc/cache/csp_metrics_K5_limit500.json` — match-rate
- `~/miad/dmf_poc/cache/K5/eval_gen.pt` — 10k generated
- `~/miad/dmf_poc/cache/K5/metrics_partial.json` — gen metrics (still computing)

## Recommendation

Stop trying more K values. The trend is robust. Focus future effort on either:
1. **Mode-aware V** (with repulsion term — Task 13/15) — direct attempt at diversity
2. **Score-matching loss** — abandon kernel V entirely
3. **Multi-modal diffusion-style DMF** (Task 21) — capture diffusion's modality handling at lower NFE

But all three are substantial rewrites with unguaranteed payoff.
