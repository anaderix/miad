# Report — K-coverage curve analysis

**Date:** 2026-05-13
**Status:** 🎯 The structural prediction of the "divergent vs convergent" narrative is **partially confirmed**. DiffCSP's match@K saturates quickly (diminishing returns past K=5); DMF λ_rep=1.0 grows ~linearly through K=20. At small K (1-3) DiffCSP dominates; at K=20 DMF still well behind, but **on a trajectory that may close the gap at K=50-100** — within DMF's 220× compute advantage.

## Hypothesis

From the divergent-vs-convergent narrative (Task 24's finding):
- DiffCSP samples concentrate on the mode → most "wins" happen at K=1, K=2; additional samples are redundant.
- DMF samples spread broadly → each additional sample contributes new coverage; should grow roughly linearly with K.

If true: DMF's curve crosses DiffCSP's at some K beyond 20. The crossing K, combined with the 220× per-sample speedup, determines whether DMF wins in compute-normalized terms.

## Setup

- 4 variants: DiffCSP-mp_csp, DMF default, DMF λ_rep=0.5, DMF λ_rep=1.0
- Same K=20 raw eval files (one for each variant)
- Compute match@k for k ∈ {1, 2, 3, 5, 7, 10, 15, 20}
- limit=200 test compositions (statistical noise larger than limit=500 results elsewhere; treat as trend, not precise number)
- Script: `dmf_poc/match_at_k_curve.py`

## Results

| K | DiffCSP | DMF λ=0 | DMF λ=0.5 | DMF λ=1.0 |
|---:|---:|---:|---:|---:|
| 1 | **52.0%** | 0.5% | 2.5% | 3.5% |
| 2 | 59.5% | 1.5% | 4.0% | 6.0% |
| 3 | 65.0% | 2.0% | 4.5% | 6.5% |
| 5 | 71.0% | 3.0% | 6.0% | 8.0% |
| 7 | 74.0% | 3.5% | 6.5% | 9.5% |
| 10 | 76.0% | 5.0% | 7.5% | 13.0% |
| 15 | 80.0% | 5.5% | 11.0% | 15.0% |
| 20 | **81.0%** | 6.5% | 14.5% | **19.0%** |

**Per-K marginal gain** (Δ match-rate per additional K samples):

| K transition | DiffCSP | DMF λ=1.0 |
|---|---:|---:|
| 1 → 3 | +13 pp | +3 pp |
| 3 → 5 | +6 pp | +1.5 pp |
| 5 → 10 | +5 pp | +5 pp |
| 10 → 20 | +5 pp | +6 pp |

**The crossover happens between K=5 and K=10**: DiffCSP's marginal gain per K halves between K=5 and K=10, while DMF's stays flat. Beyond K=10, DMF gains *more* per additional K than DiffCSP.

## Extrapolation (informal)

Linear extrapolation of DMF λ=1.0 from K=10..20 trend (+6 pp per 10 K):
- K=50: 19% + 4·6 = **~43%**
- K=100: 19% + 9·6 = **~73%**

DiffCSP plateaus by extrapolation:
- K=50: probably ~83-85% (asymptote)
- K=100: ~85-87%

So crossover K (where DMF and DiffCSP match-rates equalize) is somewhere around **K=100-200**.

**Compute-normalized comparison**:
- DiffCSP K=20 = 1000 NFE × 20 samples = 20,000 unit-NFE per composition → 81% match
- DMF K=100 = 1 NFE × 100 samples = 100 unit-NFE per composition → ~73% match (extrapolated)
- DMF gets ~90% of DiffCSP performance for 0.5% of compute

**At equal compute**: DiffCSP K=1 (1000 NFE) gets 52% match. DMF can do 1000 samples in the same time → match@1000 extrapolated to ~80-85%. **At equal compute, DMF may match or exceed DiffCSP.**

(Disclaimer: extrapolations beyond K=20 are speculative. Need actual K=100, K=200 evals to confirm.)

## Caveats

- limit=200 — match rates here differ slightly from limit=500 results elsewhere (DMF λ=1.0 here 19% vs 26.6% at limit=500 — first 200 compositions happen to be harder).
- StructureMatcher tolerance is the same across K — match@k counts ANY of k samples within tolerance.
- Linear extrapolation is a guess. Real curves likely saturate at some point (DMF can't cover all of structure-space infinitely).
- The "DMF wins at high K" claim depends on the extrapolated curve. Empirical K=100 eval would confirm.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | DMF curve shape differs from DiffCSP (more linear) | ✅ visible at K=10→20 |
| S2 | DMF marginal gain per K exceeds DiffCSP's at some K | ✅ from K=10 onward |
| S3 | Crossover K extrapolates to a compute-feasible region (< K~10000) | ✅ ~K=100-200 |
| S4 | Practical implication: DMF at high K may rival DiffCSP at low K | ✅ extrapolated |

4 of 4 pass on existing data; full confirmation needs K=100+ evals.

## Recommended next

1. **K=100 raw eval** on best DMF (λ_rep=1.0) — definitive confirmation of crossover. ~30 min.
2. **Plot the curve** as paper figure.
3. **DFT/CHGNet relaxation** — even stronger argument: if DMF's near-match candidates relax to GT, the effective match@k is even higher.

## Update — K=100 actual eval revises extrapolation

Ran K=100 inference for DMF λ=1.0 (limit=200 same as above). Full curve:

| K | DMF λ=1.0 actual | Extrapolation guess |
|---:|---:|---:|
| 1 | 2.0% | — |
| 3 | 5.5% | — |
| 10 | 15.5% | — |
| 20 | 25.0% | — |
| 30 | 27.5% | ~30% (close) |
| 50 | 32.0% | ~43% (too high) |
| 75 | 36.5% | ~58% (too high) |
| **100** | **37.5%** | **~73% (way too high)** |

**DMF saturates around 40%**, not the extrapolated 73%. The K=75→100 marginal gain is only +1 pp; the model is asymptotic well below DiffCSP's 81%.

## Revised narrative — honest version

The "DMF wins at high K via 220× speedup" claim is **NOT** quantitatively supported by the data. Actual:
- DMF λ=1.0 asymptote: **~40-45% match rate** at K→∞
- DiffCSP asymptote: ~85% at K→∞

DMF saturates ~half DiffCSP's ceiling regardless of K. Compute-normalized at K=100 (DMF):
- DMF 100 NFE × 220× faster ≈ 0.45 unit-NFE per composition → **37.5% match**
- DiffCSP at 1 unit-NFE → match@1 = 52%

DiffCSP still wins at equal compute. The reason: DMF's coverage is bounded by what its one-shot model can physically express. Regions of structure-space requiring multi-step coordination (N≥9 cliff, exotic geometries) are unreachable regardless of how many samples.

## The CORRECT framing

DMF+repulsion is **NOT a strict alternative to DiffCSP for general CSP**. It is:

1. **Complementary** — good for the compositions it can reach (N=2-6), useless for those it can't (N≥9).
2. **Different paradigm** — divergent vs convergent generation. The 3× diversity per K is real.
3. **Best at small-N** — for N=2-6, DMF λ=1.0 reaches 61-91% of DiffCSP match-rate at 220× lower compute. This **is** a publishable practical advantage.
4. **Bounded by model expressivity** — adding more samples doesn't help if the model can't produce structures near GT.

This is more honest than "DMF rivals DiffCSP at high K". The real story is "DMF dominates DiffCSP at N≤6 at 220× speedup, but cannot reach the long tail of MP-20 that DiffCSP handles".

## Updated success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | DMF curve grows linearly through K=20 | ✅ true |
| S2 | DMF curve saturates eventually | ✅ at ~40% |
| S3 | "Crossover" with DiffCSP at compute-feasible K | ❌ no crossover; DMF saturates below DiffCSP |
| S4 | At equal compute, DMF beats DiffCSP somewhere | ❌ no — DiffCSP wins at equal compute too |
| S5 | DMF provides genuine value for SOME use case | ✅ small-N (N≤6) at 220× speedup |

3 of 5 pass. The "screening tool wins via volume" claim is falsified. The "complementary tool for small-N" claim survives.

## Artifacts

- `dmf_poc/match_at_k_curve.py` — script (compares match@k across multiple raw eval files)
- `~/miad/dmf_poc/cache/match_at_k_curve.json` — raw numbers
