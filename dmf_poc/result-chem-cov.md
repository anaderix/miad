# Result — TASK chem-cov: covalent-radius-weighted V kernel

**Date:** 2026-05-17
**Status:** ⚠️ Marginal. cov (τ=0.5) gives **+10% relative match@20** over baseline (17.0% vs 15.5%) but **underperforms all Z variants** (chem2=20.5%, chem4=18.0%, chem16=19.5%). Covalent radius is *not* a better chemistry proxy than raw Z for the V-kernel weighting.

## Hypothesis

Raw Z-diff is a crude chemistry proxy: e.g. Al (Z=13) vs Ga (Z=31) sit in the same group (similar r_cov ~1.25-1.30 Å) but Z-diff=18 zeroes out their pair weight under chem_temp=4. A radius-based metric should treat them as similar → better "structural role" matching.

## Implementation

Added `chem_metric={"Z","cov"}` to `compute_V_frac`/`compute_V`/`train_dmf_mp20.py`.
- `_load_cov_radii(device)` lazy-loads `pymatgen.core.Element(Z).atomic_radius` into a (101,) tensor (fallback 1.0 Å for missing).
- Weighting: `chem_w = exp(-(r_cov(Z_a) - r_cov(Z_b))² / chem_temp)`.
- chem_temp=0.5 chosen so Δr_cov=0.5 Å → weight 0.61 (loosely comparable to Z-temp=4).

**Smoke test** confirmed expected behavior on Al-O-O vs Ga-O-O target:
- `|V_Z(τ=4)|=0.22` (Z metric heavily downweights Al↔Ga mismatch)
- `|V_cov(τ=0.5)|=0.28` ≈ `|V_blind|=0.28` (cov treats Al≈Ga → same as no chemistry)

So cov *does* behave as designed; the question is whether that behavior helps generation.

## Setup

- Same 50k iter, repulsion=1.0, MP-20, batch=64, lr=5e-4 on ctor-gpu (A100 MIG 3g.40gb)
- CSP eval: limit=200, K=20

## Results (limit=200, K=20)

| Variant | match@1 | **match@20** | RMSE@20 | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline (no chem) | 2.0% | 15.5% | 0.398 | — |
| **chem-cov τ=0.5** | 2.5% | **17.0%** | 0.408 | **+10% rel** |
| chem-Z τ=2 | 4.5% | 20.5% | 0.391 | +32% rel |
| chem-Z τ=4 | 1.5% | 18.0% | 0.389 | +16% rel |
| chem-Z τ=16 | 3.0% | 19.5% | 0.388 | +26% rel |

cov-radius is **worst chem-aware variant** but still beats baseline.

## Per-N (cov vs baseline vs chem-Z best)

| N | n | baseline | chem-Z τ=2 | **chem-cov** | Δ cov vs baseline |
|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 100% | 100% | 67% | **−33** |
| 3 | 4 | 100% | 100% | 75% | **−25** |
| 4 | 24 | 50% | **79%** | 54% | +4 |
| 5 | 8 | 88% | 75% | 75% | −13 |
| 6 | 18 | 39% | 39% | 22% | **−17** |
| 7 | 10 | 10% | n/a | 20% | +10 |
| 8 | 25 | 8% | 12% | 16% | +8 |
| ≥9 | many | 0% | 0% | 0% | = |

**Different profile than chem-Z**:
- chem-cov *loses* on small N (2, 3, 6) where chem-Z wins or ties.
- chem-cov *wins* on N=7-8 (the hard strata).
- Net +1.5pp from N=7/8 gains offsetting N=2/3/6 losses.

## Why cov underperforms Z

Hypotheses for why a "better" chemistry metric did worse:

1. **r_cov scale collapse on metals**: most transition metals + post-transition land at r_cov ∈ [1.1, 1.4] Å (Ti=1.4, Fe=1.4, Cu=1.35, Zn=1.35, …). With τ=0.5, all these get weight ≈1.0 → effectively chemistry-blind for metal-rich cells. Z had wider dynamic range (Ti=22 vs Zn=30, Δ=8 → weight 0.0007 at τ=4).
2. **The "wrong" mismatches matter**: V doesn't need a *chemically correct* prior — it needs a *discriminative* one. Raw Z, while crude, sharply distinguishes most pairs and acts as a strong signal for "different atoms = penalize". Cov dilutes this for the common metal pairs.
3. **N=2/3 regression suggests over-blending**: small cells with mixed compositions (e.g. ABX) need V to *separate* sub-modes by composition. cov tells V "all these heavy atoms look the same" → V averages over distinct structures → mode-blur.
4. **Hard-N wins are real but small**: cov's softer weighting may help where the model is sample-starved (N=7-8 baseline 8-10%) by not over-pruning candidates.

## Implication

For this V-kernel, **chemistry-as-discriminator beats chemistry-as-similarity**. Z-diff² is a useful nonsense-prior precisely *because* it's crude — it cleanly separates atom species. r_cov merges species that V should separate.

This is consistent with the chem-sweep finding (wide τ plateau): the *presence* of any per-pair weighting matters more than how chemically correct it is.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | cov ≥ baseline match@20 | ✅ +1.5pp |
| S2 | cov ≥ best chem-Z match@20 | ❌ chem-Z τ=2 still wins by 3.5pp |
| S3 | cov wins on at least one stratum where Z loses | ✅ N=7 (+10pp), N=8 (+8pp) |
| S4 | Smoke test verified cov differs from Z | ✅ Al/Ga test |

2 of 4 pass. S2 was the headline — cov did not beat Z.

## Open questions

1. **cov τ sweep** — τ=0.1 would sharpen cov (only same-Z get weight 1.0; Δ=0.1Å → 0.37). Might recover discriminative power. Unlikely to help (would converge to ~delta-on-Z which is chem=Z with very small τ).
2. **Hybrid cov+Z** — `w = w_Z · w_cov`. Sharp Z discriminator × soft cov tie-breaker. Untested.
3. **Period-group encoding** — use group/period number instead of raw radius. Same-group atoms get same code regardless of period. Could fix the metal-blur problem.

## Decision

**Drop cov-radius for the canonical PoC. Stick with chem-Z τ=2.** Don't pursue cov sweep — the Z metric is empirically better and simpler.

## Artifacts

- `~/tmp/dmf_ckpts/dmf_chemcov05_50k.pt` (ctor-gpu)
- `dmf_poc/cache/csp_chemcov05.json`, `chemcov05_raw.pt`
