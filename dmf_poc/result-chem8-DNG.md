# Result — TASK chem8-DNG: DNG-lite evaluation (V + U + Nv, no S)

**Date:** 2026-05-18
**Status:** ⚠️ Diagnostic-only. V=100%, U=99.9%, Nv=98.6% — **suspiciously high**, because without Stability filtering these metrics measure "structures are non-trivially-different random samples", which is satisfied even by pure-noise output. The real conclusion: **U and Nv are uninformative without S**. Need CHGNet/eq-V2 stability filter to make this a real DNG benchmark.

## Setup

- `dmf_chem8_50k.pt` (canonical chem8)
- 1000 compositions sampled from MP-20-train empirical (N, sorted_Z) distribution
- One structure per composition (no K-replicas; DNG measures per-sample quality)
- Metrics:
  - **Validity (V)**: pymatgen builds the structure (no parse failure)
  - **Uniqueness | Valid (U)**: per composition, structure not dup of earlier-same-composition gen via StructureMatcher
  - **Novelty | Unique (Nv)**: structure does not match any MP-20-train structure with same composition
- **Stability (S) skipped** — requires CHGNet pretrained + MP phase-diagram pickle (`2023-02-07-ppd-mp.pkl`, ~hundreds MB) which are not on ctor-gpu. See backlog.

## Top-line numbers

| Metric | chem8 |
|---|---:|
| n_gen | 1000 |
| Validity (V) | 100.0% |
| Uniqueness \| Valid (U) | 99.9% |
| Novelty \| Unique (Nv) | 98.6% |
| Combined V·U·Nv / n_gen | 98.5% |

## Per-N: novelty fully saturated for N≥5

| N | n | V | U | **Nv** |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 |
| 2 | 26 | 26 | 25 | **18** |
| 3 | 24 | 24 | 24 | **21** |
| 4 | 145 | 145 | 145 | **141** |
| 5 | 60 | 60 | 60 | **60** |
| 6 | 72 | 72 | 72 | **72** |
| 7 | 21 | 21 | 21 | **21** |
| 8 | 75 | 75 | 75 | **75** |
| 9-20 | 666 | 666 | 666 | **666** |

**Critical observation**: for N≥5, **every** generated structure is flagged "novel". Only N=2,3,4 (small cells where chem8 actually lands close to GT via match-rate evaluation) show any train-overlap (8 dups for N=2, 3 for N=3, 4 for N=4).

This is **not a signal of strong generative novelty**. It's a signal that for N≥5 the chem8 outputs are far enough from train structures (in lattice/coord space) that StructureMatcher with `ltol=0.3, stol=0.5` rejects them as matches — but these structures haven't been filtered for stability, so they could just be physically nonsense.

## Why this happens

The kernel-mean-shift V drift, even with chem8, doesn't enforce any physical constraint (no energy, no coordination, no minimum-distance term). For N≥5 the model:
- Composition input is correctly placed (one-hot)
- Lattice is "in the right ballpark" (5 Å × cube perturbed)
- Frac coords are drifted toward train neighborhood

But "drifted toward" ≠ "matches". With friction γ→1 and only one sample per composition (vs K=20 in match-rate eval), the typical output is a *plausible-looking-but-not-exact* structure. StructureMatcher accepts a wide tolerance (~30% length) but not so wide that random plausible inputs match.

**Cross-validation with match@K results**: at K=20, N=5 had match-rate 38% (chem8 50k). Now at K=1 it's effectively 0% on this train pool. That's consistent — 1 sample rarely hits the small per-composition target.

## What V/U/Nv numbers actually mean here

- **V=100%**: model output is a valid pymatgen Structure (positive lattice det, finite frac coords). Trivially satisfied by current implementation; doesn't filter unphysical configs.
- **U=99.9%**: only 1 of 1000 was a duplicate of another gen in same composition. Reflects diversity *of noise*, not diversity *of valid crystals*.
- **Nv=98.6%**: same logic — structures don't match train, but that's because they're far from train, not because they're new valid crystals.

The **V·U·Nv = 98.5%** combined metric reported by many papers is essentially meaningless without S as a precondition. Stable structures are the prerequisite for asking "is this unique and novel".

## Comparison to MiAD paper

MiAD-paper Table 2 reports for MP-20:
- DiffCSP: S-rate ≈ 50%, S·U·Nv ≈ 7-8%
- MiAD: S-rate ≈ 60%, S·U·Nv ≈ 11-12%

Our V/U/Nv numbers without S can't be compared to these. We'd need CHGNet to filter for stability first, then compute U/Nv only on stable structures.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | Pipeline runs end-to-end | ✅ 1000 samples generated, all metrics computed |
| S2 | V > 50% (basic sanity) | ✅ 100% |
| S3 | U and Nv interpretable as DNG metrics | ❌ — not without S |
| S4 | Direct comparison with DiffCSP/MiAD numbers | ❌ — S required |

2 of 4 pass. The infrastructure works; the metric is gated on S.

## What we learned (despite null result)

1. **Composition sampler from empirical (N, Z-tuple) prior works**. MP-20 train has surprisingly few unique compositions (~10k unique out of 27k train) — there's substantial duplication.
2. **N-distribution after sampling** matches train: bias toward N=4 (145), N=10 (91), N=12 (99), N=20 (89). Same skewed long-tail.
3. **The "easy strata" (N=2,3,4) are the only place where chem8 produces train-matchable structures at K=1**. This confirms what we saw in match-rate: chem8 reaches train neighborhood for small N, less so for large N.

## Next step

The clear blocker is **Stability**. Plan:
1. Install `chgnet` on ctor-gpu venv: `pip install chgnet`.
2. Adapt or reuse `lib/prerelaxations/prerelax_chgnet.py`.
3. Relax each of 1000 gen structures (1500 steps × ~10 ms per step per structure ≈ 4 hours; can parallelize across GPU MIG).
4. Compute E_hull (need MP phase diagram pickle — separately download or build from `pymatgen.ext.matproj`).
5. Re-compute U and Nv only over stable subset.
6. Report S, S·U, S·U·Nv comparable to MiAD paper.

That's ~1 day of setup + 4-8 hours of compute. Worth doing as the next priority task.

## Artifacts

- `dmf_poc/evaluate_dmf_dng.py` (new)
- `dmf_poc/cache/dng_chem8.json`
