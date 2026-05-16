# Report — Task 4a: Synthetic DMF training loop

**Date:** 2026-05-12
**Status:** ✅ 5/5 success criteria met

## Goal

Validate the **mechanics** of the DMF training loop end-to-end — `CSPNetDMF` (Task 3) + `compute_V` (Task 2) + friction-scaled MSE — independent of real MP-20 data loading (left for Task 4b). If the loop can drive a network to a known synthetic target distribution, the integration of the three pieces is correct and the next step (real data) is safe to attempt.

## Setup

- Fixed single composition: 4-atom crystals, all Si (Z=14).
- Synthetic GT distribution:
  - Lattices `L ~ 5·I + N(0, 0.3)` (trace ≈ 15)
  - Frac `F ~ N(0.5, 0.05) mod 1` (mean ≈ 0.5)
- Training: 2000 iterations, batch=64, Adam lr=5e-4, friction γ_t linear 0→1, grad-clip 5.0.
- Loss = `MSE(L̂, target_L) + λ·MSE(torus_diff(F̂, target_F))`, λ=1.0.

## Success criteria & results

| # | Criterion | Result |
|---|---|---|
| S1 | Smoothed loss at end < 50% of loss at start | ✅ `0.183 → 0.0002` |
| S2 | Convergence: \|trace(L̂_init) − 15\| → \|trace(L̂_final) − 15\| shrinks by ≥75% | ✅ `13.93 → 0.42` (97% reduction) |
| S3 | Final `trace(L̂)` within ±5 of target trace=15 | ✅ `14.58` |
| S4 | Final `mean(F̂)` within ±0.15 (torus) of target=0.5 | ✅ `0.480` |
| S5 | No NaN/Inf throughout training | ✅ |

Convergence trajectory (selected iters):

| iter | loss | mean trace(L̂) | mean(F̂) | γ |
|---:|---:|---:|---:|---:|
| 0 | 0.193 | 1.07 | 0.506 | 0.000 |
| 200 | 0.156 | 14.26 | 0.378 | 0.100 |
| 800 | 0.070 | 14.66 | 0.452 | 0.400 |
| 1400 | 0.017 | 14.50 | 0.470 | 0.700 |
| 1999 | 0.00000 | 14.58 | 0.480 | 1.000 |

Within 200 iters the lattice mean trace jumped from random-init `1.07` to ~14.3 — most of the lattice-mode discovery happens in the first 10% of training. Frac mean converges slower (drives to target by iter ~1000), reflecting a tighter prior on the torus.

## Design flaw caught during testing

First version of S2 asserted "drift norm decreases by ≥30%". This **fails by construction** because `compute_V` (Task 2) normalizes each per-sample V to unit magnitude — the batch-averaged drift norm stays ~constant regardless of distribution match. The drift's *direction* is the signal; its *magnitude* is normalized away.

Replaced S2 with the right diagnostic: distance from `target_trace` shrinks by ≥75% from iter 0 to final. This directly tests "did the model learn to point at the target distribution".

Documented in `findings.md` so future training-loop diagnostics use the right metric.

## Caveats / what this does NOT prove

This synthetic test demonstrates that the **mechanics** work — gradient flow, friction schedule, MSE on the right modalities, no NaN. It does **not** prove that DMF will work on real `mp_20`:

- Synthetic target has 1 mode; mp_20 has thousands of distinct compositions × multiple polymorphs each.
- Synthetic GT is re-sampled every step (infinite-data regime); real data has 27k crystals total.
- Lattice scale here is `≈O(5)`; real mp_20 lattices span 2–15 Å with anisotropy. Temperature schedule for V_lat may need tuning.
- 4-atom crystals at one composition — graph-structural complexity is artificially low.

These are exactly the things Task 4b (real data loader + training on mp_20) will stress-test.

## Validation environment

- `vm-gpu-2` (H100), Python 3.11.15, torch 2.4.1+cu121
- Wall-clock: ~75 sec for 2000 iters at B=64
- Run: `cd ~/miad/dmf_poc && ../.venv/bin/python train_dmf_synthetic.py`

## Next: Task 4b — real MP-20 data loader

Now unblocked. Needs:
- CSV → list of (lattice, frac, atom_types, num_atoms) tensors
- Compose-stratified batching: B crystals per batch sharing the same `(types, N)` (or close enough)
- Same training loop as 4a, but with real targets
- Held-out 100 compositions for periodic match-rate check

Open question for 4b: how strict to be on "same composition per batch"? Strict → batches limited by composition frequency; lax (e.g. same `N` only) → V kernel becomes noisier because atoms with different `Z` shouldn't be compared elementwise.
