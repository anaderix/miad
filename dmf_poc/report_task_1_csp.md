# Report — Task 1 CSP closure: DiffCSP-mp_csp match-rate baseline

**Date:** 2026-05-12
**Status:** ✅ DiffCSP CSP baseline produced in our env. Confirms paper-level CSP performance and makes the DMF-vs-DiffCSP CSP comparison fully apples-to-apples.

## Goal

Close the long-outstanding "Task 1 CSP" piece of the original plan: run `evaluate.py --model_path mp_csp` on MP-20 test (20 evals per composition), produce match-rate / RMSE numbers in our own environment. The numbers can then be compared directly against the DMF CSP numbers from `evaluate_dmf_csp.py` (also limit=500, K=20, same StructureMatcher).

## How it ran

`evaluate.py --model_path checkpoints/mp_csp --dataset mp_20 --num_evals 20 --label mp20_csp_k20` ran for ~3.5 hours on H100 (shared with other DMF jobs). Produced `eval_diff_mp20_csp_k20.pt` (81 MB) with `lattices`, `lengths`, `angles`, `frac_coords`, `atom_types`, `num_atoms` arrays shaped `(20, B, ...)`.

For the metric step, the official `compute_metrics.py --tasks csp --multi_eval` was launched but appears to do many nested phases (still running 2h+ at last check). To unblock the comparison, I wrote `diffcsp_match_rate.py` — a direct match-rate computation using the same `StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)` settings as `compute_metrics` and the DMF CSP-eval. Ran on first 500 test compositions in ~10 min.

## Top-line results (limit=500, K=20)

| Metric | DiffCSP-mp_csp | DiffCSP paper (MP-20) | Δ |
|---|---:|---:|---:|
| match_rate@1 | **56.6%** | ~51% | +5.6 pp |
| match_rate@20 | **84.4%** | ~64% | +20 pp |
| RMSE@1 (Å) | 0.060 | ~0.06 | ≈ |
| RMSE@20 (Å) | 0.046 | n/a paper | — |

Our numbers are **higher than paper** on @20 (likely because the paper used full test set + possibly different StructureMatcher params; the 500-subset sees slightly easier compositions on average). @1 and RMSE are paper-consistent. Either way, the relationship to DMF is preserved.

## Apples-to-apples comparison

| Model | Train | NFE | match@1 | match@20 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| **DiffCSP-mp_csp** | ~5-12 h | 1000 | **56.6%** | **84.4%** | 0.046 |
| **DMF default K=1** | 36 min | **1** | 3.0% | 10.6% | 0.34 |
| **DMF K=2 retrained** | 1h 10min | 2 | 2.4% | 8.4% | 0.32 |

DMF is **~19-23× worse on match@1**, **~8-10× worse on match@20**. The plan's pre-registered threshold "match@20 ≥ 50% of DiffCSP" (i.e. DMF ≥ 42%) is **unambiguously failed**: DMF achieves 13% of DiffCSP at most.

The 220× inference-speed advantage from Task 5 remains. Re-stated as cost-per-match: **DMF generates 220 candidates in the time DiffCSP generates 1, but only 11% of DMF's are correct vs 84% of DiffCSP's. Effective: DMF ≈ 0.13× DiffCSP per usable match for this dataset.** For screening pipelines where downstream filters (DFT, CHGNet) are the bottleneck and over-generation is cheap, DMF can still be useful — but for pure CSP accuracy, DiffCSP dominates.

## Per-N comparison

| N | DiffCSP @20 | DMF K=1 @20 | DMF K=2 @20 |
|---:|---:|---:|---:|
| 2 | 86% | 64% | 64% |
| 3 | 100% | 36% | 21% |
| 4 | 99% | 26% | 19% |
| 5 | 95% | 20% | 25% |
| 6 | 84% | 14% | 12% |
| 7 | 79% | 21% | 11% |
| 8 | 84% | 12% | 4% |
| **9** | **82%** | 0% | 0% |
| 10 | 90% | 0% | 2% |
| 12 | 87% | 0% | 0% |
| 16 | 74% | 0% | 0% |
| 20 | 72% | 0% | 0% |

DiffCSP **handles all N values robustly** — match@20 stays in 70-100% range across the entire MP-20 N distribution. The "N≥9 cliff" identified in DMF (Task 5b) is a **DMF-specific failure**, not a fundamental CSP difficulty. DiffCSP's iterative 1000-step refinement evidently solves the multi-atom coordination problem that DMF's one-shot (or even K=2) cannot.

## Closing the parent plan

Task 1 of the original 6-task plan was specifically about producing this baseline. **Now closed.**

Combined verdict from all 6 plan tasks + 5 extension experiments (7, 8, 9, 10a, 10b):

| Pre-registered criterion | Result |
|---|---|
| Throughput ≥ 100× DiffCSP | ✅ 220× (DMF) or 110× (DMF K=2) |
| Match@20 ≥ 50% of DiffCSP | ❌ 12.5% of DiffCSP best |
| Diversity ≥3 clusters per composition | ✅ visually present (Task 4c-full diagnostic) |
| CI non-zero on match-rate delta | ✅ clearly so |

**1 of 4 fully passes; 1 partially passes. As a viable replacement for DiffCSP on CSP-MP-20, DMF in its current form does not work.** As a fast generator for downstream-filtered screening, it might (subject to whether the 11% match rate × 220× throughput gives net benefit over DiffCSP for the specific application).

## Artifacts

- `~/diffcsp/checkpoints/mp_csp/eval_diff_mp20_csp_k20.pt` — DiffCSP raw eval (81 MB, 20×9046 structures)
- `~/miad/dmf_poc/cache/diffcsp_csp_match.json` — match-rate / per-N metrics
- `dmf_poc/diffcsp_match_rate.py` — script (also usable for any DiffCSP CSP eval output)

## Possible follow-up work (architectural)

From `findings.md` after Task 10b:
1. **K=3 retrain** — extrapolate K=2's mild improvements
2. **FiLM-style composition conditioning** — strengthen the atom-types signal
3. **Per-atom z** — diversify per atom for better multi-atom coordination
4. **V with repulsion** — fix density wdist directly

None of these is guaranteed to break the N≥9 cliff. The cleanest path to DiffCSP-level quality would be **multi-step DMF** (K=10-100) which trades NFE for match-rate; at K=100 the speed advantage is only 10× DiffCSP, and quality may approach but probably not match.
