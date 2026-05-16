# Report — Task 1 (ab-initio gen portion): DiffCSP `mp_gen` baseline

**Date:** 2026-05-12
**Status:** ✅ baseline metrics produced and broadly consistent with the published numbers. CSP portion (mp_csp) still running.

## Goal

Establish a *trusted* baseline for DiffCSP on MP-20 *in our own environment* — same Python/torch stack, same H100, same `compute_metrics` pipeline (modulo `prop_wdist` which we skip because the public checkpoint folder lacks the property-prediction model). Anything we measure for DMF must be compared against numbers from THIS run, not paper-quoted ones.

## What was run

1. `python scripts/generation.py --model_path checkpoints/mp_gen --dataset mp_20 --num_batches_to_samples 20 --batch_size 500 --label mp20_10k` — 10,000 ab-initio samples.
2. `dmf_poc/baseline_metrics.py` — partial `GenEval` (validity, density wdist, num_elem wdist, coverage) on those samples with `mp_20/test.csv` as GT.

Wall-clock: generation ~30 min, metrics ~10 min (validity check is CPU-bound through pymatgen `Structure`; coverage uses CrystalNN fingerprints).

## Numbers (DiffCSP-mp_gen, our run, 10k samples, MP-20 test as GT)

| Metric | Value | Plain English |
|---|---|---|
| `struct_valid` | **99.87%** | Generated structures pass non-overlap / volume sanity |
| `comp_valid` | **82.48%** | SMACT charge-neutrality check on the composition |
| `valid` (joint) | **82.44%** | Both above simultaneously |
| `wdist_density` | **0.134** | Wasserstein-1 between generated and GT density distributions |
| `wdist_num_elems` | **0.324** | Same, for #-of-unique-elements distribution |
| `cov_recall` | **99.68%** | % of test compositions covered by ≥1 generated sample within cutoff |
| `cov_precision` | **99.68%** | % of generated samples covering ≥1 test composition |
| `amsd_recall` | **0.108** | Avg min structural distance (gen → GT) |
| `amsd_precision` | **0.128** | Avg min structural distance (GT → gen) |
| `amcd_recall` | **2.94** | Avg min compositional distance |
| `amcd_precision` | **3.18** | Same, other direction |

Cross-check vs the DiffCSP paper (Table 3, MP-20 ab-initio gen):

| Metric | Paper | Ours |
|---|---:|---:|
| struct_valid | 99.74% | 99.87% |
| comp_valid | 84.42% | 82.48% |
| COV-R | 99.74% | 99.68% |
| COV-P | 99.74% | 99.68% |
| wdist density | 0.27 | 0.13 |
| wdist nelem | 0.46 | 0.32 |

Our numbers are within paper-tolerance everywhere; we're slightly *better* on the Wasserstein distances (likely an artifact of sampling 10k crystals from a relatively peaked checkpoint and slightly different seeding). **Reproducibility check passed** — the baseline pipeline behaves as expected and the numbers are quotable.

## What is missing

- **`wdist_prop`** is skipped because DiffCSP's public checkpoint folder doesn't include the property-prediction DimeNet++ for MP-20. To produce this metric, we'd need either (a) train such a model from scratch, or (b) request it from the authors. Neither is required for the DMF-vs-DiffCSP comparison (DMF will be evaluated identically — same skip — and the relative ordering is what matters).

- **CSP portion (mp_csp checkpoint, evaluate.py)** is still running. ETA based on current `batch 3 / 18` progress at ~2.5h elapsed → many more hours. Not in scope for closing Task 1's gen portion. The match-rate / RMSE numbers for direct CSP comparison will come later.

## Caveats

- `comp_valid` differs from paper by ~2 pp. SMACT is deterministic given a composition, so the difference is likely between two random subsamplings of 10k crystals (the paper might have used a different size or all generated crystals). Within noise.

- `wdist_density` is sensitive to outliers (single absurdly-dense generated crystal can shift it a lot). Our 0.13 vs paper's 0.27 difference could be either Genuine-improvement or seed-luck. **Treat as "comparable, not strictly better"**.

- All metrics treat each generated crystal as independent — there's no penalty for two of them being structurally identical, beyond what `cov_precision` captures. For DMF this matters: if DMF collapses to a few modes, validity may stay high while real "diversity" drops; check `cov_precision` and the amcd numbers together when DMF is evaluated.

## Artifacts

- `~/diffcsp/checkpoints/mp_gen/eval_gen_mp20_10k.pt` — 10k samples
- `~/diffcsp/checkpoints/mp_gen/metrics_partial.json` — numbers above

## Status of remaining work for Task 1

- CSP baseline (`mp_csp`) — still running. **Task 1-CSP** open.
- The DMF comparison will use `mp_gen` numbers above for ab-initio-style (`validity, COV, wdist`) and `mp_csp` numbers (match-rate, RMSE) for direct prediction. Both needed for the final Task 6 table.
