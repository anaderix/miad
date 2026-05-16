# Report — Task 6a (prep): comparison infrastructure

**Date:** 2026-05-12
**Status:** ✅ comparison script written and sanity-validated. Ready for the moment DMF gen metrics finish.

## Goal

Write a parameterized `compare_metrics.py` that ingests two JSONs (e.g. the DiffCSP-mp_gen baseline and the DMF-50k metrics, when ready) and emits a side-by-side markdown table with Δ and per-metric "winner" annotations. Doing this **before** the DMF metrics finish means no critical-path code is written under time pressure once the numbers land — we just `python compare_metrics.py --left ... --right ...` and copy the output.

## Implementation

- `dmf_poc/compare_metrics.py` — ~70 LoC.
- Hard-codes the "direction" of each metric (higher-better for validity/coverage, lower-better for wdist/amsd/amcd). Unknown keys are tolerated but not annotated with a winner.
- Output: a markdown table with columns `Left | Right | Δ(R−L) | Δ% | Better`.

## Sanity check

Ran `compare_metrics.py` with `--left == --right` on the DiffCSP-mp_gen JSON. Result:

| Metric | DiffCSP-A | DiffCSP-B | Δ (R−L) | Δ% | Better |
|---|---:|---:|---:|---:|:---:|
| comp_valid | 0.8248 | 0.8248 | 0 | 0% | ≈ |
| struct_valid | 0.9987 | 0.9987 | 0 | 0% | ≈ |
| valid | 0.8244 | 0.8244 | 0 | 0% | ≈ |
| wdist_density | 0.1336 | 0.1336 | 0 | 0% | ≈ |
| wdist_num_elems | 0.3239 | 0.3239 | 0 | 0% | ≈ |
| cov_recall | 0.9968 | 0.9968 | 0 | 0% | ≈ |
| cov_precision | 0.9968 | 0.9968 | 0 | 0% | ≈ |
| amsd_recall | 0.1078 | 0.1078 | 0 | 0% | ≈ |
| amsd_precision | 0.1282 | 0.1282 | 0 | 0% | ≈ |
| amcd_recall | 2.9356 | 2.9356 | 0 | 0% | ≈ |
| amcd_precision | 3.1773 | 3.1773 | 0 | 0% | ≈ |

All deltas zero, all rows annotated `≈`. Infrastructure works.

## Current state of dependencies

- **DiffCSP-mp_gen metrics**: ready (`~/diffcsp/checkpoints/mp_gen/metrics_partial.json`).
- **DMF-50k gen metrics**: running. 27 min elapsed, still in validity/coverage stage; large volume of `He electronegativity` warnings suggests DMF is producing noble-gas compositions which pymatgen valid-checks complain about. Will finish within next 10 min.
- **DiffCSP-mp_csp CSP eval**: still running. ~30% done (batch 5/18). Many more hours to go.

## Plan for the closing comparison (Task 6 proper)

When DMF metrics JSON lands:

```bash
python ~/miad/dmf_poc/compare_metrics.py \
  --left  ~/diffcsp/checkpoints/mp_gen/metrics_partial.json --left-label "DiffCSP-mp_gen (1000 NFE)" \
  --right ~/miad/dmf_poc/cache/metrics_partial.json         --right-label "DMF-50k (1 NFE)" \
  --out ~/miad/dmf_poc/report_task_6_table.md
```

Then write `report_task_6.md` with:
1. The table (above) plus interpretation per row
2. Throughput / NFE comparison line: DMF 220× faster at inference
3. Per-criterion success/failure against the criteria in the original plan (Match-rate via gen-cov as the proxy; Diversity via cov_precision and amcd; Throughput is a clear win)
4. Honest verdict on whether this PoC supports continuing toward "MP-20 ab-initio gen with DMF as a viable baseline" or whether something fundamental broke

## Artifacts

- `dmf_poc/compare_metrics.py` — comparison script
- `dmf_poc/report_task_6_table.md` (will be created when comparison runs)
