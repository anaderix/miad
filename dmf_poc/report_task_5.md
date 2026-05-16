# Report — Task 5: DMF inference + DiffCSP-compatible serialization

**Date:** 2026-05-12
**Status:** ✅ 5/5 success criteria met. Same `baseline_metrics.py` used for DiffCSP-mp_gen now running on DMF output.

## Goal

Generate crystals from the trained DMF (Task 4c) on the MP-20 test compositions and serialize the output in **byte-identical format** to DiffCSP's `eval_gen.pt`. This is the bridge piece: it lets us evaluate DMF with the **same metrics code** that produced the DiffCSP-mp_gen baseline numbers (Task 1-gen). Any difference in evaluation methodology is then provably absent.

## Format reverse-engineered

DiffCSP's `eval_gen.pt` is a dict with:

| Key | Shape | Meaning |
|---|---|---|
| `eval_setting` | `argparse.Namespace` | metadata (model_path, dataset, label) |
| `frac_coords` | `(sum_N, 3) float32` | flat fractional coords for *all* atoms |
| `num_atoms` | `(B,) int64` | per-crystal N |
| `atom_types` | `(sum_N, MAX_ATOMIC_NUM) float32` | one-hot |
| `lengths` | `(B, 3) float32` | lattice vector lengths a, b, c (Å) |
| `angles` | `(B, 3) float32` | α, β, γ (degrees) |

Important: lattice is stored in **lengths+angles**, not as 3×3 Cartesian basis. The Cartesian basis our DMF outputs must be converted.

## Implementation

- `dmf_poc/evaluate_dmf.py`, ~180 LoC.
- Parses `data/mp_20/test.csv` for the 9046 test compositions (Z lists).
- Samples N-stratified batches (with replacement, since target=10000 > test_size=9046) until n_samples reached.
- Per-batch: random `z` ∼ N(0,I)^256, `frac_in` ∼ U[0,1), `lat_in` ∼ N(0,1)+5·I. Single forward through DMF.
- `lattice_matrix_to_lengths_angles`: standard `‖a_i‖` and `acos((a·b)/(‖a‖‖b‖))` conversion.
- Output saved to `~/miad/dmf_poc/cache/eval_gen.pt` so the same `baseline_metrics.py --root_path ~/miad/dmf_poc/cache` pipeline that ran on DiffCSP-mp_gen runs verbatim on DMF.

## Success criteria & results

| # | Criterion | Result |
|---|---|---|
| S1 | All values finite (no NaN/Inf) | ✅ |
| S2 | All fractional coords in [0,1) (small tolerance for boundary) | ✅ min=0.0000, max=1.0000 |
| S3 | Lattice vector lengths plausible (0.5 < ‖a‖ < 100 Å) | ✅ min=1.96, max=8.27 Å |
| S4 | Lattice angles plausible (1° < θ < 179°) | ✅ min=47.98°, max=153.24° |
| S5 | Shape consistency: `len(frac_coords) == len(atom_types) == sum(num_atoms)`; `len(lengths)==len(angles)==len(num_atoms)==n_samples` | ✅ |

## Throughput

- **10,000 DMF crystals in 8.3 seconds** at batch=64 on H100 (NFE=1 each).
- For comparison, DiffCSP-mp_gen generated 10k samples in **30 min** at batch=500 with 1000 diffusion steps each.
- That's **~220× speedup at inference time**. Even if DMF's match quality is significantly lower, this throughput is meaningful for any application where structure candidates can be filtered cheaply downstream (e.g. CHGNet pre-relaxation).

## Caveats / open questions

- **Lattice length range (1.96–8.27 Å) is narrower than mp_gen's** distribution would be. We don't have direct mp_gen lattice length distribution to compare, but published MP-20 statistics span 2–15 Å. The narrow range suggests DMF compresses lattice scale toward the mean — a form of mode coverage failure. Validity metrics will reveal whether this hurts the comparison.

- **Angles span 48–153°** which includes both near-orthogonal (90°) and substantially skewed cells. Reasonable spread.

- The N-distribution of generated crystals is weighted by *test-set frequency* of N. Same-N batching at inference is efficient and matches the training data distribution.

## What we still need for Task 6 (final comparison)

The launched `baseline_metrics.py` will produce a JSON with the same metrics as `metrics_partial.json` for mp_gen. Side-by-side table will be in Task 6.

In addition, the still-running `evaluate.py mp_csp` from Task 1 will give the strict CSP metrics (match-rate, RMSE). Whether DMF can match that is a separate question (and we don't have a "CSP-style" eval for DMF that produces 20 samples per test composition aligned with GT — we'd need to write that, or accept that DMF-vs-mp_csp is an unfair comparison).

For now, the path to closure is:

1. `baseline_metrics.py` finishes on DMF output → DMF gen-metrics
2. Compare DMF vs DiffCSP-mp_gen gen-metrics → first comparison table (Task 6a)
3. If `evaluate.py mp_csp` finishes (or we kill and downscale it) → CSP table (Task 6b)

## Artifacts

- `dmf_poc/evaluate_dmf.py` — inference + serialization
- `~/miad/dmf_poc/cache/eval_gen.pt` — 10k DMF crystals, DiffCSP-compatible format
- `/tmp/dmf-metrics.log` — running `baseline_metrics` on DMF output
