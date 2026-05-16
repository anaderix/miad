# Report — Task 10a: Inference-only iterative DMF (negative result)

**Date:** 2026-05-12
**Status:** ⚠️ Inference-only K-step iteration makes default-trained DMF *worse*, not better. **Fourth consecutive negative experiment.** Confirms what was visible from the lattice attractor diagnostic: the model's `lattice_out_matrix` has eigenvalues < 1, so feeding output as input iteratively contracts the lattice. Without retraining for iterative use, the trick fails.

## Hypothesis under test

From Task 9's findings:
> Iterative DMF (NFE=2-5) is likely the biggest quality win. Drop the strict "one-shot" constraint; do 2-5 forward passes with decaying friction.

Inference-only first test: just feed the model's output back as `(lat_in, frac_in)` for the next step. No retraining. If this helps, retrain for iterative use (Task 10b). If not, skip.

## Diagnostic: lattice attractor

K-sweep on the default-trained ckpt:

| K | lattice length range | angle range |
|---:|---|---|
| 1 (one-shot) | 1.96–8.27 Å | 47.98–153.24° |
| 2 | 1.50–7.95 Å | 72.93–144.69° |
| 5 | 0.80–7.58 Å | 66.48–105.27° |
| 10 | 0.52–7.59 Å | 57.93–108.75° |

The lattice min shrinks monotonically (1.96 → 0.52). Angles converge toward 90°. Output converges to a small near-cubic cell — a contraction fixed point of `L_{k+1} = lattice_out_matrix @ L_k`.

The reason: `ip=True` in CSPNet makes the output multiplicatively conditioned on input lattice, and the learned `lattice_out_matrix` has eigenvalue magnitudes < 1 (because each training step was one-shot, so the model produces compressed outputs from larger noise priors). Iterating compresses further with each step.

## Results (K=2 metrics, the least-degenerate K>1)

| Metric | K=1 | K=2 | Δ% | Direction |
|---|---:|---:|---:|---|
| `wdist_density` (↓) | 4.376 | 6.419 | +47% | ❌ |
| `struct_valid` (↑) | 0.364 | 0.313 | −14% | ❌ |
| `valid` (↑) | 0.253 | 0.227 | −10% | ❌ |
| `wdist_num_elems` (↓) | 0.234 | 0.261 | +12% | ❌ |
| `cov_recall` (↑) | 0.370 | 0.381 | +3% | weak ✅ |
| `cov_precision` (↑) | 0.882 | 0.882 | = | = |
| `amsd_recall` (↓) | 0.560 | 0.541 | −4% | weak ✅ |
| `amsd_precision` (↓) | 0.193 | 0.192 | ≈ | = |
| `comp_valid` (↑) | 0.630 | 0.630 | = | = |

Net: validity drops 14%, density gets dramatically worse (+47%), with marginal gains on cov_recall and amsd_recall. The Pareto trade-off pattern from Tasks 7-9 reappears.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | At least 3 metrics improve over K=1 | ❌ only 2 mild wins |
| S2 | Density wdist improves | ❌ much worse |
| S3 | Validity maintained within 10% of K=1 | ❌ dropped 14% |
| S4 | Lattice attractor doesn't dominate (output remains comparable to K=1 spread) | ❌ contracts |

0 of 4 pass at the practical level. Hypothesis only true if model is **retrained** for iterative inference.

## Summary of Tasks 7-10a (four negative results)

| Task | Knob changed | density wdist | struct_valid | net |
|---|---|---:|---:|---|
| 7 (inference-only wider prior) | lat_in mean=10, std=2 | 0.91 ✅ | 0.214 ❌ | trade |
| 8 (retrain wider prior) | trained matched wider prior | 6.06 ❌ | 0.278 ❌ | both worse |
| 9 (retrain higher V_L temp) | temps_L=[5,10,20] | 4.27 ≈ | 0.199 ❌ | density flat, validity worse |
| 10a (inference K=2) | iterate K=2 forwards | 6.42 ❌ | 0.313 ❌ | both worse |

DMF on MP-20 in this configuration sits on a **validity-density Pareto frontier**. All simple knobs slide along it; none break out. The structural ceiling is the kernel-mean-shift + one-shot + same-N-batching combination.

## Honest verdict on the iteration arc

The original plan (Tasks 1-6) closed with a clear verdict: DMF doesn't match DiffCSP on gen metrics, but offers 220× throughput and is competitive for N=2 compositions (Task 5b). The four exploratory tasks (7-10a) that followed all aimed to improve the gen metrics through hyperparameter tuning and all failed. **This is signal that further improvement requires architectural changes**, not more tuning.

The proper next experiments would be:
1. **Task 10b — iterative training**: retrain with K-step rollout in the loss (model learns to refine). Requires modifying `train_dmf_mp20.py` to loop forward K times per batch. ~1.5h cycle.
2. **Task 11 — FiLM composition conditioning**: per-CSPLayer modulation by atom_types. Code change in `cspnet_dmf.py`. ~1h to implement + 1.5h cycle.
3. **Task 12 — per-atom z**: each atom gets its own latent, not just per-graph. Code change in cspnet_dmf forward + train. ~1h cycle.
4. **Task 13 — V with repulsion**: explicit "spread" term in `compute_V_*`. Code change to v_drift.py. ~1h cycle.

Each is a real research bet, not guaranteed. Recommend pausing here unless the user wants to commit to one.

## Artifacts

- `~/miad/dmf_poc/cache/iter{2,5,10}/eval_gen.pt` — outputs at K=2, 5, 10
- `~/miad/dmf_poc/cache/iter2/metrics_partial.json` — metrics for K=2
- `evaluate_dmf.py` now supports `--n_steps K` flag
