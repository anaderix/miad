# Report — Task 8: retrain with wider lattice prior (negative result)

**Date:** 2026-05-12
**Status:** ⚠️ Hypothesis from Task 7 falsified. Retraining with wider prior **does NOT** improve density-wdist; it makes most metrics worse. Reveals a deeper finding: the network self-normalizes lattice scale during training, ignoring most of the prior signal. The lattice distribution is fundamentally constrained by the **V kernel + target batch**, not by the input noise prior.

## Hypothesis under test

From Task 7's report:
> Retrain DMF with the wider prior baked into training. Expected: `wdist_density` ≈ 1, `struct_valid` ≈ 0.4-0.6 (recovered from 0.21), `valid` ≈ 0.3-0.4.

Logic: Task 7 showed inference-only prior swap improves density (4.38 → 0.91) but breaks validity. So the natural prediction: train with the wider prior, model adapts to produce physically-valid wider lattices.

## Setup

- Same `CSPNetDMF` architecture, same 50k iters, same B=64, same MP-20 train
- ONLY change: `--lat_prior_mean 10.0 --lat_prior_std 2.0` during training
- Same matched `--lat_prior_mean 10.0 --lat_prior_std 2.0` at inference

## Actual results

| Metric | DMF-default | DMF-wider-retrain | Δ% vs default | Δ% vs DiffCSP |
|---|---:|---:|---:|---:|
| `wdist_density` (↓) | 4.376 | **6.058** | +38% WORSE | +4400% |
| `wdist_num_elems` (↓) | 0.234 | 0.299 | +28% WORSE | -8% |
| `struct_valid` (↑) | 0.364 | 0.278 | -23% | -72% |
| `valid` (↑) | 0.253 | 0.218 | -14% | -74% |
| `cov_recall` (↑) | 0.370 | 0.389 | +5% (mild win) | -61% |
| `cov_precision` (↑) | 0.882 | 0.882 | = | -12% |
| `comp_valid` (↑) | 0.630 | 0.630 | = | -24% |
| `amsd_recall` (↓) | 0.560 | 0.512 | -9% (mild win) | +375% |
| `amsd_precision` (↓) | 0.193 | 0.208 | +8% | +62% |
| `amcd_*` | — | — | unchanged | unchanged |

**Pattern:** the retrained model is mildly better at coverage/distance-from-test (cov_recall, amsd_recall slightly better) but markedly worse at validity and density. Net: the trade is unfavorable.

## Diagnostic: model output range under matched-prior

At inference with `lat_prior_mean=10, std=2`:
- **default-trained** (eval-only mismatch): lengths 1.18–11.79 Å (Task 7)
- **wider-retrain** (matched): lengths 2.33–8.00 Å (this run, narrower!)

The wider-retrain model **compensates** for the wider prior during training by squeezing `lattice_out_matrix` to give narrower outputs. The actual lattice range it produces is similar to the default-trained model's range under default prior.

## Why the hypothesis was wrong

The mental model from Task 7 was: "prior scales output directly via `L̂ = lattice_out_matrix @ lat_in`, so wider prior → wider outputs". This is true for a **fixed** `lattice_out_matrix`. But during training, `lattice_out_matrix` is learned — and the V kernel pulls outputs toward the MP-20 lattice distribution regardless of prior. So the network learned a smaller `lattice_out_matrix` to undo the larger prior.

In other words: **the network's lattice distribution is pinned by V + the target batch distribution**, with the prior playing only a noise-injection role. Changing the prior cannot widen the lattice mode if the V kernel keeps pulling toward narrow MP-20 modes.

## Deeper finding (worth recording)

This is actually a **structural observation about kernel-mean-shift V training**:

The V kernel `V_L(L_gen, L_pos_batch)` for any sample is the mean-shift toward the batch centroid. With B=64 random MP-20 batches, the batch lattice distribution **is itself a sample from the dataset distribution** — width ≈ MP-20 width. The mean-shift target therefore has bounded variance proportional to MP-20's lattice variance.

The model output variance can only match (or be smaller than) the variance of its target. Since V's target inherits MP-20's variance, the output is bounded by it. Increasing input prior variance does not help because the output is *not free to wander*; it's pinned to V's target.

**Implication for any future DMF-on-crystals work:** to widen output diversity, you have to widen V's target, not the input prior. Options:
1. **Larger batches** — better sample MP-20's tails per kernel computation.
2. **Higher temperature τ** in `compute_V_*` — softer kernel = drifts toward broader average = wider target → wider output.
3. **Modified V** that encourages diversity directly (e.g. with a repulsion term).
4. **Train longer** so the model can shape `lattice_out_matrix` to better reflect data variance.

Option 2 (higher τ) is testable in ~1h.

## Success criteria for Task 8

| # | Criterion | Result |
|---|---|---|
| S1 | `wdist_density` <  default's 4.4 (matched-prior retrain) | ❌ went to 6.06 |
| S2 | `struct_valid` >  default's 0.36 | ❌ dropped to 0.28 |
| S3 | At least 2 metrics improved over default | ⚠️ only cov_recall (+5%) and amsd_recall (-9%) |
| S4 | Output lattice length spread wider than default | ❌ slightly narrower (2.33-8.00 vs 1.96-8.27) |

The expected positive result on (S1, S2) does not materialize. The negative result is itself informative — it tells us the lattice-prior is the wrong knob.

## What this changes for the parent verdict

- **Task 7's diagnosis was misleading.** The inference-only prior swap appeared to "fix" density, but it was a **mismatch artifact**, not a genuine improvement. The retrained model rejects the wider prior and reverts to the default lattice distribution.
- **The right knob is V kernel temperature or batch composition**, not the input prior. The density-wdist gap is a *kernel design* issue, not a *prior tuning* issue.
- **220× throughput remains the only unambiguous DMF win.** Validity, density, coverage, match-rate — all unchanged or worse with this experiment.

## Concrete next experiment

Vary `temperatures_L` in `compute_V_lattice`. Current `[0.5, 1.0, 2.0]` was picked from intuition. Test:
- Lower temps `[0.05, 0.1, 0.5]` — sharper kernel, V closer to nearest neighbor only
- Higher temps `[2.0, 5.0, 10.0]` — softer kernel, V toward batch mean / wider distribution

Higher τ should widen the output (each V points to batch-mean of MP-20, which has MP-20's full variance). Lower τ should narrow (V points to single nearest neighbor).

This is an inference-time evaluation IF we keep the same model, OR a retrain — but the cost is similar.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_widerprior_50k.pt` — retrained checkpoint (49.7 MB)
- `~/miad/dmf_poc/cache/wider_retrain/eval_gen.pt` — 10k generated crystals
- `~/miad/dmf_poc/cache/wider_retrain/metrics_partial.json` — metrics
- `/tmp/dmf-train-widerprior.log` — training log
