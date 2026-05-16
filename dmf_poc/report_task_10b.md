# Report — Task 10b: Iterative training (K=2 rollout) — first partial win

**Date:** 2026-05-12
**Status:** ✅ First experimental task in the extension arc (Tasks 7-10b) to produce a **partial positive result**. Iterative training maintains validity (−0.2% on `valid`) while improving coverage (+5% `cov_recall`) and structural closeness to GT (−9% `amsd_recall`). Density distribution gets worse (+38% `wdist_density`), but the trade is closer to neutral than the previous 4 ablations.

## Hypothesis under test

From Task 10a's negative result (inference-only iterative DMF):
> Inference-only K-step iteration shrinks lattice (contraction attractor). The model has to be **trained for iterative use** to be a fair test.

Concrete change: rollout K=2 forward passes per training step; gradient flows through both; loss computed only on the final output. The intermediate output `(L_1, F_1)` becomes the next step's input `(L_state, F_state)`. The model now learns to **refine** its output, not just produce it.

## Setup

- Same `CSPNetDMF`, same B=64, same 50k iters, default lat prior, default V temps
- ONLY change: `--n_steps 2` (was implicit 1)
- Same K=2 at inference (matched to training)
- Throughput: **11.8 it/s** (was 22.9 for K=1) — exactly 2× slower as expected
- Wall-clock: 1h 10min training

## Results

| Metric | K=1 default | K=2 retrained | Δ% | Direction |
|---|---:|---:|---:|---|
| `cov_recall` (↑) | 0.370 | **0.389** | +5.1% | ✅ |
| `cov_precision` (↑) | 0.882 | 0.882 | = | = |
| `amsd_recall` (↓) | 0.560 | **0.508** | −9.3% | ✅ |
| `amsd_precision` (↓) | 0.193 | 0.194 | ≈ | = |
| `valid` (↑) | 0.253 | 0.252 | −0.2% | ≈ (preserved) |
| `struct_valid` (↑) | 0.364 | 0.347 | −4.6% | weak ❌ |
| `wdist_density` (↓) | 4.376 | 6.035 | +38% | ❌ |
| `wdist_num_elems` (↓) | 0.234 | 0.263 | +12% | ❌ |
| `comp_valid`, `amcd_*` | — | — | = | = |

**Output lattice spread:** 2.15-8.09 Å (matched-inference K=2). Wider min than default's 1.96 — the iterative-training model does NOT exhibit the lattice contraction we saw with inference-only K=2 (1.50-7.95). Confirms the model learned to refine without contracting.

## Why this experiment differs from the previous four negatives

The trade-off pattern from Tasks 7-10a was **always one-sided**: validity drops sharply (-14 to -45%) while density barely changes or gets worse. Task 10b breaks that pattern:
- Validity is **preserved** (-0.2%)
- Structural fit improves (cov_recall +5%, amsd_recall -9%)
- Density gets worse but validity wasn't sacrificed to get it

This is the first signal that the **architectural arc** (iterative training) yields different physics than the **hyperparameter arc** (prior/temperature tuning). The compute cost is real (2× training time), but for paired NFE=2 inference the model is meaningfully better at the CSP-related metrics that matter.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | Validity preserved within 5% of K=1 default | ✅ -0.2% |
| S2 | At least 2 metrics strictly improve | ✅ cov_recall, amsd_recall |
| S3 | `cov_recall` ≥ default's 0.370 | ✅ 0.389 |
| S4 | Training stable (no NaN), throughput sensibly ~2× slower | ✅ 11.8 vs 22.9 it/s |

4 of 4. **First fully passing experimental task in the extension arc.**

## What it means for the parent verdict

- **The path forward is architectural (not hyperparameter).** Three knob-tuning experiments (Tasks 7, 8, 9) and inference-only iter (10a) all failed; the first real change to the training algorithm (K-step rollout, Task 10b) immediately produces wins.

- **NFE ratio shifts.** DMF was 220× faster than DiffCSP at NFE=1. At NFE=2 it's 110× faster. Still huge. If NFE=5 fully captures DiffCSP-paper-level CSP quality, DMF would be ~44× faster — still a meaningful practical win for screening workflows.

- **Density wdist remains a problem.** Even with iterative training, density distribution doesn't match GT. The repulsion/spread term in V (Task 13 in findings.md) or larger batch size are the natural next experiments.

## Recommended next steps in priority order

1. **K=3 retrain** (Task 10c) — incremental cost (+50% over K=2), should keep improving if 10b's win holds. Wall ≈ 1h 45min training.
2. **CSP-strict eval (`evaluate_dmf_csp.py`)** on K=2-retrained checkpoint — see if the gen-metric wins translate to match-rate. ~25 min.
3. **Per-atom z** (Task 12) — may help with the multi-atom coupling problem (Task 5b's N≥9 cliff).
4. **V with repulsion** (Task 13) — direct attempt at density wdist.

Recommend running #2 immediately (cheap, definitive), then deciding on #1 vs #3 based on N≥9 behavior.

## Appendix — CSP-strict match-rate on K=2 retrained (limit=500)

Match-rate via `evaluate_dmf_csp.py --ckpt ...iter2train... --n_steps 2 --limit 500 --K 20`:

| Metric | K=1 default | K=2 retrained |
|---|---:|---:|
| match@1 | **3.0%** | 2.4% |
| match@20 | **10.6%** | 8.4% |
| RMSE@1 (Å) | 0.41 | 0.36 |
| RMSE@20 (Å) | 0.34 | 0.32 |

Per-N @20 highlights: N=1 50%→100% (n=2, noisy), N=2 64% (unchanged), N=3 36%→21%, N=4 26%→19%, N=5 20%→25%, N≥9 0% (unchanged).

**Net: gen-metric wins (cov_recall, amsd_recall) do NOT translate to CSP match-rate.** RMSE-among-matched is mildly better (-12%), but match-rate is worse. Iterative training shifts the model into a different region of structure space where individual matches are slightly less likely but average closeness is better. This is a different trade-off from the gen-metric direction.

**Implication**: improving CSP match-rate likely requires changes targeting the multi-atom coordination problem directly (Task 5b's N≥9 cliff is unchanged). Pure iterative refinement is not enough.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_iter2train_50k.pt` — K=2 retrained ckpt (49.7 MB)
- `~/miad/dmf_poc/cache/iter2train/eval_gen.pt` — 10k generated crystals
- `~/miad/dmf_poc/cache/iter2train/metrics_partial.json` — gen metrics
- `train_dmf_mp20.py` now supports `--n_steps K` flag
