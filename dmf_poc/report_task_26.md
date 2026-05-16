# Report — Task 26: Memory bank for V positives (negative — wrong abstraction)

**Date:** 2026-05-12
**Status:** ⚠️ Per-N memory bank alone hurts match-rate (10.6% → 8.2%). The implementation is sound but the **keying granularity (per-N) is wrong** for this task. Per-composition keying would be the right abstraction but is impossible with 94% singletons.

## Hypothesis under test

From the reference JAX impl (`/home/anaderi/FlowMatching/DM/memory_bank.py`):
> Class-wise ring buffer of past samples → V kernel sees many positives per class → richer drift signal.

For ImageNet (1000 classes, 64 samples/class), this gives 64× larger effective V batch per evaluation. For our MP-20:
- **Per-composition keying** would be the direct analog, but 94% of compositions are singletons (Task 4b finding). Mem bank is empty 94% of the time.
- **Per-N keying** (compromise) gives 20 keys × 256 samples = 5120 retained crystals. Many positives, but they're from **different compositions** for any given target.

Implemented per-N. Hypothesis was that the looser keying still gives V richer positives in the right *N-conditional* neighborhood.

## Implementation

- `mem_bank.py` (~80 LoC): `CrystalMemoryBank(max_per_N=256)`. Methods: `add(L_batch, F_batch, N)`, `sample(N, n_samples)`.
- `train_dmf_mp20.py`: per step, add current batch's GT to bank, sample `mem_extra=128` extra positives, concatenate into V target.
- Effective V batch positives: 64 (current batch) + 128 (memory bank) = 192.

## Results (CSP-strict, limit=500)

| Variant | match@1 | match@20 | RMSE@20 |
|---|---:|---:|---:|
| Default K=1 | 3.0% | **10.6%** | 0.34 |
| **Repulsion-V** | **3.4%** | **16.8%** | 0.38 |
| Mem_bank alone | 1.6% | 8.2% | 0.32 |

Mem_bank ALONE underperforms default. RMSE among matches is slightly better (0.32 vs 0.34) but match-rate drops.

## Interpretation

**Per-N is too coarse.** The 128 memory-bank positives sampled for a given training step come from crystals with the same N but **wildly different compositions**. The V kernel drifts the generated sample toward the mean of these — i.e. toward the *average MP-20 N-atom crystal*, not toward the *target composition's* crystal.

This is a stronger version of the "mean-seeking" failure mode we've seen all along:
- Default V: 64 same-batch positives, half similar composition → moderate composition signal
- Mem_bank-augmented V: 64 current + 128 random-N positives → composition signal *diluted* 3× by random N-peers → stronger mean-seeking, weaker composition-conditional

**The reference impl's per-class memory bank works for ImageNet because ImageNet HAS positive class-conditional history.** Each class has thousands of training images. MP-20 has 94% singletons — there's no per-composition history to draw from.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | match@20 ≥ default's 10.6% | ❌ 8.2% |
| S2 | More balanced N-distribution (memory bank effect) | ✅ ✓ visually, but doesn't translate to match-rate |
| S3 | At least one positive metric | ⚠️ only RMSE@20 marginally better (0.32 vs 0.34) |
| S4 | Sensible memory bank dynamics (stats grow) | ✅ |

1 of 4 pass. Implementation correct, hypothesis falsified.

## What's the right abstraction then

Per-composition memory bank is **the right semantic** but impossible due to MP-20 singletons. Possible variants:

1. **Per-(N, atom-set-type) keying** — group compositions by which elements appear, not their proportions. E.g. all "binary metals with N≤6" share a bank. Reduces sparsity while preserving more compositional similarity than pure-N.
2. **Chemical similarity-weighted sampling** — sample from bank with weights based on Z-distance from current composition. Soft version of per-composition.
3. **Cross-dataset bank** — combine mp_20 + perov_5 + mpts_52 + carbon_24. More datasets dilute singleton fraction. Touched on in Task 17.
4. **Skip memory bank**, accept that V's 64-positives-per-step is the fundamental signal and don't try to "augment" it. This is what current best-DMF (repulsion) does.

Pragmatic recommendation: option 4 + repulsion. The 16.8% match@20 from Task 13 is the current ceiling, not bypassable by augmenting V's positive set with composition-incoherent samples.

## What this DOES validate

- The reference impl from /home/anaderi/FlowMatching/DM has techniques that **assume class-conditional data abundance**. MP-20 fails that precondition.
- Loss-side intervention (repulsion) remains the strongest move in our domain.
- Engineering quality of `mem_bank.py` is fine (smoke tests pass, training stable, throughput 50 it/s only ~6% slower than default). Reusable for future cross-dataset experiments (Task 17).

## Appendix — Mem_bank gen-metrics confirm negative

| Metric | Default | **Mem_bank** | Δ |
|---|---:|---:|---:|
| `valid` (↑) | 0.253 | 0.167 | **−34%** |
| `struct_valid` (↑) | 0.364 | 0.230 | **−37%** |
| `cov_recall` (↑) | 0.370 | 0.338 | −9% |
| `wdist_num_elems` (↓) | 0.234 | 0.311 | +33% (worse) |
| `cov_precision` (↑) | 0.882 | 0.882 | ≈ |
| `wdist_density` (↓) | 4.376 | 4.283 | −2% (≈ tied) |
| `amsd_recall` (↓) | 0.560 | 0.531 | −5% (mildly better) |

Validity collapses by ~35% on both metrics. The mechanism: V-kernel sees 128 per-N (composition-arbitrary) positives, drifts gen toward their geometric mean — which is *not a physically valid crystal* because it's the average of unrelated chemistries. Stronger mean-seeking on the wrong target = worse physical quality.

This **double-confirms** that the per-N memory bank is the wrong abstraction for MP-20. Useful negative result: rules out a tempting "more positives = better" intuition.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_membank_50k.pt` — ckpt
- `~/miad/dmf_poc/cache/csp_metrics_membank_limit500.json`
- `~/miad/dmf_poc/cache/membank/eval_gen.pt`, `metrics_partial.json` (gen-metrics computing)
- `mem_bank.py` (~80 LoC), `train_dmf_mp20.py` integration

## Next

Per the user's priority order: continue down the list. Since mem_bank as-is doesn't help, **next is λ_rep sweep** (Task ~13b — find optimal repulsion strength) and **mem_bank + repulsion combined** (composition test). After that, **drift_loss formulation without friction γ** (Task 27) — drops the wasteful second half of training.
