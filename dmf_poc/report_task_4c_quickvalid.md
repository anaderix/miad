# Report — Task 4c (quickvalid): DMF training integrates with real MP-20

**Date:** 2026-05-12
**Status:** ✅ 3/3 mechanics criteria met. Full 50k-iter run launched in background.

## Goal

Verify the **integration** of Tasks 2, 3, 4a, 4b on real MP-20 data: dataloader yields valid batches, V kernel + friction loss + CSPNetDMF compose into a stable optimizer, training is GPU-bound at reasonable throughput.

This is a *mechanics* validation, not a *convergence-quality* validation. The latter requires actual generation at inference and match-rate against GT — that's Tasks 5 + 6.

## What was run

- `train_dmf_mp20.py --max_iter 5000 --batch_size 64`
- Auto-calibrated `lambda_frac = 7.31` (ratio of initial `loss_L / loss_F`)
- GPU: H100, parallel with the running DiffCSP CSP eval

## Success criteria & results

| # | Criterion | Result |
|---|---|---|
| S1 | Smoothed loss decreases by ≥20% (start vs end) | ✅ 0.289 → 0.0002 (>3 orders of magnitude) |
| S2 | No NaN/Inf | ✅ |
| S3 | Final F_hat ∈ [0,1)^3 | ✅ min=0.0001, max=0.9998 |

Throughput: **22.8 it/s** at batch=64. Same-N batching with N varying from 3 to 20 between batches works fine (model handles variable N graphs without code changes — was a design concern, now resolved).

## Caveats (what the run does NOT prove)

The loss → 0 result is partly the **friction-schedule artifact** noted in Task 4a's findings: at γ=1 the target collapses to `x_gen`, so loss → 0 trivially. The drop from `0.289` (γ≈0) → `0.0002` (γ=1) is consistent with what friction does even on a model that learned nothing.

The *real* convergence diagnostic is match-rate on held-out GT, which requires inference and StructureMatcher — that's Task 5 + 6. This sub-task closes only the *mechanics-integration* validation.

## Full run kicked off

Launched 50,000 iters in background (`/tmp/dmf-train-50k.log`, ckpt `cache/dmf_mp20_50k.pt`). Expected wall: ~37 min at observed throughput. After completion + Tasks 5+6, we will know whether the model *learned anything CSP-useful*.

## Validation environment

- `vm-gpu-2`, MiAD venv, torch 2.4.1+cu121
- Cached `cache/mp20_train.pt` (Task 4b artifact)
- 5k-iter run wall: ~3.5 min

## Next: Task 4c-full evaluation, then Task 5

Task 4c-full closes when the 50k-iter checkpoint exists (≈ next /loop iteration) and an inference-side smoke test produces non-degenerate samples. Task 5 then writes the DiffCSP-compatible eval script.
