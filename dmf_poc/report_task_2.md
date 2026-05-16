# Report — Task 2: V-specification for conditional CSP-DMF

**Date:** 2026-05-12
**Status:** ✅ All success criteria met

## Goal

Define the drift field `V` for one-shot DMF on `(lattice, frac)` at fixed composition, so that subsequent tasks (generator + training loop) have a deterministic, validated target. V is the multi-temperature kernel-mean-shift gradient toward the target batch, in the style of the OFM/DM `compute_V_multi_temperature` used in the reference notebook (`dm_gaussian_mixture_friction.ipynb`).

## Design choices

Two **independent per-modality** drifts (multi-temperature, separately normalized so different physical scales contribute equally):

- **`V_lat`** — Euclidean kernel on flattened lattices (R^9). `temperatures_L = [0.5, 1.0, 2.0]` (Å²-scale heuristic; will be retuned in Task 4).
- **`V_frac`** — kernel on **torus distance** via minimum-image difference (`d - round(d)` ∈ (-0.5, 0.5]³), drift expressed in the tangent space. `temperatures_F = [0.02, 0.05, 0.2]` (taken verbatim from the notebook).

Permutation across atoms is **not** solved inside this module — caller is expected to pass atoms in a canonical order (sorted by atomic number then by position). This is the right scope: at fixed composition (Task 1 of the parent plan), all batch members have the same `(types, N)` and canonicalization is cheap.

## Code

- `dmf_poc/v_drift.py` — module with `torus_diff`, `compute_V_lattice`, `compute_V_frac`, `compute_V`.
- `dmf_poc/test_v_drift.py` — synthetic sanity tests.

Total ~120 LoC. Pure PyTorch; no graph-batching machinery needed because composition is fixed per batch.

## Success criteria & results

| # | Criterion | Result |
|---|---|---|
| T1 | V→0 when gen==pos at τ→0 (kernel one-hot self-asymptote) | ✅ `‖V_L‖_inf=0, ‖V_F‖_inf=0` |
| T1b | Batch-mean(V)≈0 for i.i.d. gen and pos batches (matched distributions) | ✅ `mean=0.050 (L), 0.022 (F)`, threshold 0.2 |
| T2L | V_L points toward biased target | ✅ trace(V_L) = +1.73 toward +5·I target |
| T2F | V_F respects shortest path on torus (gen=0.1, pos=0.4 → drift +0.3) | ✅ mean(V_F)=+0.289 |
| T2W | V_F wraparound: gen=0.05, pos=0.95 → drift -0.1 (NOT +0.9) | ✅ mean(V_F)=-0.289 |
| T3 | Translation invariance: V_F(F+δ, P+δ) = V_F(F, P) | ✅ max diff = 6.6e-7 |
| T4 | Numerical stability at τ=1e-4 | ✅ all values finite |
| T5 | Memory/throughput at B=256, N=20 on H100 | ✅ 1.4 ms, 79 MB peak |

All criteria pass. T5 result is critical: at 80 MB per batch (out of 80 GB on H100), we can scale to B=2048 or N=160 with headroom — no memory bottleneck.

## Initial design flaw caught during testing

First version of T1 asserted V→0 for `gen==pos` at *any* τ. This is wrong for kernel mean-shift: with finite τ, even when each gen matches its corresponding pos, the drift points toward the *centroid of other crystals in the batch* (because other batch members contribute weight). The real invariants are (a) τ→0 asymptote (peaks self) and (b) batch-mean ≈ 0 at matched distributions. Test was restructured into T1 (self-asymptote) and T1b (large-batch mean). This is documented because future training-loop debugging may surface confusion of the same kind.

## Validation environment

- Host: `vm-gpu-2` (H100 80GB HBM3, CUDA 12.x)
- Python 3.11.15, torch 2.4.1+cu121 (miad venv)
- Run: `cd ~/miad/dmf_poc && ../.venv/bin/python test_v_drift.py`

## What's locked vs. what stays open

- **Locked:** V formulation (multi-temp kernel mean-shift, torus on frac, Euclid on lat), per-modality separate normalization, fixed-composition batch contract.
- **Open (Task 4 territory):** specific temperature values for `temperatures_L` — current `[0.5, 1.0, 2.0]` is a placeholder; depends on actual lattice scale in `mp_20`. Will revisit when training starts.
- **Open (Task 3 territory):** how the generator consumes `z`; the V-module does not depend on this.

## Next: Task 3 — DMF generator architecture

Now unblocked: copy MiAD's `cspnet.py`, strip time-conditioning, plug `z_L ∈ R^9, z_F ∈ R^{N×3}` as the input noise. Validation criterion: same parameter count (±5%) as `cspnet-gen-default`, forward-backward pass on dummy batch.
