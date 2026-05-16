# Report — Task 4b: MP-20 data loader for DMF training

**Date:** 2026-05-12
**Status:** ✅ 5/5 success criteria met

## Goal

Build a data loader that yields batches of MP-20 crystals in a format the DMF training loop (`v_drift.compute_V` + `CSPNetDMF`) can consume. Key constraint: V is per-atom-positional, so crystals within a batch must share `N` (number of atoms) — otherwise the (B, N, 3) frac tensor is undefined.

## Implementation

- `dmf_poc/mp20_loader.py`, ~130 LoC, pure-pymatgen (uses the same `pymatgen.Structure.from_str` + `get_reduced_structure` as DiffCSP).
- Loads `~/diffcsp/data/mp_20/train.csv` (DiffCSP-format CSV with a `cif` column), parses each CIF via `p_map`, applies Niggli reduction (matches DiffCSP hparams `niggli=True`).
- Caches the parsed tensors to `~/miad/dmf_poc/cache/mp20_train.pt` for ~instant subsequent loads.
- `same_N_iterator(data, batch_size)` yields random batches of `B` crystals all sharing the same `N`; iterates groups (N values) in random order, shuffles crystals within each group, falls through partial last batches.

## Success criteria & results

| # | Criterion | Result |
|---|---|---|
| T1 | All MP-20 train crystals load without error | ✅ 27,136 / 27,136 |
| T2 | N-distribution printed for sanity | ✅ N ∈ [1, 20]; mode N=4 (4144), N=10 (2640), N=12 (2624) |
| T3 | Each batch has constant N | ✅ verified across all 433 batches |
| T4 | Shapes consistent: L=(B,3,3), F=(B,N,3), Z=(B,N) | ✅ |
| T5 | One-epoch iteration < 60 s after cache | ✅ **0.05 s** (essentially free) |

Parse wall: **35.5 s** for 27,136 CIFs (single-machine `p_map` parallel). Cache: 1× the time, then ~instant.

## Stats and design implication: composition stratification

Compositions in MP-20 train (counting `(sorted_Z_tuple, N)` as identity):

- **Unique compositions:** 24,830
- **Median frequency per composition:** 1
- **Singletons (composition appears once):** 23,340 (94%)
- **Top frequencies:** 16, 16, 15, 14, 13 — i.e. even the most-repeated composition has only ~16 GT crystals

→ **94% of MP-20 compositions are singletons.** Strict per-composition batching (Plan task 1: "conditional CSP given exact composition") would force batch_size = 1 for 94% of training, which is degenerate for kernel-mean-shift (V needs ≥2 neighbours).

This rules out strict per-composition V. The default in `same_N_iterator` is the looser strategy: batch by N only. Crystals with the same N but different compositions co-exist in a batch; V pulls a generated crystal toward neighbours in (L, F)-space regardless of their composition. The generator network (CSPNetDMF) still gets atom_types as input and must learn the composition→structure mapping implicitly through the gradient signal.

**This is a meaningful narrowing of Task 1's CSP setup.** Pure CSP requires (composition → distribution-over-structures); MP-20 simply does not provide enough samples per composition to model that distribution kernel-empirically. What we'll be training is more honestly described as **N-conditional generation, with atom types as conditioning input**. Match-rate evaluation against ground-truth (Task 6) still works as the comparison metric, but the underlying task is structurally different from DiffCSP-`mp_csp`.

## Validation environment

- `vm-gpu-2`, MiAD venv (Python 3.11, pymatgen)
- Cache path `~/miad/dmf_poc/cache/mp20_train.pt` (~150 MB-ish)
- Run: `cd ~/miad/dmf_poc && ../.venv/bin/python mp20_loader.py`

## Next: Task 4c — Real MP-20 training of CSPNetDMF

Now unblocked. Plug `same_N_iterator` into the synthetic training loop from Task 4a, with:
- `atom_types` from `Z` field (one-hot to MAX_ATOMIC_NUM=100)
- λ_frac tuning per the finding in Task 4a (raw modality losses observed on first batch)
- Periodic eval against a held-out 100-composition subset
- 200k iters target (per plan), but watch for convergence; can stop earlier

Also Task 1 (DiffCSP baseline) still gated on running `baseline_metrics` and `evaluate.py --model_path mp_csp`. Not blocking Task 4c.
