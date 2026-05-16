# Report — Task 3: DMF-CSPNet generator

**Date:** 2026-05-12
**Status:** ✅ All 7 success criteria met

## Goal

Adapt MiAD's `CSPNet` into a one-shot DMF generator `(z, atom_types, N) -> (L̂, F̂)`. The architecture must stay byte-for-byte equivalent to `cspnet-gen-default` (so capacity/inductive-bias comparison with diffusion-CSPNet is fair); the only changes are removing time conditioning and reinterpreting outputs as absolute (lattice, frac).

## Design

- File `dmf_poc/cspnet_dmf.py`, ~150 LoC.
- **Same modules:** `node_embedding` (Linear when `smooth=True`), `atom_latent_emb`, six `CSPLayer` blocks (same `edge_mlp`+`node_mlp`+`layer_norm`+`ip` knobs), `final_layer_norm`, `coord_out`, `lattice_out`, optional `type_out`.
- **What changed in `forward`:** signature is `(z_graph, atom_types, frac_in, lattices_in, num_atoms, node2graph)`. No `t` or `t_emb`. The `atom_latent_emb` layer now consumes `cat(node_features, z_per_atom)` where `z_per_atom` is the per-graph noise `z_graph` repeat-interleaved across atoms. Same Linear shape `(hidden + latent, hidden)` — repurposed.
- **Output semantics:** `F_hat = coord_out(node_features) % 1` (absolute frac on torus); `L_hat = lattice_out(scatter_mean(node_features))` (absolute lattice; `ip=True` contracts with `lattices_in` exactly as MiAD does).
- **Inputs at inference:** `frac_in ~ Uniform(0,1)^{N×3}`, `lattices_in ~ Gaussian(3,3) + 5·I`. These play the same role as the initial state in the diffusion process — message-passing happens on this initial guess and the model produces a transformed result in one pass.

## Success criteria & results

| # | Criterion | Result |
|---|---|---|
| T1 | Builds with default config | ✅ 12,295,168 params |
| T2 | Forward on (B=4, N_total=29) returns correct shapes | ✅ `L̂=(4,3,3)`, `F̂=(29,3)` |
| T3 | Backward populates `.grad` on every parameter | ✅ 68/68 params have grad |
| T4 | Param count within ±5% of `cspnet-gen-default` reference | ✅ 0.00% diff |
| T5 | Output sensitive to `atom_types` (composition conditioning works) | ✅ `max\|dL\|=0.48`, `max\|dF\|=0.11` |
| T6 | Output sensitive to `z` (diversity at fixed composition) | ✅ `max\|dL\|=9.41`, `max\|dF\|=0.89` |
| T7 | `F̂ ∈ [0,1)^3` (torus-valid) | ✅ min=0.025, max=0.90 |

The huge `max|dL|` from varying `z` (9.4) and tiny dependence on types (0.48) is **untrained-network behaviour** — at init the network amplifies its random-noise input strongly through 6 CSPLayer blocks. After training the relative magnitudes will rebalance: types carry more semantic signal, z carries the remaining stochasticity. Test passes for *qualitative* sensitivity, which is what matters at this stage.

## Validation environment

- `vm-gpu-2`, Python 3.11.15, torch 2.4.1+cu121 (miad venv)
- Run: `cd ~/miad/dmf_poc && ../.venv/bin/python test_cspnet_dmf.py`

## What's locked vs. what stays open

- **Locked:** the architecture (same as `cspnet-gen-default`), output semantics (absolute (L, F), F wrapped mod 1), z-injection point (via `atom_latent_emb`), no time conditioning.
- **Open (Task 4):** the choice of `frac_in`/`lattices_in` prior at training time. Two options to test:
  - (a) sample fresh noise every step — closest to the toy notebook;
  - (b) use the target's noisy version (curriculum). Plan to start with (a) and only revisit if loss doesn't drop.
- **Open (Task 4):** how to weight `‖L − L_target‖²` vs `‖d_tor(F, F_target)‖²` — different physical units (Å² vs torus²). Will tune λ in training.

## Next: Task 4 — training loop

Now unblocked. The pieces ready: `v_drift.compute_V(L_gen, F_gen, L_pos, F_pos)` and `CSPNetDMF`. Training loop = sample z + initial-guess → forward → V → friction-scaled target → MSE → step. Validation criterion: held-out 100-composition match-rate @20 > 30% by end of training.
