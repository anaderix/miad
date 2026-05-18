# Backlog — priority tasks for loop runs

Pick from here when current task is done. Move completed items to `findings.md` summary.

## High priority

- **~~chem8-DNG (DNG-lite, V+U+Nv)~~** — DONE 2026-05-18. NULL result without Stability. See `result-chem8-DNG.md`. Triggered next task:

- **chem8-S (Stability filter via CHGNet)** — gating step for real DNG numbers. Plan:
  1. `pip install chgnet` on ctor-gpu venv.
  2. Adapt `lib/prerelaxations/prerelax_chgnet.py` for our 1000-CIF input.
  3. Relax all 1000 gen structures (1500 steps each); E_hull via MP phase diagram pickle (need to source).
  4. Re-compute U and Nv only on stable subset → S, S·U, S·U·Nv numbers comparable to MiAD/DiffCSP Table 2.
  - Effort: ~1d setup + 4-8h compute.

## Medium priority

- **Larger capacity** (CSPNet 8 layers, hidden_dim 768) with chem8 — does cliff (N≥10) crack?
- **Curriculum on big-N** — oversample N=8-20 in training to push frontier.
- **Continuous N→τ schedule** — smoother per-N chem_temp, may avoid perN's N=8 regression.
- **Multi-seed perN stability** — 3 seeds to distinguish N=8 regression from noise.

## Lower priority

- **chem_temp 6, 12 fill-in** — narrow non-monotonic peak around τ=8.
- **Hybrid Z·cov chem metric** — sharp Z discriminator × soft cov tie-breaker.
- **chem8 + multi-force decomposition** (research dir a.2).
- **chem8 + learnable scalars** (research dir a.1).
