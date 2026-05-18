# Backlog — priority tasks for loop runs

Pick from here when current task is done. Move completed items to `findings.md` summary.

## High priority

- **~~chem8-DNG (DNG-lite, V+U+Nv)~~** — DONE 2026-05-18. NULL result without Stability. See `result-chem8-DNG.md`. Triggered next task:

- **~~chem8-S (Stability via CHGNet proxy)~~** — DONE 2026-05-18. S~·U·Nv = 8.5% (n=200), in DiffCSP ballpark. See `result-chem8-S.md`.

- **chem8-S full pipeline** — scale up: n=1000, max_steps=1500, real E_hull via MP phase diagram (download `2023-02-07-ppd-mp.pkl` or build from pymatgen MPRester). Tighter comparison to MiAD paper. ~8h compute.

- **chem-temp × DNG sweep** — map the CSP↔DNG trade-off curve. Retrain & DNG-relax for chem2, chem16. Hypothesis: monotonic — sharper chem hurts S more, looser chem ≈ baseline. ~6h.

- **chem8 × repulsion sweep** — does lower repulsion preserve DNG quality while keeping chem8's CSP wins? Try chem8 with `--repulsion 0.5` and `0.0`. ~4h.

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
