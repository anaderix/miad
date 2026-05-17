# Backlog — priority tasks for loop runs

Pick from here when current task is done. Move completed items to `findings.md` summary.

## High priority

- **chem8-DNG (S.U.N. with empirical-prior composition sampling)** — currently chem8 only does CSP (composition given). For real S.U.N. evaluation:
  1. Sample (N, atom_types) from MP-20-train empirical distribution as a prior.
  2. Run chem8 generation per sampled composition (existing `evaluate_dmf_csp.py`-style pipeline but with prior-sampled compositions instead of test.csv).
  3. Relax with CHGNet (cheaper) via existing `lib/prerelaxations/`.
  4. Compute S.U.N. via `lib/metrics/`.
  5. Compare with DiffCSP-baseline numbers in MiAD paper.
  - Why: match-rate gain (15.5→39.5%) needs to translate to a real generative metric. Expect higher Stability (mode-attraction helps); risk: lower Novelty (model lands on train-neighbors).
  - Effort: ~1d (composition sampler is new; relaxation pipeline exists).

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
