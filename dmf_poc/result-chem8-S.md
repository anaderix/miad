# Result — TASK chem8-S: real DNG metrics with CHGNet stability proxy

**Date:** 2026-05-18
**Status:** ✅ First real DNG numbers. **chem8 S~·U·Nv = 8.5%** (17/200 generated structures are stable+unique+novel under CHGNet relax with `|ΔE|≤0.3 eV/atom` vs train baseline). This is **in the same ballpark as DiffCSP** (S·U·Nv ≈ 7-8% per MiAD paper Table 2) using a proxy metric. Convergence rate 15.5% confirms most chem8 outputs are physically far from minima.

## Setup

- Ckpt: `~/projects/dmf_ckpts/dmf_chem8_50k.pt` (re-trained after `/tmp/` wipe)
- 200 compositions sampled from MP-20-train empirical (N, sorted_Z) prior
- One DMF structure per composition (no K-replicas)
- **CHGNet StructOptimizer** (412K params, runs on GPU MIG)
  - max_steps = 500
  - fmax convergence = 0.1 eV/Å
- Per-composition **train baseline**: relax first train structure of the same composition, save final E/atom (200/200 = 100% converged in 42s total)
- Metrics chain:
  - **V**: pymatgen-buildable
  - **converged**: CHGNet relax reached `fmax < 0.1` within 500 steps
  - **S~** (stability proxy): converged AND `|E_gen − E_train_baseline| ≤ 0.3 eV/atom`
  - **U|S~**: pairwise unique within same composition (on relaxed structs)
  - **Nv|S~·U**: not matching any train structure of same composition (on relaxed)

## Top-line numbers

| Metric | chem8 | Notes |
|---|---:|---|
| Validity (V) | 100.0% | parseable; baseline check |
| **CHGNet converged** | **15.5%** (31/200) | physical-plausibility test — most gen structs hit max_steps |
| **Stability proxy (S~)** | **10.5%** (21/200) | converged ∧ ΔE OK |
| U \| S~ | 100.0% (21/21) | all stable gen are mutually distinct |
| Nv \| S~·U | 81.0% (17/21) | most stable+unique are not in train |
| **S~·U·Nv / n_gen** | **8.5%** (17/200) | headline DNG metric |
| Mean gen final force | **7.34 eV/Å** | train converges to <0.1; gen is ~70× farther from minima |
| Mean gen final E | −3.08 eV/atom | compare to typical crystal ~−5 to −7 eV/atom |

Compare to MiAD-paper Table 2 (full pipeline, n=10k, E_hull stability):
- DiffCSP: S ≈ 50%, S·U·Nv ≈ 7-8%
- MiAD: S ≈ 60%, S·U·Nv ≈ 11-12%

Our S~ is 10.5% (vs MiAD's S≈60%) because:
1. **Stricter convergence criterion** here: we *require* CHGNet to converge in 500 steps. MiAD-paper allows 1500 steps + uses E_hull check on final structure regardless of CHGNet convergence.
2. **Proxy ΔE filter** (0.3 eV/atom from per-composition train baseline) — different from MP convex-hull check.
3. **n=200 vs 10k**: significant noise — `S~ = 0.105 ± √(0.105·0.895/200) ≈ ±2.2%` 95% CI.

The **combined S~·U·Nv = 8.5% is the most comparable** number — it's near DiffCSP's reported 7-8%.

## What the convergence rate tells us

- **CHGNet did converge for 31/200 gen structures**. These are "the model can produce structures that relax to a stable state".
- **169/200 hit max_steps** (avg final force 7.3 eV/Å). For these, gen output is so far from any minimum that 500 steps of relaxation can't bring it there.
- **All 200 train baselines converged in <0.5s/struct** (42s total). For comparison gen took 5868s = ~29s/struct average, **~140× slower per struct**.

This is **direct evidence** that chem8 outputs are far from physical minima. The kernel-mean-shift V doesn't enforce physics — it just averages toward train neighbors in (L, F) space.

## Per-composition energy gap analysis (post-hoc)

For the 31 converged structures:
- 21/31 are within `|ΔE| ≤ 0.3 eV/atom` of train baseline (the S~ criterion) — model lands near correct energy
- 10/31 converged to a state >0.3 eV/atom away from baseline (different local min — model converges but to a different polymorph)

This 21/10 split is interesting: when chem8 converges, it's ~67% likely to be near the right energy. Train-conditional generation → relaxation reaches the right basin most of the time.

## Interpretation

The picture across DNG-lite and DNG-relax:
- **DNG-lite** (no S filter): V=100, U=99.9, Nv=98.6 — all trivially saturated. Useless.
- **DNG-relax** (with S~ filter): V=100, S~=10.5, U=100, Nv=81. **Stability is the only discriminative dimension**.

So:
1. **The bottleneck for chem8 (and probably any DMF variant) is Stability**, not Uniqueness or Novelty.
2. **Stability is a continuous spectrum** in this PoC: V=100% (parseable), converges=15.5% (settles), S~=10.5% (settles near correct E). Each level is a stricter filter.
3. **chem8's match@K wins translate to S~·U·Nv ≈ 8.5%**, which sits at DiffCSP level. Whether that means "chem8 ≈ DiffCSP-quality" or "our proxy S is just easier than MP E_hull" — we can't disentangle without running the full E_hull pipeline.

## Comparison to expected (from result-chem-DNG.md hypothesis)

Predicted:
- "Higher Stability (mode-attraction helps)" → S~ 10.5% > 0 ✓ but well below MiAD-paper's 60%
- "Lower Novelty (model lands on train-neighbors)" → 81% Nv is surprisingly **high** for a mode-attraction model. Either:
  - chem8 doesn't actually collapse to train-neighbors (StructureMatcher tolerance is wide; chem8 relaxes to *near* train but not exact)
  - Sample size effects (4/21 lost to train-match is ~19% non-novelty; weak signal at n=21)

The "model lands on train-neighbors" worry was overstated. chem8 produces structures that **relax to states near** train compositions, but not identical to specific train structures.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | Pipeline runs end-to-end | ✅ 200 gen, all metrics |
| S2 | S~ > 0 (model produces some physical structures) | ✅ 10.5% |
| S3 | S~·U·Nv in same order-of-magnitude as DiffCSP | ✅ 8.5% vs 7-8% reported |
| S4 | Mechanism understandable | ✅ stability is the bottleneck |

4 of 4 pass. First real DNG result for the PoC.

## Headline numbers for paper/summary

```
chem8_50k DMF, MP-20, n_gen=200:
  V          = 100.0%
  converged  =  15.5%
  S~         =  10.5%  (CHGNet relax + ΔE ≤ 0.3 eV/atom vs train baseline)
  U|S~       = 100.0%
  Nv|S~·U    =  81.0%
  S~·U·Nv    =   8.5%
```

vs literature (MiAD paper, MP-20, n_gen=10000, full E_hull S):
```
DiffCSP    S=50%, S·U·Nv ≈ 7-8%
MiAD       S=60%, S·U·Nv ≈ 11-12%
```

## Open questions / next experiments

1. **Larger n_gen** — re-run with n=1000 (5× more compute = 8h on this MIG). Better confidence intervals; CHGNet may converge faster after the first 200 (cached state?).
2. **Looser ΔE threshold** — try 0.5 eV/atom or full E_hull. Likely raises S~ by 2-5pp.
3. **Longer max_steps** (1000 or 1500) — directly comparable to MiAD-paper. Likely lifts convergence rate.
4. **Compare baseline chem-free DMF** (15.5% match@20) — does it produce ANY converged structures? Useful negative control.
5. **Compare chem2 ckpt** — was 20.5% match@20. May give different S~ profile.
6. **Real E_hull integration** — install pymatgen.ext + source MP API key. Then can report S directly comparable to paper.

## Artifacts

- `~/projects/dmf_ckpts/dmf_chem8_50k.pt`
- `dmf_poc/cache/dng_relax_chem8.json`
- `dmf_poc/evaluate_dmf_dng_relax.py` (script supporting both CSV and train cache fallback)
