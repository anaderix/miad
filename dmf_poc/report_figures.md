# Paper figures generated

**Date:** 2026-05-13
**Status:** ✅ Three paper-ready figures generated from existing metric JSONs.

## Figures

### `fig1_k_coverage.png` — K-coverage curve (the "saturation" figure)

DiffCSP saturates by K=20 at ~80% aggregate match-rate. DMF λ=1.0 (1 NFE/sample) grows roughly linearly through K=200 to ~41%. DMF's curve doesn't catch up at the aggregate level — honest depiction of the "DMF saturates lower" finding. Use this for "complementary tool" framing in the paper.

### `fig2_per_N_compute_parity.png` — Compute-parity per-N (the killer figure)

DiffCSP K=5 (5000 NFE/comp) vs DMF K=200 (200 NFE/comp = 25× less). Green-highlighted region: **N=4-8 where DMF wins**. Bar chart cleanly shows:
- N=4: DMF 100% vs DiffCSP 87% (+13 pp)
- N=5: DMF 100% vs DiffCSP 75% (+25 pp)
- N=6: DMF 94% vs DiffCSP 83% (+11 pp)
- N=7: tied at 80%
- N=8: DMF 68% vs DiffCSP 64% (+4 pp)
- N≥9: DiffCSP wins decisively

**Use this as the main paper figure.** It captures the core claim: "DMF beats diffusion at compute-parity for small-to-medium crystals."

### `fig3_per_N_K_curve.png` — Per-N saturation curves

For N=4, 6, 8, 10, 12: how does DMF's per-N match-rate evolve with K? Shows:
- N=4: already at 80% @K=20, hits 100% @K=100 (parity with DiffCSP horizontal dotted line)
- N=6: climbs from 51% @K=20 → 94% @K=200 (above DiffCSP)
- N=8: 22% → 68% (still climbing, probably saturates near DiffCSP at K=500+)
- N=10: 0% → 4% (cliff softens but slow)
- N=12: flat at 0% (true cliff)

Dotted horizontal lines mark DiffCSP K=20 baseline per N.

## Script

`dmf_poc/plot_paper_figures.py` — generates all 3 figures from existing JSON outputs in `cache/`. Idempotent.

## How to use in paper

1. **Method section**: figure for DMF architecture (CSPNet + repulsion-V), can be made manually.
2. **Results section**:
   - Fig 2 as main result figure (per-N compute parity)
   - Table with aggregate numbers + per-N breakdown (data in `report_compute_parity.md`)
3. **Discussion**: Fig 1 for "DMF saturates lower than DiffCSP at aggregate but wins at right per-N strata" honest framing. Fig 3 for "how K shifts the support range".

## Artifacts

- `dmf_poc/fig1_k_coverage.png`
- `dmf_poc/fig2_per_N_compute_parity.png`
- `dmf_poc/fig3_per_N_K_curve.png`
- `dmf_poc/plot_paper_figures.py`
