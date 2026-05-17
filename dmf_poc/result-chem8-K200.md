# Result — TASK chem8-K200: cliff extrapolation at K=200

**Date:** 2026-05-17
**Status:** ⚠️ Mixed. **Aggregate** climbs from 36.0% (K=100) → 39.5% (K=200), +3.5pp. **N=6 saturates** to 100% (+22pp from K=100). **N=8 keeps gaining** (+12pp to 52%). But **the cliff is reach-bound, not budget-bound**: N=10 stays at 1/26 (3.8%), no new N≥11 strata cracked despite 2× budget.

## Setup

Eval-only of `dmf_chem8_50k.pt` at K=200, limit=200, same eval seed as K=20/K=100 runs.

## chem8 across K (aggregate match-rate)

| K | match@K | Δ vs prev | rel gain per K-doubling |
|---:|---:|---:|---:|
| 20 | 22.5% | — | — |
| 100 | 36.0% | +13.5pp | +60% rel (5× budget) |
| 200 | 39.5% | +3.5pp | +10% rel (2× budget) |

**Strong diminishing returns**: K=20→100 gave +60% rel; K=100→200 gave only +10%. chem8 is approaching its inherent ceiling on this dataset.

## Per-N at K=200 (with K=20/100 for trajectory)

| N | n | base@100 | c8@20 | c8@100 | **c8@200** | Δ200 vs 100 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 100 | 100 | 100 | 100 | = |
| 3 | 4 | 100 | 100 | 100 | 100 | = |
| 4 | 24 | 83 | 75 | 100 | 100 | = |
| 5 | 8 | 88 | 38 | 88 | 88 | = |
| **6** | 18 | 94 | 56 | 78 | **100** | **+22** |
| 7 | 10 | 30 | 30 | 80 | 80 | = |
| **8** | 25 | 28 | 16 | 40 | **52** | **+12** |
| 9 | 5 | 20 | 0 | 20 | 20 | = |
| 10 | 26 | 0 | 0 | 3.8 | **3.8** | **= (1 match)** |
| ≥11 | many | 0 | 0 | 0 | 0 | = |

## What K=200 reveals

**Two distinct dynamics:**

### (1) Reachable strata still have headroom — K helps
N=6: 78→100% (saturated to perfect). N=8: 40→52% (still climbing — would likely keep going at K=500). These compositions live in regions chem8 *can* reach; more samples → more matches.

### (2) Cliff strata are reach-bound — K doesn't help
N=10: chem8 found 1 structure reachable at K=100 (3.8%) — the **same** 1 structure remains the only one found at K=200. The model cannot reach the other 25 N=10 compositions regardless of budget. Same for N=11-20 (all 0% at K=200 as at K=100).

This is a clean diagnostic: **the cliff is an architectural/training limitation, not a sampling limitation**. Doubling K bought zero new N≥10 matches.

## What this means for the canonical pipeline

- **chem8 + K=100 is a sensible sweet spot**. K=200 adds 3.5pp but at 2× compute. K=500 would likely add another 1-2pp.
- **To break N≥10**, chem8 won't suffice — need either model changes (larger network, better lattice prior, attention over coordination geometry) or training changes (longer iter, schedule, curriculum on big-N).
- **N=8 is the frontier**: it's the largest N still gaining at K=200 (+12pp). Targeted N=8 improvements likely cheapest win toward an aggregate boost.

## Aggregate progress arc through this PoC

| Stage | aggregate (best metric) | Note |
|---|---|---|
| Initial baseline (no chem, K=20) | 15.5% | repulsion-only |
| chem2 K=20 | 20.5% | first chem variant |
| chem8 K=20 | 22.5% | best sweep result |
| chem8 K=100 | 36.0% | K-scaling |
| chem8 K=200 | **39.5%** | current best |

From 15.5% to 39.5% — **2.55× match-rate** via two knobs (chem_temp + K). Most of the budget gain came from K=20→100; chem_temp accounts for the rest.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | K=200 ≥ K=100 aggregate | ✅ +3.5pp |
| S2 | New N strata cracked beyond K=100 | ❌ — same N=10 (1 match), N≥11 all 0 |
| S3 | At least one N saturates at 100% | ✅ N=4,6 |
| S4 | K-budget vs reach effect distinguished | ✅ clear diagnostic |

3 of 4 pass. The cliff diagnostic (S4) is more valuable than the failed S2.

## Implication for next steps

The PoC's match-rate ceiling on N≤9 strata is now well-characterized. Further gains require:

1. **Capacity** — bigger CSPNet (8 layers vs 6, hidden_dim 768 vs 512)
2. **Training time** — 100k iter (chem8 may not have converged at 50k for hard strata)
3. **Curriculum** — focus training on N=8-12 to push frontier
4. **Architecture** — coordination-aware modules for N≥10 (lattice/space-group inductive bias)

These are bigger investments than knob-tuning. The PoC has reached its knob-tuning limit.

## Artifacts

- `dmf_poc/cache/csp_chem8_K200.json`, `chem8_K200_raw.pt`
