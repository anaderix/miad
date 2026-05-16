# Report — Task 13c: Repulsion + FiLM composition (mild interference)

**Date:** 2026-05-13
**Status:** ⚠️ Composition partially negative. repel=1.0 + FiLM gives match@20 = **24.4%** — 2 pp below repel=1.0 alone (26.6%). FiLM and repulsion **partially cancel**: FiLM's stronger composition conditioning re-introduces mean-seeking that repulsion was trying to fix.

## Hypothesis

From Task 13b's recommended next:
> **λ_rep=1.0 + FiLM** — FiLM gave best RMSE-among-matches. Combined with repulsion's diversity, could give both spread AND tight matches.

FiLM modulates each CSPLayer by composition embedding → stronger composition signal. Repulsion pushes generated samples away from each other → spread. The hope: composition keeps samples on the right "family" while repulsion spreads them within that family.

## Setup

- Default arch + `--film` + `--repulsion 1.0`
- 50k iters @ 47.7 it/s (~17 min train) — FiLM's extra MLP adds ~7% wall time vs repel alone
- Same evals as Task 13b

## Results vs reference variants

| Variant | NFE | match@1 | match@20 | RMSE@20 |
|---|---:|---:|---:|---:|
| Default K=1 | 1 | 3.0% | 10.6% | 0.34 |
| FiLM alone | 1 | 2.6% | 4.6% | **0.22** |
| **repel=1.0** | 1 | 3.4% | **26.6%** | 0.39 |
| repel=2.0 | 1 | 2.6% | 20.8% | 0.38 |
| **repel=1.0 + FiLM** | 1 | 2.8% | 24.4% | 0.38 |
| DiffCSP-mp_csp | 1000 | 56.6% | 84.4% | 0.046 |

FiLM+repulsion composition is **worse on aggregate match@20** but with interesting per-N shifts.

## Per-N: composition shifts the distribution

| N | repel=1.0 | **repel+FiLM** | DiffCSP @20 | Δ vs repel |
|---:|---:|---:|---:|---:|
| 2 | 79% | **86%** | 86% | **+7 pp** — matches DiffCSP exactly! |
| 3 | 86% | 86% | 100% | = |
| 4 | 80% | 79% | 99% | −1 |
| 5 | 70% | 50% | 95% | −20 |
| 6 | 51% | 44% | 84% | −7 |
| 7 | 26% | 21% | 79% | −5 |
| 8 | 22% | 14% | 84% | −8 |
| **10** | 0% | **2%** | 90% | **+2 pp — first N≥9 nonzero in entire arc!** |
| 9, 11-20 | 0% | 0% | 70-90% | = |

**Two notable findings**:
1. **N=2 hits 86% — matches DiffCSP exactly** on 2-atom cells.
2. **N=10 has one match** (1/51 = 2%). This is **the first nonzero N≥9 match across all our experiments**. Statistical noise on n=51, but the right ballpark to flag.

Loss column: N=5, 6, 7, 8 all drop. The composition is a redistribution — small-N gains, mid-N losses.

## Interpretation

FiLM and repulsion are **partially antagonistic loss-side interventions**:
- **FiLM** strengthens composition conditioning → mean-seeking toward composition's typical structure
- **Repulsion** pushes samples away from each other → anti-mean-seeking

When composed, the model's effective drift direction is a vector sum of these two forces. Net effect: per-composition samples spread slightly less than under pure repulsion → tighter match for very-small-N (where compositional info dominates) but less spread for mid-N (where repulsion's spread was the key win).

The N=10 single match is anomalous but interesting. Maybe FiLM's per-layer composition signal is just-barely enough to handle 10-atom coordination occasionally. Worth a closer look with full 9046 eval to determine if this is reproducible.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | match@20 ≥ repel=1.0's 26.6% | ❌ 24.4% (-2pp) |
| S2 | N≥9 stratum breaks (any nonzero) | ✅ N=10 hits 2% (1/51) |
| S3 | At least one N stratum improves vs repel=1.0 alone | ✅ N=2 +7pp to 86% (matches DiffCSP) |
| S4 | No N stratum regresses by >10pp | ❌ N=5 −20pp |

2 of 4 pass. Composition shifts the per-N distribution but reduces aggregate.

## Implication

repel=1.0 alone is the **single best DMF variant for aggregate match@20** found in this PoC. Composition with FiLM is interesting for *some N* but not for *aggregate*. For specialized models:
- **N=2 service** — use repel=1.0+FiLM (86% = DiffCSP-parity)
- **N=3-8 general** — use repel=1.0 alone (best per-N profile)
- **N≥9** — open problem, not solved by any DMF variant

## Appendix — Gen-metrics for repel+FiLM

| Metric | Default | repel=0.5 | repel=1.0 | **repel=1.0+FiLM** |
|---|---:|---:|---:|---:|
| `valid` (↑) | 0.253 | **0.408** | 0.349 | 0.270 |
| `struct_valid` (↑) | 0.364 | **0.624** | 0.511 | 0.388 |
| `cov_recall` (↑) | 0.370 | 0.395 | 0.354 | 0.381 |
| `wdist_density` (↓) | 4.38 | 4.50 | 4.75 | **3.73** ← best of all variants |
| `wdist_num_elems` (↓) | 0.234 | **0.113** | 0.210 | 0.199 |
| `amsd_recall` (↓) | 0.560 | 0.541 | 0.563 | 0.540 |
| `amsd_precision` (↓) | 0.193 | 0.191 | 0.193 | **0.187** ← best |
| `cov_precision` (↑) | 0.882 | 0.882 | 0.882 | 0.882 |

**Multiple Pareto-optima**:
- **Match-rate optimum**: repel=1.0 (26.6%)
- **Validity optimum**: repel=0.5 (val=0.41, struct=0.62)
- **Density-distribution optimum**: repel=1.0+FiLM (wdist_density=3.73)
- **Compositional-distribution optimum**: repel=0.5 (wdist_num_elems=0.11)
- **Precision optimum**: repel=1.0+FiLM (amsd_precision=0.187)

No single variant dominates. **For paper-grade results**, recommend reporting repel=1.0 as the headline (best match-rate) but also showing the Pareto frontier across other metrics — different operating points are useful for different downstream applications.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_repel10_film_50k.pt` — ckpt (62 MB)
- `~/miad/dmf_poc/cache/csp_metrics_repel10film_limit500.json`
- `~/miad/dmf_poc/cache/repel10film/eval_gen.pt`, `metrics_partial.json`

## Recommended next

1. **Full 9046 CSP eval on repel=1.0** — tighten CI on the 26.6% headline, paper-grade
2. **λ_rep=1.5** — fine-tune optimum between 1.0 (best) and 2.0 (saturated)
3. **Scale-normalized V** (from /home/anaderi/FlowMatching/DM reference) — different cure for the lat-vs-frac scale mismatch
4. **DFT relaxation** on repel=1.0 candidates — see if near-match crystals are at least near-stable
