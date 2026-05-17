# Result — TASK chem-perN: per-N chem_temp schedule

**Date:** 2026-05-17
**Status:** ⚠️ Wash. perN (τ=2 for N≤5, τ=8 for N≥6) **ties chem8 on aggregate** (both 22.5%) but per-N redistribution is real: perN wins N=5 (+25pp vs c8) and N=7 (+10pp) but loses N=8 (−12pp). Hypothesis ("each N gets its individually-optimal τ") **not validated as net improvement**.

## Hypothesis (from chem-fill)

chem-fill showed N=4-5 want τ=2 (sharp) and N=6-8 want τ=8 (loose). A schedule that gives each batch its locally-optimal τ should beat any global τ:
- Expected: +5-10pp on N=4-5, hold or +0-3pp on N=6-8, net +3-5pp aggregate over chem8.
- Actual: N=5 +25pp gain, N=7 +10pp gain, **N=8 −12pp regression** that cancels everything.

## Setup

- `--chem_temp_per_N "N<=5:2,N>=6:8"` (new flag)
- 50k iter, repulsion=1.0, MP-20, batch=64, lr=5e-4 on ctor-gpu
- Eval: limit=200, K=20 (same seed as other variants)

## Results vs best single-τ variants

| Variant | match@1 | **match@20** | RMSE@20 | Δ vs c8 |
|---|---:|---:|---:|---:|
| baseline | 2.0% | 15.5% | 0.398 | −7pp |
| chem2 | 4.5% | 20.5% | 0.412 | −2pp |
| chem8 | 2.0% | 22.5% | 0.401 | — |
| **chemperN** | 2.5% | 22.5% | 0.403 | 0pp (tie) |

## Per-N (the actual story)

| N | n | base | c2 | c8 | **perN** | perN vs c8 | perN vs c2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 100 | 100 | 100 | 100 | = | = |
| 3 | 4 | 100 | 75 | 100 | 100 | = | +25 |
| 4 | 24 | 50 | **79** | 75 | 75 | = | −4 |
| 5 | 8 | 25 | 50 | 38 | **63** | **+25** | +13 |
| 6 | 18 | 39 | 39 | 56 | 56 | = | +17 |
| 7 | 10 | 10 | 20 | 30 | **40** | **+10** | +20 |
| 8 | 25 | 8 | 12 | **16** | **4** | **−12** | −8 |
| ≥9 | many | 0 | 0 | 0 | 0 | = | = |

**The N=8 regression is the killer.** perN matches or beats c2/c8 everywhere except N=8 (n=25 — the single biggest stratum), where it drops to 4%. The +25pp on N=5 (n=8) and +10pp on N=7 (n=10) can't overcome a 12pp loss on a stratum 2-3× larger.

## Why N=8 regresses

Hypothesis: training with mixed τ on different batches makes the model **confused on the boundary** (N=6,7,8 all get τ=8 but the model also sees τ=2 batches for N=4-5). The CSPNet weights don't know which "chemistry regime" to operate in — it averages over both.

Evidence for this:
- N=7 *does* benefit (+10pp), suggesting the model can route by N for some strata.
- N=8 loses, suggesting the routing fails specifically there.
- N=6 unchanged (tied with c8 at 56%).
- This is a *training-time* confusion, not an eval-time issue (eval has no chem_temp).

Alternative hypothesis: noise. n=25 → standard error ~10pp at p≈0.1, so a 12pp swing on a single seed is within noise. **Would need 3-5 seeds to distinguish.**

## Why N=5 and N=7 win

N=5 jumping from 38→63% (c8→perN) and N=7 from 30→40% suggests perN's "per-N τ" matching does help where the regime is unambiguous (N=5 cleanly in "small" regime, N=7 cleanly in "big"). N=8 sits at the high end of the dataset where perN's τ=8 is the same as c8's τ=8 — so the regression there is genuinely surprising.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | perN aggregate match@20 > best single τ (22.5%) | ❌ tie at 22.5% |
| S2 | N=4-5 stratum match c2's level (79%, 50%) | ⚠️ N=5 beats (63 vs 50); N=4 worse (75 vs 79) |
| S3 | N=6-8 stratum match c8's level | ❌ N=8 regresses (4 vs 16); N=6,7 hold/improve |
| S4 | Mechanism understood (why each N reacts) | ⚠️ partial — N=8 regression unexplained without more seeds |

1 of 4 pass. Cleanest negative outcome of the chem series.

## Decision

**Stick with chem_temp=8 as canonical default.** perN is not worth the complexity — same aggregate, more uncertain per-N profile. The +25pp on N=5 is the only standalone win, and could be captured with a dedicated N=5 model if needed.

## Open questions

1. **Multi-seed stability** — single-seed runs make 10pp differences ambiguous. Would need 3-5 seeds to claim perN truly hurts N=8 (vs noise).
2. **Continuous N→τ schedule** — instead of step function, use `τ(N) = 2 + (N-4) * 1.5` or similar. Smoother transitions may avoid the boundary confusion.
3. **Larger model** — confusion hypothesis predicts a bigger network could handle multi-regime chem better. Untested.
4. **τ=8 with extra training (100k iter)** — does plain c8 keep improving or saturate at 22.5%? More direct path to higher numbers than schedule tricks.

## Artifacts

- `~/tmp/dmf_ckpts/dmf_chemperN_50k.pt`
- `dmf_poc/cache/csp_chemperN.json`, `chemperN_raw.pt`
