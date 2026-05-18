# Result — TASK chem-linear: continuous τ(N) = N schedule

**Date:** 2026-05-18
**Status:** ❌ Worst variant. linear τ=N gives match@20 = 14.0%, **below baseline (15.5%)** and far below chem8 (22.5%). Per-N: N=7 collapses to 10%, N=8 to 4%. Continuous schedule is no better than the step-function perN — in fact strictly worse on aggregate.

## Hypothesis (from chem-perN open question)

perN's discrete schedule (τ=2 for N≤5, τ=8 for N≥6) tied chem8 aggregate but regressed N=8 by 12pp. Suspected mechanism: training-time "regime confusion" at the boundary. A **continuous** τ(N) might smooth out this confusion.

Picked τ(N) = N as simplest interpretable linear schedule: τ=2 for N=2, τ=8 for N=8, τ=20 for N=20.

**Result: prediction wrong**. Continuous is *worse* than step-function. The "regime confusion" hypothesis is therefore probably wrong; the real issue is something else.

## Setup

- `--chem_temp_linear "0,1"` → τ = N (per-batch)
- 50k iter, repulsion=1.0, MP-20, batch=64, lr=5e-4, ctor-gpu A100 MIG
- Eval: limit=200, K=20

## Aggregate

| Variant | match@1 | match@20 | RMSE@20 |
|---|---:|---:|---:|
| baseline | 2.0% | **15.5%** | 0.398 |
| chem8 | 2.0% | **22.5%** | 0.401 |
| chem-perN (step) | 2.5% | 22.5% | 0.403 |
| **chem-linear τ=N** | 1.5% | **14.0%** | 0.409 |

chem-linear is **the only chem variant tested that underperforms baseline**. Even chem1 (τ=1) tied baseline at 15.5%.

## Per-N

| N | n | base | c8 | perN | **lin** | Δ lin vs base |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 100 | 100 | 100 | **100** | = |
| 3 | 4 | 100 | 100 | 100 | **75** | −25 |
| 4 | 24 | 50 | 75 | 75 | **38** | **−13** |
| 5 | 8 | 25 | 38 | 63 | 38 | +13 |
| 6 | 18 | 39 | 56 | 56 | **44** | +5 |
| 7 | 10 | 10 | 30 | 40 | **10** | = |
| 8 | 25 | 8 | 16 | 4 | **4** | −4 |
| ≥9 | many | 0 | 0 | 0 | 0 | = |

**N=4 is the worst stratum** (−13pp vs base, −37pp vs chem8). Reason: linear schedule gives τ=4 for N=4, which is **the dip in the chem-fill curve** (chem4 = 18.0%, worst of all chem-aware variants). Linear schedule unwittingly enforces the worst-known τ for the most populous medium stratum.

**N=7, 8 also bad**. Linear gives τ=7 for N=7 and τ=8 for N=8. τ=8 is supposed to be optimal — but here N=8 lands at 4% (vs chem8's 16% with global τ=8). So even when linear coincidentally picks the right τ per stratum, the result is worse than uniform τ=8 training.

**N=20 gets τ=20**, way past the optimum. Likely irrelevant since N=20 baseline is 0% anyway, but probably waste of effort.

## Mechanism

The perN result suggested "regime confusion" between τ=2 and τ=8 batches. If true, continuous τ(N) should be *better* than step (smoother boundary). But it's *worse*. So the confusion explanation is incomplete.

Alternative: **multi-τ training is fundamentally harder than single-τ training**, regardless of smoothness. The model has to learn how to interpret V-drifts at multiple scales simultaneously, which dilutes representation capacity. Single τ=8 lets the model specialize on one scale and use full capacity for it.

Equivalent restatement: chem_temp is essentially a hyper-parameter of the LOSS, not of the model. Changing it per-batch makes the loss landscape non-stationary across batches. Optimizer struggles.

This is consistent with chem8 100k overfitting at the same uniform τ=8: the optimal-with-uniform-τ model overfits when trained too long. The optimization landscape with single τ is well-defined; with varying τ it's a moving target.

## Decision

- **Drop both schedule variants** (perN and linear). Use uniform chem_temp=8 as canonical.
- **Per-N τ optimization** would need to happen via **separate models** per N-bucket, not a schedule. But that's expensive (5× training) for marginal gains (N=5 +25pp at best in perN, lost elsewhere).
- **Multi-τ training is harder than single-τ** — useful lesson for any future scheduling experiments (incl. lr schedules, repulsion schedules, etc.).

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | linear τ aggregate > chem8 | ❌ −8.5pp |
| S2 | linear τ aggregate ≥ baseline | ❌ −1.5pp |
| S3 | At least one stratum benefits | ⚠️ only N=5 (+13 vs base) and N=6 (+5) |
| S4 | "Smoother boundary" hypothesis validated | ❌ refuted: continuous worse than step |

0 of 4 pass. Cleanest negative outcome since cov.

## Open questions

1. **Multi-τ via separate ensembles** — train 2-3 specialist models (τ=2 for N≤5, τ=8 for N≥6), route at inference by N. No training-time confusion, but inference complexity. ~3× compute.
2. **Logarithmic schedule τ(N) = log(N) · c** — gentler increase, may avoid the τ=4 dip on N=4. Untested. Low priority.
3. **Repulsion schedule** — vary `--repulsion` instead of chem_temp. May orthogonal to chem regime issue.

## Artifacts

- `~/tmp/dmf_ckpts/dmf_chemlin_50k.pt` **(lost — see below)**
- `dmf_poc/cache/csp_chemlin.json` ✅ (survived)

## ⚠️ Environment incident note

After the eval completed, the machine was restarted and `/tmp/` (containing `~/tmp/venv-dmf/` and `~/tmp/dmf_ckpts/*.pt`) was wiped. All ckpts including this one (`dmf_chemlin_50k.pt`) and the canonical `dmf_chem8_50k.pt` are gone. Only the eval JSON results in `~/miad/dmf_poc/cache/` survived (they're on `/home/` not `/tmp/`). The repo on `/home/coder/miad/` and `~/diffcsp/data/mp_20/` are intact.

**To resume work**: rebuild venv (uv install), retrain canonical ckpts as needed (~1h each on this MIG). The canonical chem8 50k would be the first priority to restore.
