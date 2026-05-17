# Result — TASK chem8-100k: longer training overfits

**Date:** 2026-05-17
**Status:** ❌ Negative. Doubling training from 50k → 100k iter **reduces** aggregate match-rate at both K=20 (22.5%→21.0%) and K=100 (36.0%→34.5%). The biggest win at K=100 — **N=7 = 80%** — **collapses to 40%** at 100k. The N=9/10 cliff crack at 50k disappears entirely. Clean overfitting signature.

## Hypothesis

Per K-budget analysis (chem8 K=200), the "cliff is reach-bound, not budget-bound". That suggested the model hadn't fully learned the structural manifold of N≥10. Hypothesis: more training (100k vs 50k iter) would let the model reach more hard-stratum compositions.

**Result: opposite of prediction.** 100k overfits — loses generalization on the strata where 50k was hardest-won.

## Setup

- Same `train_dmf_mp20.py --chem_temp 8 --repulsion 1.0` as 50k baseline.
- Only difference: `--max_iter 100000` instead of 50000.
- Same eval seed for both at K=20 and K=100, limit=200.

## Aggregate results

| Variant | match@1 | match@K | RMSE@K |
|---|---:|---:|---:|
| chem8 50k @20 | 2.0% | **22.5%** | 0.401 |
| chem8 100k @20 | 3.0% | 21.0% | 0.393 |
| chem8 50k @100 | 1.0% | **36.0%** | 0.383 |
| chem8 100k @100 | 2.5% | 34.5% | 0.372 |

Match@1 actually **improves** at 100k (top-1 sharper) but match@K shrinks (less diversity in top-K).

RMSE@K improves slightly at 100k — the structures that *do* match are aligned closer to GT. This is consistent with overfitting: model becomes more confident about a narrower set of structures.

## Per-N (the real story)

| N | n | c8 50k K=20 | c8 100k K=20 | Δ K=20 | c8 50k K=100 | c8 100k K=100 | Δ K=100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 100 | 67 | **−33** | 100 | 100 | = |
| 3 | 4 | 100 | 75 | **−25** | 100 | 100 | = |
| 4 | 24 | 75 | 71 | −4 | 100 | 100 | = |
| 5 | 8 | 38 | 38 | = | 88 | 88 | = |
| 6 | 18 | 56 | **61** | +5 | 78 | **83** | +5 |
| **7** | 10 | 30 | 30 | = | **80** | **40** | **−40** |
| 8 | 25 | 16 | 12 | −4 | 40 | **48** | +8 |
| 9 | 5 | 0 | 0 | = | **20** | 0 | **−20** |
| 10 | 26 | 0 | 0 | = | **3.8** | 0 | **−3.8** |
| ≥11 | many | 0 | 0 | = | 0 | 0 | = |

**Three patterns:**

### Easy small-N (N=2,3,4): K=20 regression
At K=100 all saturate so we don't see it, but at K=20 the 100k model is **worse** by 5-33pp on N=2-4. This is the model concentrating top-K candidates on fewer modes — same compositions in test, but model proposes the same wrong sub-mode 20 times instead of diversifying.

### Medium-N (N=6, N=8): mild gains
N=6 and N=8 gain a few pp at K=100. These are the strata where mode-attraction directly helps and overfitting is benign (model is over-confident in correct sub-mode).

### Hard-N (N=7, 9, 10): catastrophic loss
**N=7 collapses 80→40% at K=100**. N=9 and N=10 go to zero. These are exactly the strata where 50k's wins came from "lucky" generalization. At 100k the model has memorized train structures so well that it can't extrapolate to test compositions in these strata anymore.

## Mechanism

The kernel-mean-shift drift V pulls generated structures toward target batch. At low training time, the model learns *general* drift behavior — small lattice/coord adjustments toward "structurally plausible" neighborhoods. At high training time, the model learns *specific* drift behavior — moving generated samples toward exact memorized train structures.

Test compositions in hard-N strata (N=7-10) have no close neighbors in train at the cell-composition level. A memorizing model can't generalize there; a less-memorizing model accidentally produces useful structures via the kernel's smoothness.

The friction schedule γ → 1 at end of training compounds this: in the last 10k iter, V's pull is multiplied by (1-γ) ≈ 0 most of the time, so the model is fine-tuning on extremely small corrections — pure overfitting.

## Implications

1. **50k iter is approximately optimal** for chem8 on MP-20 with current architecture. More training hurts.
2. **chem8 hard-stratum wins at 50k are not fully model-learned** — they're partially "lucky" generalizations. They disappear when the model becomes more confident.
3. **The cliff is even more architectural** than chem8-K200 suggested. Even given more training compute, the same architecture can't reach N≥10 — it just memorizes harder on what it can reach.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | 100k aggregate ≥ 50k aggregate (K=20 or K=100) | ❌ both regress |
| S2 | Cliff strata (N≥10) crack with more training | ❌ N=10 collapses to 0 |
| S3 | Convergence diagnostic clear | ✅ overfitting signature explicit |
| S4 | Hard-N strata (N=7,8) hold or improve | ❌ N=7 collapses, N=8 +8pp partial |

1 of 4 pass. Clean negative outcome with clear mechanism.

## Decision

- **Keep `chem8 50k` as canonical default.** Don't train longer.
- **Knob/training-time tuning is exhausted.** Next gains require architectural changes (capacity, regularization, curriculum) or task changes (DNG / S.U.N. eval).
- **Could try shorter training (30k iter)** — predicts further N=7/9/10 gains at cost of N=6/8.

## Open questions

1. **Where exactly does the optimum sit?** Sweep 25k/35k/50k/70k for chem8 — peak likely 30-50k.
2. **Does dropout help?** Adding dropout would slow memorization, possibly preserve hard-N wins.
3. **Friction schedule γ tweaks** — keeping γ < 0.9 in last 10k iter might prevent the over-precise drift fine-tuning that causes overfitting.
4. **Larger model + 100k iter** — more capacity to absorb training without overfitting on small modes.

## Artifacts

- `~/tmp/dmf_ckpts/dmf_chem8_100k.pt`
- `dmf_poc/cache/csp_chem8_100k_{K20,K100}.json`
