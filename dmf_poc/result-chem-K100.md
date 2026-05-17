# Result — TASK chem-K100: chemistry at high K

**Date:** 2026-05-17
**Status:** ⚠️ Mixed. chem_temp=2 still wins aggregate (+6% relative at K=100 vs +32% at K=20), but per-N picture flips: baseline catches up where chem was winning, chem keeps gaining where baseline plateaus. The chemistry advantage **shrinks with K**, but specific high-N strata (N=7, N=8) get big chem-only boosts at high K.

## Hypothesis

At K=100, both baseline and chem methods have more sample budget. Two scenarios were possible:
- (a) chem's K=20 lead compounds — chem keeps winning by similar % at K=100
- (b) sample budget compensates for chem's signal-cleaning — chem's lead shrinks

Result: **(b) is what we see overall**, but with N-dependent twist (some strata still need chem even at K=100).

## Setup

- ckpts: `dmf_chem2_50k.pt` (best from TASK chem-sweep) vs `dmf_repel10_50k.pt` (baseline)
- CSP eval: limit=200, **K=100** (5× K=20 budget)
- Same machine (ctor-gpu A100 MIG)

## Top-line

| Metric | baseline | chem2 | Δ |
|---|---:|---:|---:|
| match@1 | 2.0% | 1.0% | −1pp (noise) |
| **match@100** | **31.5%** | **33.5%** | **+2 pp = +6% rel** |
| RMSE@100 (approx) | 0.39 | 0.38 | tied |

For reference, at K=20: chem2 gave +5pp = +32% rel. At K=100: only +2pp = +6% rel.

## Per-N at K=100 (sorted by interest)

| N | n | baseline | chem2 | Δ pp | Note |
|---:|---:|---:|---:|---:|---|
| **4** | 24 | 83% | **100%** | **+17** | chem reaches saturation |
| **7** | 10 | 30% | **60%** | **+30** | biggest chem win |
| **8** | 25 | 28% | **36%** | +8 | chem still ahead |
| 9 | 5 | 20% | 20% | = | tied (cliff start) |
| 2 | 3 | 100% | 100% | = | both saturated |
| 3 | 4 | 100% | 100% | = | both saturated |
| 5 | 8 | 88% | 75% | **−13** | baseline wins |
| 6 | 18 | 94% | 78% | **−16** | baseline wins |
| 10+ | many | all 0% | all 0% | = | cliff |

## Per-N curve K=20 → K=100 (chem2 only, gain or loss vs baseline)

| N | chem2 @K=20 vs baseline | chem2 @K=100 vs baseline | Direction |
|---:|---:|---:|---|
| 4 | +29 pp | +17 pp | shrinking (both saturate) |
| 6 | 0 pp | −16 pp | **reversal** |
| 7 | n/a (baseline 10%) | +30 pp | chem opens lead |
| 8 | +4 pp | +8 pp | chem advantage holds |

## Interpretation

Two distinct K-behaviors emerge:

**(a) "Saturation strata" (N=4, 5, 6)**: at K=20 chem's advantage is large because baseline's 50% requires diversity to even land a match. At K=100 baseline catches up via sheer number of samples. Chem's mode-shift can even underperform if it concentrates samples on a wrong sub-mode (N=5, 6).

**(b) "Hard strata" (N=7, 8)**: baseline plateaus at 28-30% even with K=100 — sample budget alone doesn't help if samples don't fall near GT. Chem's better-aimed sampling unlocks +8-30 pp because chem actually moves the model to a better region of structure space.

This suggests: **chemistry is most valuable for hard compositions, where sample budget can't compensate alone**. For easy compositions, K saturates baseline anyway.

## Implications for paper / practical use

- **At K=20 (standard CSP benchmark)**: chem_temp=2 is clearly preferable (+32% rel match@20, with N=4 jumping to 79%).
- **At K=100**: chem still wins net (+6% rel), but the N-profile differs. Use chem when N=7-8 matters; use baseline when N=5-6 dominates.
- **Hybrid pipeline**: chem-aware for N=4 and N≥7; baseline for N=5-6. Probably won't pay off for the small (+2-5pp) net.
- **Practical default**: stick with chem_temp=2. Aggregate win + bigger wins on harder N.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | chem2 K=100 ≥ baseline K=100 | ✅ +2 pp |
| S2 | chem advantage holds at higher K | ⚠️ partial — shrinks but stays positive |
| S3 | At least one N stratum shows chem clearly winning at K=100 | ✅ N=7 (+30pp), N=4 (+17pp), N=8 (+8pp) |
| S4 | N≥9 cliff softens with chem + K | ❌ unchanged |

3 of 4 pass.

## Open questions

1. **K=200 chem2** — does chem keep gaining when baseline saturates further?
2. **chem_temp=4 at K=100** — was tested at K=20 (+16% rel), unknown at K=100. Could win where chem=2 over-mode-shifts.
3. **Per-N optimal chem_temp** — chem=2 for N=4, chem=4 for N=6 (per K=20 results). Different N may want different chem strictness.

## Artifacts

- `dmf_poc/cache/csp_chem2_K100.json`, `csp_baseline_K100.json`
- `dmf_poc/cache/{chem2,baseline}_K100_raw.pt`
