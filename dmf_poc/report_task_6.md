# Report — Task 6: DMF vs DiffCSP-mp_gen final comparison

**Date:** 2026-05-12
**Status:** Comparison complete. PoC verdict: **infrastructure works; DMF quality on this configuration is far below DiffCSP**. Throughput win is real (~220×) but structural validity collapses.

## Setup

| | DiffCSP-mp_gen | DMF-50k |
|---|---|---|
| Architecture | CSPNet (hidden=512, layers=6, latent=0) | CSPNetDMF (hidden=512, layers=6, latent=256) |
| Params | ~12.3 M | 12.3 M (matched) |
| Training | 8000 epochs on MP-20 (authors' run) | 50,000 iters on MP-20 (our run, B=64) |
| Inference NFE | 1000 (diffusion reverse) | 1 (one-shot) |
| Inference wall (10k crystals) | ~30 min | **8.3 s** (~220× faster) |
| Eval pipeline | `baseline_metrics.py` (`GenEval` minus `prop_wdist`) | identical |

Same `eval_gen.pt` schema, same metrics code, same MP-20 test set as GT. The comparison is apples-to-apples on methodology; the only thing that differs is the model.

## Side-by-side metrics

| Metric (Δ% = DMF vs DiffCSP) | DiffCSP-mp_gen | DMF-50k | Δ (R−L) | Δ% |
|---|---:|---:|---:|---:|
| `valid` (joint, higher=better) | **0.8244** | 0.2525 | −0.572 | −69.4% |
| `struct_valid` (higher=better) | **0.9987** | 0.3635 | −0.635 | −63.6% |
| `comp_valid` (higher=better) | **0.8248** | 0.6300 | −0.195 | −23.6% |
| `cov_recall` (higher=better) | **0.9968** | 0.3699 | −0.627 | −62.9% |
| `cov_precision` (higher=better) | **0.9968** | 0.8821 | −0.115 | −11.5% |
| `wdist_density` (lower=better) | **0.1336** | 4.3763 | +4.243 | +3175% |
| `wdist_num_elems` (lower=better) | 0.3239 | **0.2344** | −0.090 | −27.7% |
| `amsd_recall` (lower=better) | **0.1078** | 0.5603 | +0.453 | +420% |
| `amsd_precision` (lower=better) | **0.1282** | 0.1928 | +0.065 | +50.4% |
| `amcd_recall` (lower=better) | **2.9356** | 4.8378 | +1.902 | +64.8% |
| `amcd_precision` (lower=better) | **3.1773** | 5.1685 | +1.991 | +62.7% |

**DMF wins 1 of 11 metrics** (and `wdist_num_elems` is plausibly noise — both values small relative to scale).

## Verdict against the plan's pre-registered success criteria

From `plan` Task 6 ("paper-ready" thresholds):

| Pre-registered criterion | Target | Actual | Status |
|---|---|---|---|
| Match@20 DMF ≥ 50% of DiffCSP — minimum | ≥ 50% | n/a* | — |
| Match@20 DMF ≥ 80% of DiffCSP — target | ≥ 80% | n/a* | — |
| Throughput DMF ≥ 100× DiffCSP | ≥ 100× | **~220×** | ✅ |
| Diversity: ≥3 structural clusters per composition | qualitative | n/a* | — |
| CI non-zero on match-rate delta | yes | n/a* | — |

\* Match-rate is a CSP-specific metric (predict structure given composition). It needs the still-running `evaluate.py mp_csp` to finish, plus a CSP-style DMF eval (20 samples per test composition aligned with GT). Neither is closed yet. The gen-task metrics above are the closest available proxy.

**Proxy verdict using gen metrics:** DMF coverage recall is **37% of test compositions covered** — about 37% of what DiffCSP achieves (99.7%). Validity drops to 25% from 82%. By any reasonable reading, DMF in this configuration does **not** meet the "≥50% of DiffCSP" minimum.

**Throughput criterion is met decisively (220×).**

## Why is DMF underperforming?

The diagnostics from earlier findings (and the metrics here) point to a few related failure modes:

1. **Lattice scale collapse.** Generated lattice vector lengths span only 1.96–8.27 Å (compared to MP-20's natural 2–15 Å range, with most crystals around 4–10 Å). Compressing lattice mode coverage hurts density (Wasserstein 4.4 vs 0.13) and structural validity (overlapping atoms in small cells).

2. **Composition conditioning is weak.** Task 3's diagnostic showed (after training) `max|dL|=9.4` when varying `z` vs only `0.5` when varying `atom_types`. The network is dominated by noise, not by composition. This explains the low `cov_recall` (the same z gives different compositions broadly similar outputs).

3. **Friction schedule wastes 50% of training.** γ_t linear 0→1 means after iter 25,000 the target weight `(1-γ)·V` is already <0.5, and by iter 45,000 it's <0.1. The model effectively stops getting meaningful gradient updates well before the run ends. Cosine schedule or much earlier truncation would help.

4. **Same-N batching is a weak proxy for composition-conditional CSP.** As shown in Task 4b: 94% of MP-20 compositions are singletons. The V kernel drifts toward N-neighbours of arbitrary composition. The network is trained on a noisier objective than strict CSP — and we see the result in `cov_recall`.

5. **No SMACT-aware composition shaping.** DMF generates He / noble-gas-containing compositions because the loss has no chemistry prior. DiffCSP-mp_gen samples the composition from the train distribution (no such failure mode).

## What this PoC actually shows

**Positive:**
- The architecture and training pipeline integrate cleanly. End-to-end mechanics validated.
- ~220× inference speedup is real, robust, reproducible. **DMF-as-a-cheap-pre-screener** for a downstream relaxation/DFT pipeline is plausible even at this quality, because filtering 10k DMF crystals through CHGNet would cost roughly the same as generating them.
- Comparison framework is rigorous: same metrics, same GT, same code path. Numbers are quotable.

**Negative:**
- The plain DMF formulation with same-N kernel V does **not** produce competitive crystals on MP-20.
- A long list of improvements (items 1–5 above) are needed before this is a credible CSP method. None of them are trivial; some require redesign.

## Concrete next steps if continuing

Roughly in order of expected impact (from earlier `findings.md`):

1. **Stricter conditioning** — make `atom_types` enter the network in a way that the gradient signal cannot be diluted. E.g. concatenate per-atom type embeddings with `z` inside `atom_latent_emb`, or use FiLM-style modulation of CSPLayer features.
2. **Shorter γ schedule** — stop at γ=0.5 or use cosine. Don't waste compute.
3. **SMACT prior on generated compositions** — but in CSP setup we *receive* the composition, so this is more about ensuring the network can produce structures for any composition, including those with noble gases.
4. **Wider lattice mode** — sample lattice prior from `Lognormal(scale~MP-20-mean)` instead of N(0,1)+5·I. Would also help match the wider density distribution.
5. **CSP-strict eval** — write a DMF eval that produces 20 samples per test composition (aligned with GT order), pair with the still-pending `evaluate.py mp_csp` results, and report match-rate / RMSE directly. Then the head-to-head numbers are on the published-paper metric, not on gen-task proxies.

## Closing the parent plan

- ✅ Tasks 2, 3, 4a, 4b, 4c, 5 — done, all sub-reports under `dmf_poc/report_task_*.md`.
- ✅ Task 1 ab-initio gen baseline — done (`report_task_1_gen.md`).
- ⏳ Task 1 CSP baseline (`evaluate.py mp_csp`) — still running, ~30% (sample 10/20 in batch 5/18). Will finish many hours later. Closing it does not change the verdict above.
- ✅ Task 6 — done (this report).

## Artifacts

- `~/miad/dmf_poc/cache/metrics_partial.json` — DMF gen metrics
- `~/diffcsp/checkpoints/mp_gen/metrics_partial.json` — DiffCSP baseline
- `~/miad/dmf_poc/report_task_6_table.md` — raw comparison table
- `~/miad/dmf_poc/cache/dmf_mp20_50k.pt` — trained DMF checkpoint
- `~/miad/dmf_poc/cache/eval_gen.pt` — 10k DMF crystals in DiffCSP format
