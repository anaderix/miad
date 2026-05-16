# Report — Task 11: Per-atom z (negative result, big)

**Date:** 2026-05-12
**Status:** ⚠️ Hypothesis falsified. Per-atom z **dramatically hurts** match-rate at all N. The "more degrees of freedom per atom = better multi-atom coordination" intuition was backwards: per-atom independent noise *destroys* inter-atomic coherence, not enhances it.

## Hypothesis under test

From `findings.md` after Task 10b:
> **Per-atom z**: each atom gets its own latent, not just per-graph. May help with multi-atom coordination (Task 5b's N≥9 cliff).

Concrete change: sample `z ~ N(0,1)^{N_total, latent_dim}` instead of `(B, latent_dim)`. CSPNetDMF was already updated to handle either shape (backward-compatible).

## Setup

- Same architecture, 50k iters, K=1, default lat prior
- ONLY change: `--per_atom_z` flag during train AND eval (matched)
- Throughput: **53 it/s** (exclusive GPU; would have been ~25 with shared) → ~15 min train

## Results (CSP-strict eval, limit=500)

| Metric | DMF K=1 default (per-graph z) | DMF K=1 per-atom z |
|---|---:|---:|
| match@1 | **3.0%** | 2.0% (−33%) |
| match@20 | **10.6%** | **3.0% (−72%)** |
| RMSE@20 (Å) | 0.34 | 0.26 (mild improvement among matched) |

**Per-N @20 (limit=500):**

| N | n | Default | Per-atom z |
|---:|---:|---:|---:|
| 2 | 14 | 64.3% | 42.9% |
| 3 | 14 | 35.7% | 21.4% |
| 4 | 70 | **25.7%** | **1.4%** |
| 5 | 20 | 20.0% | 10.0% |
| 6 | 43 | 14.0% | 2.3% |
| 7-20 | 281 | 0-21% | 0% |

The drop is severe especially for N=4 (25.7% → 1.4%). The N≥9 cliff is unchanged (still 0%). **Per-atom z broke the model's small-N capability without recovering any large-N performance.**

## Interpretation

The intuition behind "per-atom z" was that more DoF per atom would let the network express different per-atom drift decisions. The actual effect:

- **Per-graph z** acts as a *shared context vector* across all atoms in a crystal. All atoms see the same z and produce coherent outputs (the network learns to use this shared signal as a "this is what crystal this is" instruction).
- **Per-atom z** gives each atom **independent** random conditioning. The V kernel computes drift per atom, and with independent z's the drift signals decorrelate. Inter-atomic message passing has to fight harder to coordinate atoms toward a single coherent crystal.

Result: the network's ability to produce a coherent crystal degrades. Match-rate, which requires the entire crystal to match GT, suffers most.

## Success criteria

| # | Criterion | Result |
|---|---|---|
| S1 | Per-N N≥9 match@20 > default's 0% | ❌ still 0% |
| S2 | Top-line match@20 ≥ default's 10.6% | ❌ dropped to 3.0% |
| S3 | At least 2 per-N improvements over default | ❌ all per-N degraded |
| S4 | RMSE@20 improves OR validity preserved | ⚠️ RMSE@20 mildly improved (0.34 → 0.26) but match-rate collapsed |

0 of 4 pass. Sixth negative result in the extension arc.

## Lessons reinforced

1. **Architectural changes are not automatically beneficial.** Just because per-atom z gives "more parameters per crystal" doesn't mean training finds a useful use for them. The coupling provided by shared z appears essential to coherent crystal generation.

2. **The per-graph z is more than a noise injection — it's a crystal-level context.** Each crystal in a batch sees its own z, and that z effectively *names* the sample. When you replace it with per-atom z, you lose the crystal-naming and each atom thinks it's in a different sample.

3. **Cross-atom coherence is what diffusion's iterative refinement gives for free**, and what DMF cannot easily replicate. Each diffusion step lets atoms "see" each other's current positions and adjust accordingly. Per-graph z + 1-shot DMF tries to do this in a single forward, and apparently this works for small N but breaks for ≥4-5 atoms with too many independent noise channels.

## What this still doesn't help

The N≥9 cliff is unchanged. None of the architectural experiments tried so far (K=2, per-atom z, prior tuning, V temp tuning) has put a single match in N≥9 territory.

The remaining viable architectural levers per `findings.md`:
- FiLM-style composition conditioning (different from input z)
- V kernel with explicit repulsion (mostly affects density wdist)
- Larger K (K=5, K=10) iterative training — but expensive

None promises to break N≥9. At this point the honest verdict is: **multi-atom coordination on MP-20 needs either (a) iterative refinement à la diffusion or (b) much more expressive architecture than CSPNetDMF**. DMF as a one-shot or few-shot method has a hard ceiling.

## Appendix — Gen metrics on per-atom z (10k samples)

| Metric | default | per-atom z | Δ% |
|---|---:|---:|---:|
| `wdist_density` (↓) | 4.376 | **2.805** | **−36% improvement** |
| `struct_valid` (↑) | 0.364 | 0.228 | −37% |
| `valid` (↑) | 0.253 | 0.185 | −27% |
| `cov_recall` (↑) | 0.370 | 0.240 | −35% |
| `wdist_num_elems` (↓) | 0.234 | 0.329 | +40% |
| `amsd_recall` (↓) | 0.560 | 0.596 | +6% |
| `comp_valid`, `cov_precision`, `amcd_*` | — | — | = |

Per-atom z is the **first** experiment that meaningfully improved `wdist_density` (4.38 → 2.80) without an inference-prior trick. But it pays for it with the same validity/coverage drops as the prior-tuning experiments — same Pareto frontier, just a different point on it.

**Net:** Density wdist gain doesn't compensate for the match-rate disaster (10.6% → 3.0% @20). Per-atom z is overall negative.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_peratom_50k.pt` — per-atom-z retrained checkpoint
- `~/miad/dmf_poc/cache/csp_metrics_peratom_limit500.json` — match-rate
- `~/miad/dmf_poc/cache/peratom/eval_gen.pt` — 10k generated crystals
- `~/miad/dmf_poc/cache/peratom/metrics_partial.json` — gen metrics (still computing)
- `evaluate_dmf.py`, `evaluate_dmf_csp.py`, `cspnet_dmf.py`, `train_dmf_mp20.py` now support `--per_atom_z`
