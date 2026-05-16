# DMF-on-MP20 PoC

Adaptation of Drifting Models with Friction (DMF) to crystal structure prediction (CSP) on the MP-20 dataset. PoC tests whether a one-shot generative model with kernel-mean-shift drift loss can compete with diffusion-based CSP (DiffCSP) on the same benchmark.

## Headline result

**DMF+repulsion-V matches diffusion-based CSP for half of MP-20 (compositions with N=2-8 atoms) at 40× lower inference compute.** Hard cliff at N≥9 due to one-shot architecture limitations.

See `SUMMARY.md` for the full verdict.

## Where to read first

1. **`SUMMARY.md`** — final consolidated summary, paper-grade verdict, headline numbers.
2. **`findings.md`** — running log of all 30+ key findings across 25 sub-tasks. Read entries with dates to follow chronology.
3. **Figures** — `fig1_k_coverage.png`, `fig2_per_N_compute_parity.png`, `fig3_per_N_K_curve.png` (paper-ready).
4. **Best result report** — `report_per_N_K100.md` (covers K=20, 100, 200, 500 results per-N).

## Code layout

```
v_drift.py            — V kernel (multi-temp + torus + repulsion)
cspnet_dmf.py         — CSPNet adapted: no time conditioning, optional FiLM
mp20_loader.py        — MP-20 loader with same-N batching
mem_bank.py           — per-N memory bank (negative-result experiment, kept for reference)
train_dmf_mp20.py     — training loop (--repulsion, --film, --n_steps, --no_friction, --per_atom_z, --mem_bank, --lat_prior_*)
evaluate_dmf.py       — DiffCSP-compatible gen serialization (--n_steps, --film, --per_atom_z, --lat_prior_*)
evaluate_dmf_csp.py   — CSP-strict eval (--save_raw for downstream analyses)
baseline_metrics.py   — GenEval (minus prop_wdist) for any eval_gen.pt
diffcsp_match_rate.py — fast match-rate from DiffCSP CSP eval output
diversity_analysis.py — per-composition StructureMatcher cluster count
match_at_k_curve.py   — match@k for k ∈ {1,...,K} across multiple raw eval files
compare_metrics.py    — side-by-side comparison table generator
plot_paper_figures.py — generates fig1, fig2, fig3 from cache/*.json
test_v_drift.py, test_cspnet_dmf.py — sanity tests (all pass)
```

## Reports (chronological)

| Report | Topic |
|---|---|
| `report_task_1_gen.md`, `report_task_1_csp.md` | DiffCSP baseline reproduction |
| `report_task_2.md` | V kernel design |
| `report_task_3.md` | CSPNet-DMF generator |
| `report_task_4a.md`, `report_task_4b.md`, `report_task_4c_*.md` | Training mechanics + MP-20 loader |
| `report_task_5.md`, `report_task_5b.md` | Inference + CSP-strict eval |
| `report_task_6.md`, `report_task_6a_prep.md` | First DMF vs DiffCSP comparison |
| `report_task_7.md` — `report_task_11.md` | 5 architectural ablations (mostly negative) |
| **`report_task_13.md`, `report_task_13b.md`, `report_task_13c.md`** | **Repulsion-V breakthrough + λ sweep** |
| `report_task_14.md` | FiLM (negative) |
| `report_task_24.md` | Diversity analysis (3× more diverse than DiffCSP) |
| `report_task_26.md` | Memory bank (negative — wrong granularity for MP-20) |
| `report_task_27.md` | No-friction experiment (mildly negative) |
| `report_task_k_curve.md` | K-coverage curve |
| **`report_per_N_K100.md`** | **Per-N at K=100/200/500 — the killer result** |
| `report_compute_parity.md` | Compute-parity head-to-head |
| `report_figures.md` | Paper figures |

## Reproduce the best result

```bash
ssh vm-gpu-2  # H100 instance
cd ~/miad/dmf_poc

# Train (50k iters, ~17 min on H100, exclusive GPU)
../.venv/bin/python train_dmf_mp20.py --max_iter 50000 --repulsion 1.0 \
    --ckpt_path cache/dmf_mp20_repel10_50k.pt

# CSP eval @ K=200, limit=200 compositions (~50 min)
../.venv/bin/python evaluate_dmf_csp.py --ckpt cache/dmf_mp20_repel10_50k.pt \
    --K 200 --limit 200 \
    --out cache/csp_metrics_repel10_K200.json \
    --save_raw cache/repel10_csp_raw_K200.pt

# Diversity vs DiffCSP
~/diffcsp/.venv/bin/python diversity_analysis.py \
    --eval_files "diffcsp=/home/anaderi/diffcsp/checkpoints/mp_csp/eval_diff_mp20_csp_k20.pt,dmf=cache/repel10_csp_raw_K200.pt" \
    --limit 200 --out cache/diversity_final.json

# Paper figures
../.venv/bin/python plot_paper_figures.py
```

## Cached artifacts (on `vm-gpu-2:~/miad/dmf_poc/cache/`)

- `mp20_train.pt` (~150 MB) — parsed MP-20 train tensors
- `dmf_mp20_*.pt` (~50 MB each) — trained checkpoints for each variant
- `eval_gen.pt`, `*_csp_raw_*.pt` — inference output dumps
- `*.json` — per-variant metric dumps for `compare_metrics.py` / `plot_paper_figures.py`

## Limitations (open problems)

1. **N≥12 hard cliff**: DMF cannot generate 12+ atoms per cell regardless of K. One-shot architecture insufficient for multi-atom coordination.
2. **N=9-10 partial coverage** (~20% at K=500): cliff softens but doesn't disappear. Likely needs much larger K or architectural changes.
3. **Per-composition memory bank impossible** on MP-20 (94% singletons). Combined-dataset training might help.

## Possible follow-up directions

If continuing this PoC for a paper, in priority order:
1. **Full 9046 CSP eval** at K=200 — paper-grade CI on per-N numbers
2. **DFT/CHGNet relaxation** of DMF candidates — strengthens "screening tool" narrative
3. **K=1000 specifically for N=8-10** — see how far cliff softens
4. **Train on combined datasets** (mp_20+perov_5+mpts_52+carbon_24) — more data, potentially better generalization
5. **Multi-step DMF (K=5-20 iterative training)** — may unlock N≥9 if combined with repulsion

## Sessions arc

- 25 sub-tasks completed over a single intensive session (May 12-13, 2026)
- 17 markdown reports + this README
- ~2000 LoC focused PoC code
- Major findings: (a) repulsion in V loss is necessary for competitive match-rate; (b) DMF is intrinsically divergent (3× DiffCSP diversity); (c) effective support range N=2-8 at 40× lower compute
- 8 of 9 hyperparameter/architecture ablations were net-negative; the single loss-side intervention (repulsion) broke through.
