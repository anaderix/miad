# Report — Task 4c (full): 50k-iter DMF training on MP-20

**Date:** 2026-05-12
**Status:** ✅ Training completed; 4/4 inference smoke-checks pass. Real CSP-quality evaluation deferred to Task 5+6.

## Goal

Run a full-scale DMF training on MP-20 and produce a usable checkpoint. Quickvalid (5k iters) already confirmed mechanics; this iteration produced the actual model the comparison will use.

## What was run

- `train_dmf_mp20.py --max_iter 50000 --batch_size 64 --log_every 1000`
- Wall-clock: **36 min** at 22.9 it/s on H100 (shared with the running DiffCSP CSP eval)
- Throughput is constant — no GPU-memory pressure, no slowdown
- Saved checkpoint: `~/miad/dmf_poc/cache/dmf_mp20_50k.pt` (49.7 MB)

Training trajectory (selected iters):

| iter | loss | loss_L | loss_F | γ |
|---:|---:|---:|---:|---:|
| 0 | 0.222 | 0.111 | 0.0152 | 0.000 |
| 10000 | (per log, see file) | | | 0.200 |
| 20000 | (per log) | | | 0.400 |
| 30000 | (per log) | | | 0.600 |
| 40000 | 0.009 | 0.0044 | 0.0006 | 0.800 |
| 49999 | 0.0000 | 0.0000 | 0.0000 | 1.000 |

Loss-to-zero is expected (friction-schedule artifact, documented from Task 4a onward).

## Inference smoke-test

Loaded the 50k checkpoint, conditioned on 64 N=4 compositions sampled from the train cache, generated `(L̂, F̂)` once per composition with fresh z.

| Check | Result |
|---|---|
| All values finite | ✅ |
| F_hat ∈ [0,1)³ | ✅ min=0.001, max=0.997 |
| Lattice vector lengths plausible (0.5 < ‖a_i‖ < 100 Å) | ✅ min=3.26, mean=3.90, max=4.55 Å |
| Cross-sample diversity (different compositions → different outputs) | ✅ F.std=0.29, L.std=0.42 |

The mean lattice-vector length of 3.9 Å is **physically reasonable** for compact N=4 inorganic crystals (e.g. NaCl ≈ 5.6 Å, ZnS-type ≈ 5.4 Å, simple metals ~3-4 Å). The model is producing structures in the right *order of magnitude* — a non-trivial check given the noise priors (lat_in ≈ Gaussian + 5·I had mean lattice length ~7-8 Å as input, so the network is learning to compact it).

## Caveats

This validation confirms the trained model is **non-degenerate and produces physically-plausible outputs**, but it does **NOT** prove the model produces good CSP predictions. Match-rate against ground-truth structures (the gold-standard metric, identical to what we measure for DiffCSP-mp_csp) requires:

1. Inference on the full test set (Task 5)
2. StructureMatcher comparison against GT (Task 6)

Until then, this is "the training mechanic produced a working forward pass", not "DMF solves CSP".

## Open issues to revisit in Task 5/6

- **Lattice vectors all ~3.9 Å.** With std=0.42 in the cross-sample diversity, the *spread* per composition is small. The model may be producing similar-shape lattices for different compositions — i.e. the composition conditioning may be too weak. Task 6's per-composition diversity check (StructureMatcher between the 20 samples for one composition) will reveal this.

- **Did the model converge or just ride friction down?** Loss going to 0 is consistent with both (a) actually learning and (b) the friction schedule collapsing targets to gen outputs. The right diagnostic is mid-training match-rate, not loss. Could be added as a held-out evaluator if Task 6 results are weak.

- **Composition coverage of train cache.** We trained on 27,136 crystals (full MP-20 train). The N=4 group has 4,144 crystals — plenty of neighbors for V kernel. But singleton compositions (94% of train) still suffer from "no GT neighbor of the same composition" — V drifts to *near-N* peers regardless of types. This is the documented trade-off from Task 4b; the model may have learned to handle it, may not.

## Artifacts

- `~/miad/dmf_poc/cache/dmf_mp20_50k.pt` — 12.3 M-param checkpoint, 49.7 MB
- `/tmp/dmf-train-50k.log` — full training log on `vm-gpu-2`

## Next: Task 5 — DiffCSP-compatible inference + serialization

Build `evaluate_dmf.py` that:
1. Loads MP-20 test split
2. For each composition, generates 20 samples with the trained DMF
3. Serializes in DiffCSP `eval_csp_*.pt` format (compatible with their `compute_metrics.py --tasks csp`)

Once that's done + the still-running `evaluate.py mp_csp` finishes, Task 6 (final comparison) is unblocked.
