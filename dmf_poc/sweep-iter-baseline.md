# Sweep — baseline iter ladder

Goal: find baseline (no chem) training-iter optimum. We know chem8 at 100k overfits
(match@20 22.5→21.0, N=7 80→40). Test whether baseline (no chem-attraction) tolerates
longer training.

Gate at each step: continue only if metrics improve OR are within noise.

## Gate criteria (compared to 50k baseline reference)

50k baseline reference:
- match@20 (CSP, n=200, K=20): **15.5%**
- real S·U·Nv (n=200, max_steps=500, E_hull≤0.08): **6.5%**

For each next step:
- 🟢 GREEN (continue): match@20 ≥ 16.5% (+1pp) AND S·U·Nv ≥ 6.0%
- 🟡 YELLOW (continue cautiously, one more step): match@20 ∈ [14.5%, 16.5%] AND S·U·Nv ∈ [5.5%, 7.5%]
- 🔴 RED (stop, document): match@20 < 14% OR S·U·Nv < 5%

## Steps (sequential)

1. **baseline 100k iter** (~1h train + 12min CSP + 1.5h ehull = 3h total)
   - Gate check → decide on 250k
2. **baseline 250k iter** if green/yellow at 100k (~5h train + 1.7h eval = 7h)
   - Gate check → decide on 500k
3. **baseline 500k iter** if green/yellow at 250k (~10h train + 1.7h eval = 12h)
   - Final result, compare to DiffCSP

Total worst-case: ~22h compute.

## Tracker

| Step | Started | Train done | Eval done | match@20 | real S·U·Nv | Gate | Decision |
|---|---|---|---|---:|---:|---|---|
| 50k (ref) | — | — | — | 15.5% | 6.5% | — | reference |
| 100k | _pending_ | | | | | | |
| 250k | _gated_ | | | | | | |
| 500k | _gated_ | | | | | | |
