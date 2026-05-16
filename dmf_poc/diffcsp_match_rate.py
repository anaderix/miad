"""Fast match-rate from DiffCSP CSP eval output (eval_diff_mp20_csp_k20.pt).

Bypasses the slow compute_metrics.py --multi_eval. Uses the same StructureMatcher
settings, parallelized with p_map.
"""

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher


def parse_test_cif(cif):
    try:
        s = Structure.from_str(cif, fmt="cif").get_reduced_structure()
        return s
    except Exception:
        return None


def build_structure(L_3x3, F, Z):
    try:
        lat = Lattice(L_3x3)
        return Structure(lat, [int(z) for z in Z], F, coords_are_cartesian=False)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_pt", default=str(Path.home() / "diffcsp/checkpoints/mp_csp/eval_diff_mp20_csp_k20.pt"))
    ap.add_argument("--test_csv", default=str(Path.home() / "diffcsp/data/mp_20/test.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "cache/diffcsp_csp_match.json"))
    args = ap.parse_args()

    print("loading DiffCSP eval ...")
    d = torch.load(args.eval_pt, weights_only=False, map_location="cpu")
    L_all = d["lattices"]  # (20, B, 3, 3)
    F_all = d["frac_coords"]  # (20, sum_N_per_eval, 3) — note: sum_N differs per eval slot!
    nums = d["num_atoms"]  # (20, B)
    Z_all = d["atom_types"]  # (20, sum_N)
    K, B = nums.shape
    print(f"  K={K}, B={B}")

    print(f"parsing test CIFs ...")
    df = pd.read_csv(args.test_csv)
    if args.limit:
        df = df.head(args.limit)
        B = min(B, args.limit)
    test_structures = p_map(parse_test_cif, df["cif"].tolist())
    test_structures = test_structures[:B]
    valid_idx = [i for i, s in enumerate(test_structures) if s is not None]
    print(f"  GT ok: {len(valid_idx)}/{B}")

    # Build per-eval-slot cumulative-atom pointers
    # nums[k] has shape (B,); cumsum gives start offsets
    cumsum = nums.cumsum(dim=1)  # (K, B)
    # start_k_b = (cumsum[k, b-1] if b>0 else 0)
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

    def eval_one(b):
        gt = test_structures[b]
        if gt is None:
            return {"b": b, "N": 0, "matched_first": False, "matched_any": False,
                    "rmse_first": None, "rmse_best": None}
        N_b = int(nums[0, b].item())
        matched_first = False
        matched_any = False
        rmse_first = None
        rmse_best = None
        for k in range(K):
            start = int(cumsum[k, b-1].item()) if b > 0 else 0
            end = int(cumsum[k, b].item())
            Z = Z_all[k, start:end].numpy().tolist()
            F = F_all[k, start:end].numpy()
            L = L_all[k, b].numpy()
            cand = build_structure(L, F, Z)
            if cand is None:
                continue
            try:
                rms = matcher.get_rms_dist(cand, gt)
            except Exception:
                rms = None
            if rms is None:
                continue
            rms_v = rms[0]
            if k == 0:
                matched_first = True
                rmse_first = rms_v
            matched_any = True
            if rmse_best is None or rms_v < rmse_best:
                rmse_best = rms_v
        return {"b": b, "N": N_b, "matched_first": matched_first, "matched_any": matched_any,
                "rmse_first": rmse_first, "rmse_best": rmse_best}

    print("matching ...")
    import time
    t0 = time.time()
    results = p_map(eval_one, list(range(B)))
    print(f"matched in {time.time()-t0:.1f}s")

    n = len(results)
    n_first = sum(r["matched_first"] for r in results)
    n_any = sum(r["matched_any"] for r in results)
    rmses_first = [r["rmse_first"] for r in results if r["rmse_first"] is not None]
    rmses_best = [r["rmse_best"] for r in results if r["rmse_best"] is not None]

    metrics = {
        "n_compositions": n, "K": K,
        "match_rate_at_1": n_first / max(1, n),
        f"match_rate_at_{K}": n_any / max(1, n),
        "rmse_at_1_mean": float(np.mean(rmses_first)) if rmses_first else None,
        f"rmse_at_{K}_mean": float(np.mean(rmses_best)) if rmses_best else None,
    }

    by_N = defaultdict(list)
    for r in results:
        by_N[r["N"]].append(r)
    per_N = {}
    for Nk, rs in sorted(by_N.items()):
        if Nk == 0:
            continue
        per_N[Nk] = {
            "n_comps": len(rs),
            "match_rate_at_1": sum(r["matched_first"] for r in rs) / max(1, len(rs)),
            f"match_rate_at_{K}": sum(r["matched_any"] for r in rs) / max(1, len(rs)),
        }
    metrics["per_N"] = per_N

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_N"}, indent=2, default=str))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2, default=str))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
