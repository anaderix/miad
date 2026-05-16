"""Compute match@k for k=1, 2, 3, 5, 10, 15, 20 across raw eval files.

Tests the "DMF wins at higher K" hypothesis: if DMF's match rate at K=100
(or K=1000) caught up with DiffCSP's match@20, the screening-tool narrative
is quantitatively supported.

For now, since we have K=20 raw files, max k=20. Curve shape tells us
whether DMF is on a steeper trajectory than DiffCSP.
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
        return Structure.from_str(cif, fmt="cif").get_reduced_structure()
    except Exception:
        return None


def build_structure(L_3x3, F, Z):
    try:
        return Structure(Lattice(L_3x3), [int(z) for z in Z], F, coords_are_cartesian=False)
    except Exception:
        return None


def per_comp_matches(args_tuple):
    """For one composition, return list of bools: matched[k] for k=0..K-1."""
    b, gt, samples, K = args_tuple
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    matched = [False] * K
    for k, (L_k, F_k, Z) in enumerate(samples[:K]):
        cand = build_structure(L_k, F_k, Z)
        if cand is None:
            continue
        try:
            rms = matcher.get_rms_dist(cand, gt)
        except Exception:
            rms = None
        matched[k] = rms is not None
    return b, matched


def load_eval(eval_pt, test_csv, limit):
    """Returns list of (b, gt_structure, list of K samples (L, F, Z))."""
    d = torch.load(eval_pt, weights_only=False, map_location="cpu")
    K, Bn = d["num_atoms"].shape
    if limit:
        Bn = min(Bn, limit)
    df = pd.read_csv(test_csv).head(Bn)
    gts = p_map(parse_test_cif, df["cif"].tolist())
    L_all = d["lattices"]
    F_all = d["frac_coords"]
    Z_all = d["atom_types"]
    nums = d["num_atoms"]
    cumN = nums.cumsum(dim=1)
    items = []
    for b in range(Bn):
        if gts[b] is None:
            continue
        samples = []
        for k in range(K):
            start = int(cumN[k, b-1].item()) if b > 0 else 0
            end = int(cumN[k, b].item())
            Z = Z_all[k, start:end].numpy().tolist()
            F = F_all[k, start:end].numpy()
            L = L_all[k, b].numpy()
            samples.append((L, F, Z))
        items.append((b, gts[b], samples, K))
    return items, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_files", required=True,
                    help="name=path/raw.pt,... comma separated")
    ap.add_argument("--test_csv", default=str(Path.home() / "diffcsp/data/mp_20/test.csv"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=str(Path(__file__).parent / "cache/match_at_k_curve.json"))
    args = ap.parse_args()

    K_grid = [1, 2, 3, 5, 7, 10, 15, 20]
    results = {}
    for entry in args.eval_files.split(","):
        if "=" not in entry:
            continue
        name, path = entry.split("=", 1)
        print(f"\n=== {name} ===")
        items, K = load_eval(path, args.test_csv, args.limit)
        if not items:
            print("  empty")
            continue
        # match each composition against all K samples
        matches = p_map(per_comp_matches, items)
        # matches: list of (b, [bool_k])
        cov_at_k = {}
        for k in K_grid:
            if k > K:
                continue
            # match@k = fraction of compositions where ANY of first k samples matched
            count = sum(1 for _, m in matches if any(m[:k]))
            cov_at_k[k] = count / len(matches)
        results[name] = {
            "n_comps": len(matches),
            "K_available": K,
            "match_at_k": cov_at_k,
        }
        print(json.dumps(results[name], indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
