"""Per-composition diversity analysis: count distinct structural clusters
among the K generated samples for each test composition.

Reads CSP-style eval files (with eval data per-composition × K) and the
test CSV. For each composition: cluster the K samples by StructureMatcher,
count distinct clusters. Aggregate across compositions.

Higher cluster-count = more diverse generations = less mode-collapse.

Usage:
  python diversity_analysis.py \
    --eval_files variant_name=path/to/eval.pt,name2=...
  → reads each file's 20 samples × test compositions, computes cluster counts
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher


def parse_eval_for_clusters(eval_pt, test_csv, limit=None):
    """For each composition, return list of K Structure objects."""
    d = torch.load(eval_pt, weights_only=False, map_location="cpu")
    L_all = d["lattices"]  # CSP-format (20, B, 3, 3) OR gen-format (B, 3, 3)
    if L_all.ndim == 3:
        # gen-format: K=1, only one sample per composition
        return None
    K, B = d["num_atoms"].shape
    if limit:
        B = min(B, limit)

    F_all = d["frac_coords"]   # (K, sum_N, 3)
    nums = d["num_atoms"]       # (K, B)
    Z_all = d["atom_types"]     # (K, sum_N)
    cumN = nums.cumsum(dim=1)   # (K, B)

    out = {}
    for b in range(B):
        samples = []
        for k in range(K):
            start = int(cumN[k, b-1].item()) if b > 0 else 0
            end = int(cumN[k, b].item())
            Z = Z_all[k, start:end].numpy().tolist()
            F = F_all[k, start:end].numpy()
            L = L_all[k, b].numpy()
            try:
                s = Structure(Lattice(L), [int(z) for z in Z], F)
                samples.append(s)
            except Exception:
                pass
        out[b] = samples
    return out


def count_clusters(structures, matcher):
    """Cluster structures by StructureMatcher equivalence; return number of distinct clusters."""
    if not structures:
        return 0
    clusters = []
    for s in structures:
        merged = False
        for c in clusters:
            try:
                if matcher.fit(s, c[0]):
                    c.append(s)
                    merged = True
                    break
            except Exception:
                pass
        if not merged:
            clusters.append([s])
    return len(clusters)


def analyze_one(args_tuple):
    """Worker for p_map: (b, structures, matcher_config)."""
    b, structures = args_tuple
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    return b, count_clusters(structures, matcher)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_files", required=True,
                    help="comma-separated name=path/eval.pt pairs; only CSP-format (K, B, ...) accepted")
    ap.add_argument("--limit", type=int, default=200, help="first N test compositions to analyze (default 200)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "cache/diversity_analysis.json"))
    args = ap.parse_args()

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    summary = {}
    for entry in args.eval_files.split(","):
        if "=" not in entry:
            print(f"skip malformed entry: {entry}", file=sys.stderr)
            continue
        name, path = entry.split("=", 1)
        print(f"\n=== {name} ({path}) ===")
        per_comp = parse_eval_for_clusters(path, None, limit=args.limit)
        if per_comp is None:
            print(f"  skip: not CSP-format (need K>1 samples per composition)")
            continue
        n_comps = len(per_comp)
        print(f"  loaded {n_comps} compositions, K={len(next(iter(per_comp.values())))} samples each")

        items = list(per_comp.items())[: args.limit]
        results = p_map(analyze_one, items)
        cluster_counts = [c for _, c in results]
        cluster_counts = [c for c in cluster_counts if c > 0]

        s = {
            "n_comps_analyzed": len(cluster_counts),
            "K_samples_per_comp": len(next(iter(per_comp.values()))),
            "mean_clusters": float(np.mean(cluster_counts)),
            "median_clusters": float(np.median(cluster_counts)),
            "max_clusters": int(np.max(cluster_counts)),
            "frac_collapsed_1cluster": float(np.mean(np.array(cluster_counts) == 1)),
            "frac_diverse_ge3": float(np.mean(np.array(cluster_counts) >= 3)),
        }
        print(json.dumps(s, indent=2))
        summary[name] = s

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
