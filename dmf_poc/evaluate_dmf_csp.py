"""Task 5b — CSP-strict evaluation of DMF: match-rate @1 and @K on MP-20 test.

For each test composition, sample K=20 DMF structures conditioned on that
composition's atom_types and N. Use pymatgen StructureMatcher (ltol=0.3,
stol=0.5, angle_tol=10° — standard CSP-paper settings) to compare each
generated sample against the test GT.

Reports:
  - match_rate@1:  % test compositions where the FIRST sample matches GT
  - match_rate@K:  % test compositions where ANY of K samples matches GT
  - rmse@1, rmse@K:  among matched pairs, average StructureMatcher.get_rms_dist
  - per-N breakdown

Comparable to DiffCSP paper Table 1 (MP-20 CSP) — they report @1 ≈ 51%, @20 ≈ 64%.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher

sys.path.insert(0, str(Path(__file__).parent))
from cspnet_dmf import CSPNetDMF, MAX_ATOMIC_NUM


def parse_test_cif(cif: str):
    """Return (gt_structure, Z_list, N) or None on failure."""
    try:
        s = Structure.from_str(cif, fmt="cif").get_reduced_structure()
        Z = [site.specie.Z for site in s.sites]
        return s, Z, len(Z)
    except Exception:
        return None


def make_one_hot(Z_batch_list, device):
    flat = torch.cat([torch.as_tensor(z, device=device, dtype=torch.long) for z in Z_batch_list])
    flat = flat.clamp(0, MAX_ATOMIC_NUM - 1)
    out = torch.zeros(flat.shape[0], MAX_ATOMIC_NUM, device=device)
    out.scatter_(1, flat.unsqueeze(1), 1.0)
    return out


def build_structure(L: torch.Tensor, F: torch.Tensor, Z: list[int]):
    """L: (3,3), F: (N,3), Z: list of atomic numbers -> pymatgen Structure or None."""
    try:
        lat = Lattice(L.cpu().numpy())
        species = [int(z) for z in Z]
        coords = F.cpu().numpy()
        return Structure(lat, species, coords, coords_are_cartesian=False)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "cache/dmf_mp20_50k.pt"))
    ap.add_argument("--test_csv", default=str(Path.home() / "diffcsp/data/mp_20/test.csv"))
    ap.add_argument("--K", type=int, default=20, help="num_evals per composition")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="optionally limit test compositions for quicker turnaround")
    ap.add_argument("--out", default=str(Path(__file__).parent / "cache/csp_metrics.json"))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n_steps", type=int, default=1)
    ap.add_argument("--per_atom_z", action="store_true")
    ap.add_argument("--film", action="store_true")
    ap.add_argument("--save_raw", default=None,
                    help="if set, also save raw (K, B, ...) tensors in DiffCSP CSP format to this path")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    torch.manual_seed(args.seed)

    # load model
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = CSPNetDMF(
        hidden_dim=512, latent_dim=256, num_layers=6, num_freqs=128,
        ln=True, ip=True, smooth=True, pred_type=False, film=args.film,
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # parse test
    df = pd.read_csv(args.test_csv)
    if args.limit:
        df = df.head(args.limit)
    print(f"parsing {len(df)} test CIFs ...")
    parsed = p_map(parse_test_cif, df["cif"].tolist())
    items = [(i, p[0], p[1], p[2]) for i, p in enumerate(parsed) if p is not None]
    print(f"  ok: {len(items)}")

    # group test items by N for efficient batched inference
    from collections import defaultdict
    by_N = defaultdict(list)
    for idx, gt, Z, N in items:
        by_N[N].append((idx, gt, Z))

    # generate K samples per composition, grouped by N
    print(f"\ngenerating K={args.K} samples per composition ...")
    t0 = time.time()
    gen_per_comp = {idx: [] for idx, _, _, _ in items}
    for N, group in sorted(by_N.items()):
        for start in range(0, len(group), args.batch_size):
            chunk = group[start:start + args.batch_size]
            B = len(chunk)
            Z_list = [Z for _, _, Z in chunk]
            atom_types = make_one_hot(Z_list, dev)
            num_atoms_t = torch.full((B,), N, device=dev, dtype=torch.long)
            node2graph = torch.arange(B, device=dev).repeat_interleave(N)

            for k in range(args.K):
                if args.per_atom_z:
                    z = torch.randn(B * N, model.latent_dim, device=dev)
                else:
                    z = torch.randn(B, model.latent_dim, device=dev)
                frac_in = torch.rand(B * N, 3, device=dev)
                lat_in = torch.randn(B, 3, 3, device=dev) + torch.eye(3, device=dev) * 5.0
                with torch.no_grad():
                    L_state, F_state = lat_in, frac_in
                    for _ in range(args.n_steps):
                        L_hat, F_hat = model(z, atom_types, F_state, L_state, num_atoms_t, node2graph)
                        L_state, F_state = L_hat, F_hat
                F_hat_bn = F_hat.view(B, N, 3)
                for i, (idx, _, _) in enumerate(chunk):
                    gen_per_comp[idx].append((L_hat[i].cpu(), F_hat_bn[i].cpu()))
        print(f"  N={N}: done ({len(group)} comps)  elapsed={time.time()-t0:.1f}s")

    print(f"\ngeneration done in {time.time()-t0:.1f}s")

    # match against GT
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

    def evaluate_one(item):
        idx, gt, Z, N = item
        samples = gen_per_comp[idx]
        matched_any = False
        matched_first = False
        rmse_first = None
        rmse_best = None
        for k, (L_k, F_k) in enumerate(samples):
            cand = build_structure(L_k, F_k, Z)
            if cand is None:
                continue
            try:
                rms = matcher.get_rms_dist(cand, gt)  # None if no match
            except Exception:
                rms = None
            if rms is None:
                continue
            rms_value = rms[0]
            if k == 0:
                matched_first = True
                rmse_first = rms_value
            matched_any = True
            if rmse_best is None or rms_value < rmse_best:
                rmse_best = rms_value
        return {
            "idx": idx, "N": N,
            "matched_first": matched_first, "matched_any": matched_any,
            "rmse_first": rmse_first, "rmse_best": rmse_best,
        }

    print("\nmatching against GT ...")
    t0 = time.time()
    results = p_map(evaluate_one, items)
    print(f"matching done in {time.time()-t0:.1f}s")

    # ---- optionally dump raw structures in DiffCSP CSP-eval format (K, B, ...) ----
    if args.save_raw is not None:
        items_idx = [it[0] for it in items]
        K = args.K
        Bn = len(items_idx)
        # Each comp has N atoms; build per-K stacks
        all_L = torch.zeros(K, Bn, 3, 3)
        all_num = torch.zeros(K, Bn, dtype=torch.long)
        F_parts = [[] for _ in range(K)]
        Z_parts = [[] for _ in range(K)]
        for b_pos, (idx, _, Z, N) in enumerate(items):
            samples = gen_per_comp[idx]
            for k in range(min(K, len(samples))):
                L_k, F_k = samples[k]
                all_L[k, b_pos] = L_k
                all_num[k, b_pos] = N
                F_parts[k].append(F_k)
                Z_parts[k].append(torch.tensor(Z, dtype=torch.long))
        all_F = torch.cat([torch.cat(p, dim=0) for p in F_parts], dim=0).view(K, -1, 3) \
                if False else None
        # Simpler: per-K cat then stack
        F_per_k = [torch.cat(p, dim=0) for p in F_parts]
        Z_per_k = [torch.cat(z, dim=0) for z in Z_parts]
        all_F = torch.stack(F_per_k, dim=0)
        all_Z = torch.stack(Z_per_k, dim=0)
        from argparse import Namespace
        out = {
            "eval_setting": Namespace(model_path=args.ckpt, dataset="mp_20", K=K),
            "lattices": all_L,
            "frac_coords": all_F,
            "atom_types": all_Z,
            "num_atoms": all_num,
        }
        Path(args.save_raw).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, args.save_raw)
        print(f"saved raw (K={K}, B={Bn}) -> {args.save_raw}")

    # aggregate
    n = len(results)
    n_first = sum(r["matched_first"] for r in results)
    n_any = sum(r["matched_any"] for r in results)
    rmses_first = [r["rmse_first"] for r in results if r["rmse_first"] is not None]
    rmses_best = [r["rmse_best"] for r in results if r["rmse_best"] is not None]

    metrics = {
        "n_compositions": n,
        "K": args.K,
        "match_rate_at_1": n_first / max(1, n),
        f"match_rate_at_{args.K}": n_any / max(1, n),
        "rmse_at_1_mean": float(np.mean(rmses_first)) if rmses_first else None,
        f"rmse_at_{args.K}_mean": float(np.mean(rmses_best)) if rmses_best else None,
    }

    # per-N
    by_N_results = defaultdict(list)
    for r in results:
        by_N_results[r["N"]].append(r)
    per_N = {}
    for Nk, rs in sorted(by_N_results.items()):
        per_N[Nk] = {
            "n_comps": len(rs),
            "match_rate_at_1": sum(r["matched_first"] for r in rs) / max(1, len(rs)),
            f"match_rate_at_{args.K}": sum(r["matched_any"] for r in rs) / max(1, len(rs)),
        }
    metrics["per_N"] = per_N

    print("\n=== CSP METRICS ===")
    import json
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_N"}, indent=2, default=str))
    print("\nper-N (first 8 rows):")
    for Nk in list(per_N)[:8]:
        d = per_N[Nk]
        print(f"  N={Nk:3d} (n={d['n_comps']:4d}): @1={d['match_rate_at_1']:.3f}  @{args.K}={d[f'match_rate_at_{args.K}']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
