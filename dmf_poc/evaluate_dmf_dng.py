"""DNG-lite evaluation for chem8: composition sampled from MP-20-train prior.

Pipeline:
  1. Build empirical composition prior from train.csv: P(N, sorted(Z)).
  2. Sample N_GEN compositions from that prior (with replacement).
  3. Generate one DMF structure per sampled composition with the chem8 ckpt.
  4. Compute:
     - Validity: pymatgen accepts the structure and atoms don't overlap.
     - Pairwise Uniqueness (U): among VALID gen samples, fraction unique under
       StructureMatcher (comparing only same-composition pairs to keep O(N²) tractable).
     - Train-Novelty (N): among VALID gen samples, fraction NOT matching any train
       structure of the same composition.

Stability (the "S" in S.U.N.) requires CHGNet/eq-V2 + MP phase-diagram pickle
and is NOT computed here — see backlog.

Output: cache/dng_<name>.json with aggregate metrics + per-composition flags.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

sys.path.insert(0, str(Path(__file__).parent))
from cspnet_dmf import CSPNetDMF, MAX_ATOMIC_NUM


def parse_train_cif(cif: str):
    try:
        s = Structure.from_str(cif, fmt="cif").get_reduced_structure()
        Z = tuple(sorted(site.specie.Z for site in s.sites))
        return s, Z, len(Z)
    except Exception:
        return None


def make_one_hot(Z_list_of_lists, device):
    flat = torch.cat([torch.as_tensor(z, device=device, dtype=torch.long) for z in Z_list_of_lists])
    flat = flat.clamp(0, MAX_ATOMIC_NUM - 1)
    out = torch.zeros(flat.shape[0], MAX_ATOMIC_NUM, device=device)
    out.scatter_(1, flat.unsqueeze(1), 1.0)
    return out


def build_structure(L, F, Z):
    try:
        s = Structure(Lattice(L.cpu().numpy()), [int(z) for z in Z],
                      F.cpu().numpy(), coords_are_cartesian=False)
        return s
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / "tmp/dmf_ckpts/dmf_chem8_50k.pt"))
    ap.add_argument("--train_csv", default=str(Path.home() / "diffcsp/data/mp_20/train.csv"))
    ap.add_argument("--n_gen", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=str(Path(__file__).parent / "cache/dng_chem8.json"))
    ap.add_argument("--film", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- 1. Parse train ----
    print(f"parsing train CIFs from {args.train_csv} ...")
    df = pd.read_csv(args.train_csv)
    parsed = p_map(parse_train_cif, df["cif"].tolist())
    train_items = [p for p in parsed if p is not None]
    print(f"  parsed {len(train_items)} train structures")
    train_by_comp: dict[tuple, list[Structure]] = defaultdict(list)
    for s, Z, N in train_items:
        train_by_comp[Z].append(s)
    # build prior: list of (N, Z) with multiplicities = count
    comp_keys = list(train_by_comp.keys())
    comp_counts = np.array([len(train_by_comp[c]) for c in comp_keys], dtype=np.float64)
    comp_probs = comp_counts / comp_counts.sum()
    print(f"  unique compositions: {len(comp_keys)}")

    # ---- 2. Sample compositions ----
    sampled_idx = rng.choice(len(comp_keys), size=args.n_gen, p=comp_probs, replace=True)
    sampled_Z = [comp_keys[i] for i in sampled_idx]   # tuples sorted
    print(f"  sampled {args.n_gen} compositions; N hist: {Counter(len(z) for z in sampled_Z)}")

    # ---- 3. Generate ----
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = CSPNetDMF(
        hidden_dim=512, latent_dim=256, num_layers=6, num_freqs=128,
        ln=True, ip=True, smooth=True, pred_type=False, film=args.film,
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # group by N
    by_N = defaultdict(list)
    for i, Z in enumerate(sampled_Z):
        by_N[len(Z)].append((i, list(Z)))

    gen_structs = [None] * args.n_gen
    t0 = time.time()
    for N, group in sorted(by_N.items()):
        for start in range(0, len(group), args.batch_size):
            chunk = group[start:start + args.batch_size]
            B = len(chunk)
            Z_list = [Z for _, Z in chunk]
            atom_types = make_one_hot(Z_list, dev)
            num_atoms_t = torch.full((B,), N, device=dev, dtype=torch.long)
            node2graph = torch.arange(B, device=dev).repeat_interleave(N)
            z = torch.randn(B, model.latent_dim, device=dev)
            frac_in = torch.rand(B * N, 3, device=dev)
            lat_in = torch.randn(B, 3, 3, device=dev) + torch.eye(3, device=dev) * 5.0
            with torch.no_grad():
                L_hat, F_hat = model(z, atom_types, frac_in, lat_in, num_atoms_t, node2graph)
            F_hat_bn = F_hat.view(B, N, 3)
            for i_chunk, (i_global, Z) in enumerate(chunk):
                gen_structs[i_global] = build_structure(L_hat[i_chunk], F_hat_bn[i_chunk], Z)
        print(f"  N={N}: done ({len(group)} comps)  elapsed={time.time()-t0:.1f}s")

    # ---- 4. Evaluate ----
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

    # Validity: pymatgen built it (build_structure returned non-None)
    valid_mask = [s is not None for s in gen_structs]
    n_valid = sum(valid_mask)
    print(f"\nValidity: {n_valid}/{args.n_gen} = {100*n_valid/args.n_gen:.1f}%")

    # Group valid by composition for pairwise unique + novelty
    valid_by_comp = defaultdict(list)  # Z_tuple -> list of (gen_idx, struct)
    for i, (s, Z) in enumerate(zip(gen_structs, sampled_Z)):
        if s is not None:
            valid_by_comp[Z].append((i, s))

    # Uniqueness: per composition, mark first occurrence as unique, subsequent matching as dup
    unique_flags = [False] * args.n_gen
    print("\nUniqueness (pairwise within same composition) ...")
    t1 = time.time()
    for Z, items in valid_by_comp.items():
        kept = []  # list of (i, s) treated as canonical-unique within this comp
        for i, s in items:
            is_dup = False
            for _, s_keep in kept:
                try:
                    if matcher.fit(s, s_keep):
                        is_dup = True
                        break
                except Exception:
                    pass
            if not is_dup:
                unique_flags[i] = True
                kept.append((i, s))
    n_unique = sum(unique_flags)
    print(f"  unique among valid: {n_unique}/{n_valid} = "
          f"{100*n_unique/max(1,n_valid):.1f}%  ({time.time()-t1:.1f}s)")

    # Novelty: among unique-valid, fraction not matching ANY train structure with same composition
    novel_flags = [False] * args.n_gen
    print("\nNovelty (vs train, same composition) ...")
    t2 = time.time()
    n_to_check = sum(unique_flags)
    checked = 0
    for Z, items in valid_by_comp.items():
        train_pool = train_by_comp.get(Z, [])
        for i, s in items:
            if not unique_flags[i]:
                continue
            checked += 1
            is_novel = True
            for ts in train_pool:
                try:
                    if matcher.fit(s, ts):
                        is_novel = False
                        break
                except Exception:
                    pass
            novel_flags[i] = is_novel
        if checked % 100 < len(items):
            print(f"  progress: {checked}/{n_to_check}  ({time.time()-t2:.1f}s)", flush=True)
    n_novel = sum(novel_flags)
    print(f"  novel among unique-valid: {n_novel}/{n_unique} = "
          f"{100*n_novel/max(1,n_unique):.1f}%  ({time.time()-t2:.1f}s)")

    # Save
    out = {
        "n_gen": args.n_gen,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_novel": n_novel,
        "rate_valid": n_valid / args.n_gen,
        "rate_unique_of_valid": n_unique / max(1, n_valid),
        "rate_novel_of_unique": n_novel / max(1, n_unique),
        "rate_unique_of_total": n_unique / args.n_gen,
        "rate_novel_of_total": n_novel / args.n_gen,
        "per_N": {},
    }
    by_N_flags = defaultdict(lambda: {"n": 0, "valid": 0, "unique": 0, "novel": 0})
    for i, Z in enumerate(sampled_Z):
        d = by_N_flags[len(Z)]
        d["n"] += 1
        d["valid"] += int(valid_mask[i])
        d["unique"] += int(unique_flags[i])
        d["novel"] += int(novel_flags[i])
    for N, d in sorted(by_N_flags.items()):
        out["per_N"][str(N)] = d

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== DNG-lite summary ===")
    print(f"  Validity (V):              {100*out['rate_valid']:.1f}%")
    print(f"  Uniqueness | Valid (U):    {100*out['rate_unique_of_valid']:.1f}%")
    print(f"  Novelty | Unique (Nv):     {100*out['rate_novel_of_unique']:.1f}%")
    print(f"  Combined V·U·Nv / n_gen:   {100*out['rate_novel_of_total']:.1f}%")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
