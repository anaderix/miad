"""DNG with REAL E_above_hull via MP convex hull (PatchedPhaseDiagram).

Like evaluate_dmf_dng_relax.py but replaces the ΔE-vs-train-baseline proxy
with real E_above_hull from the 2023-02-07 MP phase diagram.

Stability (S) := converged AND E_above_hull ≤ ehull_threshold (default 0.08 eV/atom).

Output: cache/dng_ehull_<name>.json with S, U|S, Nv|US, S·U·Nv.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings, pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from cspnet_dmf import CSPNetDMF, MAX_ATOMIC_NUM


def parse_train_cif(cif: str):
    try:
        s = Structure.from_str(cif, fmt="cif").get_reduced_structure()
        Z = tuple(sorted(site.specie.Z for site in s.sites))
        return s, Z, len(Z)
    except Exception:
        return None


def make_one_hot(Z_list, dev):
    flat = torch.cat([torch.as_tensor(z, device=dev, dtype=torch.long) for z in Z_list])
    flat = flat.clamp(0, MAX_ATOMIC_NUM - 1)
    out = torch.zeros(flat.shape[0], MAX_ATOMIC_NUM, device=dev)
    out.scatter_(1, flat.unsqueeze(1), 1.0)
    return out


def build_structure(L, F, Z):
    try:
        return Structure(Lattice(L.cpu().numpy()), [int(z) for z in Z],
                         F.cpu().numpy(), coords_are_cartesian=False)
    except Exception:
        return None


def relax_one(struct, relaxer, max_steps, fmax):
    if struct is None:
        return {"valid": False, "converged": False, "E": None,
                "fmax_final": None, "n_steps": 0, "relaxed": None}
    try:
        r = relaxer.relax(struct, steps=max_steps, fmax=fmax, verbose=False)
        traj = r["trajectory"]; relaxed = r["final_structure"]
        E_total = float(traj.energies[-1]); E_per = E_total / len(struct)
        last_forces = np.asarray(traj.forces[-1])
        fmax_final = float(np.linalg.norm(last_forces, axis=-1).max())
        return {"valid": True, "converged": fmax_final < fmax, "E": E_per,
                "fmax_final": fmax_final, "n_steps": len(traj.energies),
                "relaxed": relaxed}
    except Exception as e:
        return {"valid": False, "converged": False, "E": None,
                "fmax_final": None, "n_steps": 0, "relaxed": None,
                "error": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ppd_path", default=str(Path.home() / "miad/datasets/energy_hulls/2023-02-07-ppd-mp.pkl"))
    ap.add_argument("--train_cache", default=str(Path(__file__).parent / "cache/mp20_train.pt"))
    ap.add_argument("--n_gen", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--fmax", type=float, default=0.1)
    ap.add_argument("--ehull_threshold", type=float, default=0.08,
                    help="E_above_hull threshold for stability (eV/atom), MiAD-paper standard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--film", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 1. Load train cache for composition prior + novelty pool
    print(f"loading train cache {args.train_cache}")
    cache = torch.load(args.train_cache, weights_only=False)
    train_by_comp = defaultdict(list)
    for N, items in cache["by_N"].items():
        for it in items:
            L = it["L"].cpu().numpy() if hasattr(it["L"], "cpu") else it["L"]
            F = it["F"].cpu().numpy() if hasattr(it["F"], "cpu") else it["F"]
            Z = it["Z"].cpu().numpy().tolist() if hasattr(it["Z"], "cpu") else list(it["Z"])
            Z_int = [int(z) for z in Z]
            try:
                s = Structure(Lattice(L), Z_int, F, coords_are_cartesian=False)
                train_by_comp[tuple(sorted(Z_int))].append(s)
            except Exception:
                pass
    print(f"  {sum(len(v) for v in train_by_comp.values())} train structs, "
          f"{len(train_by_comp)} unique compositions")

    comp_keys = list(train_by_comp.keys())
    comp_counts = np.array([len(train_by_comp[c]) for c in comp_keys], dtype=np.float64)
    comp_probs = comp_counts / comp_counts.sum()

    # 2. Sample compositions
    sampled_idx = rng.choice(len(comp_keys), size=args.n_gen, p=comp_probs, replace=True)
    sampled_Z = [comp_keys[i] for i in sampled_idx]
    print(f"  sampled {args.n_gen}; N hist: {Counter(len(z) for z in sampled_Z)}")

    # 3. Load model + generate
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = CSPNetDMF(hidden_dim=512, latent_dim=256, num_layers=6, num_freqs=128,
                     ln=True, ip=True, smooth=True, pred_type=False, film=args.film).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    by_N = defaultdict(list)
    for i, Z in enumerate(sampled_Z):
        by_N[len(Z)].append((i, list(Z)))
    gen_structs = [None] * args.n_gen
    t0 = time.time()
    for N, group in sorted(by_N.items()):
        for start in range(0, len(group), 64):
            chunk = group[start:start + 64]
            B = len(chunk); Z_list = [Z for _, Z in chunk]
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
    print(f"  generation done in {time.time()-t0:.1f}s")

    # 4. Load phase diagram
    print(f"loading PPD from {args.ppd_path} ...")
    t0 = time.time()
    with open(args.ppd_path, "rb") as f:
        ppd = pickle.load(f)
    print(f"  loaded in {time.time()-t0:.1f}s, {len(ppd.all_entries)} entries")

    # 5. Load CHGNet
    from chgnet.model import CHGNet, StructOptimizer
    relaxer = StructOptimizer(model=CHGNet.load())

    # 6. Relax + compute E_above_hull
    print(f"\nrelaxing {args.n_gen} generated structures + computing E_above_hull ...")
    t0 = time.time()
    gen_results = []
    for i, s in enumerate(gen_structs):
        r = relax_one(s, relaxer, args.max_steps, args.fmax)
        # compute E_above_hull
        ehull = None
        if r["converged"] and r["E"] is not None:
            try:
                e_hull_per_atom = ppd.get_hull_energy_per_atom(s.composition)
                ehull = r["E"] - e_hull_per_atom
            except Exception:
                ehull = None
        r["E_above_hull"] = ehull
        gen_results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n_gen}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  done in {time.time()-t0:.1f}s")

    # 7. Compute S / U / Nv
    n_valid = sum(1 for r in gen_results if r["valid"])
    n_converged = sum(1 for r in gen_results if r["converged"])
    n_have_ehull = sum(1 for r in gen_results if r.get("E_above_hull") is not None)

    stable_flags = [False] * args.n_gen
    for i, r in enumerate(gen_results):
        eh = r.get("E_above_hull")
        if r["converged"] and eh is not None and eh <= args.ehull_threshold:
            stable_flags[i] = True
    n_stable = sum(stable_flags)

    # U|S
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    unique_flags = [False] * args.n_gen
    by_comp_stable = defaultdict(list)
    for i in range(args.n_gen):
        if stable_flags[i]:
            by_comp_stable[sampled_Z[i]].append((i, gen_results[i]["relaxed"]))
    for Z, items in by_comp_stable.items():
        kept = []
        for i, s in items:
            dup = False
            for _, sk in kept:
                try:
                    if matcher.fit(s, sk):
                        dup = True; break
                except Exception: pass
            if not dup:
                unique_flags[i] = True; kept.append((i, s))
    n_unique = sum(unique_flags)

    # Nv|S·U
    novel_flags = [False] * args.n_gen
    for i in range(args.n_gen):
        if not unique_flags[i]: continue
        s_gen = gen_results[i]["relaxed"]
        pool = train_by_comp[sampled_Z[i]]
        is_novel = True
        for ts in pool:
            try:
                if matcher.fit(s_gen, ts):
                    is_novel = False; break
            except Exception: pass
        novel_flags[i] = is_novel
    n_novel = sum(novel_flags)

    # Per-N-stability breakdown
    per_N = defaultdict(lambda: {"n":0,"stable":0,"unique":0,"novel":0,"converged":0})
    for i, Z in enumerate(sampled_Z):
        d = per_N[len(Z)]
        d["n"] += 1
        d["converged"] += int(gen_results[i]["converged"])
        d["stable"] += int(stable_flags[i])
        d["unique"] += int(unique_flags[i])
        d["novel"] += int(novel_flags[i])

    out = {
        "n_gen": args.n_gen,
        "max_steps": args.max_steps,
        "fmax_threshold": args.fmax,
        "ehull_threshold": args.ehull_threshold,
        "ckpt": args.ckpt,
        "counts": {
            "valid": n_valid, "converged": n_converged,
            "have_ehull": n_have_ehull,
            "stable": n_stable, "unique_of_stable": n_unique,
            "novel_of_unique_stable": n_novel,
        },
        "rates": {
            "V": n_valid / args.n_gen,
            "converged": n_converged / args.n_gen,
            "S": n_stable / args.n_gen,
            "U_given_S": n_unique / max(1, n_stable),
            "Nv_given_US": n_novel / max(1, n_unique),
            "S_U_Nv": n_novel / args.n_gen,
        },
        "energies": {
            "gen_E_mean": float(np.mean([r["E"] for r in gen_results if r["E"] is not None]))
                if any(r["E"] is not None for r in gen_results) else None,
            "gen_fmax_mean": float(np.mean([r["fmax_final"] for r in gen_results
                                            if r["fmax_final"] is not None])),
            "ehull_mean_converged": float(np.mean([r["E_above_hull"] for r in gen_results
                                                    if r.get("E_above_hull") is not None]))
                if n_have_ehull > 0 else None,
        },
        "per_N": dict(per_N),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== DNG real-E_hull summary ===")
    print(f"  Validity (V):                {100*out['rates']['V']:.1f}%")
    print(f"  CHGNet converged:            {100*out['rates']['converged']:.1f}%")
    print(f"  Stability (S, E_hull ≤ {args.ehull_threshold}): {100*out['rates']['S']:.1f}%")
    print(f"  U | S:                       {100*out['rates']['U_given_S']:.1f}%")
    print(f"  Nv | S·U:                    {100*out['rates']['Nv_given_US']:.1f}%")
    print(f"  S·U·Nv / n_gen:              {100*out['rates']['S_U_Nv']:.1f}%")
    print(f"  Mean gen final force:        {out['energies']['gen_fmax_mean']:.3f} eV/Å")
    print(f"  Mean E_above_hull (conv):    {out['energies']['ehull_mean_converged']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
