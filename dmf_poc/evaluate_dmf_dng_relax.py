"""DNG with CHGNet-based Stability PROXY (no MP phase diagram).

Pipeline:
  1. Sample N_GEN compositions from MP-20 train empirical (N, Z-tuple) prior.
  2. Generate one DMF structure per composition with the given ckpt.
  3. CHGNet-relax each gen structure (max_steps, fmax threshold).
  4. CHGNet-relax one matching train structure per composition for energy baseline.
  5. Compute:
     - Validity (V): structure built and pymatgen-valid.
     - Convergence: CHGNet relax `fmax < FMAX_THRESH` within max_steps.
     - Stability proxy (S~): converged AND final E within Δ eV/atom of train-baseline.
     - Uniqueness | S~: pairwise unique within same composition via StructureMatcher
       on RELAXED structures.
     - Novelty | S~ · U: not matching any train (same composition) on relaxed.

S~ is a proxy: real S.U.N. would need MP convex hull (E_above_hull).
We report S~ relative to the per-composition train baseline energy.

Output: cache/dng_relax_<name>.json
"""
from __future__ import annotations
import argparse, json, sys, time
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
    """Returns dict: converged, final_E_per_atom, final_fmax, n_steps, relaxed_struct."""
    if struct is None:
        return {"valid": False, "converged": False, "E": None,
                "fmax_final": None, "n_steps": 0, "relaxed": None}
    try:
        r = relaxer.relax(struct, steps=max_steps, fmax=fmax, verbose=False)
        # chgnet's relax returns trajectory with forces & energies
        traj = r["trajectory"]
        relaxed = r["final_structure"]
        E_total = float(traj.energies[-1])
        E_per = E_total / len(struct)
        # final fmax from last forces
        last_forces = np.asarray(traj.forces[-1])
        fmax_final = float(np.linalg.norm(last_forces, axis=-1).max())
        converged = fmax_final < fmax
        return {"valid": True, "converged": converged, "E": E_per,
                "fmax_final": fmax_final, "n_steps": len(traj.energies),
                "relaxed": relaxed}
    except Exception as e:
        return {"valid": False, "converged": False, "E": None,
                "fmax_final": None, "n_steps": 0, "relaxed": None,
                "error": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_csv", default=str(Path.home() / "diffcsp/data/mp_20/train.csv"))
    ap.add_argument("--n_gen", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--fmax", type=float, default=0.1)  # eV/Å — standard threshold
    ap.add_argument("--stable_dE", type=float, default=0.3,  # eV/atom margin vs train baseline
                    help="gen is 'stable-proxy' if relax converged AND |E_gen - E_train_baseline| <= stable_dE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--film", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 1. Parse train
    print(f"parsing train CIFs ...")
    df = pd.read_csv(args.train_csv)
    parsed = p_map(parse_train_cif, df["cif"].tolist())
    train_items = [p for p in parsed if p is not None]
    train_by_comp = defaultdict(list)
    for s, Z, N in train_items:
        train_by_comp[Z].append(s)
    comp_keys = list(train_by_comp.keys())
    comp_counts = np.array([len(train_by_comp[c]) for c in comp_keys], dtype=np.float64)
    comp_probs = comp_counts / comp_counts.sum()
    print(f"  {len(train_items)} train structs, {len(comp_keys)} unique compositions")

    # 2. Sample compositions
    sampled_idx = rng.choice(len(comp_keys), size=args.n_gen, p=comp_probs, replace=True)
    sampled_Z = [comp_keys[i] for i in sampled_idx]
    print(f"  sampled {args.n_gen}; N hist: {Counter(len(z) for z in sampled_Z)}")

    # 3. Generate
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
    print(f"  generation done in {time.time()-t0:.1f}s")

    # 4. Relax (gen first; pick one train per composition for baseline)
    from chgnet.model import CHGNet, StructOptimizer
    print("loading CHGNet ...")
    relaxer = StructOptimizer(model=CHGNet.load())

    # Train baseline per composition: pick first train struct per sampled composition,
    # relax it once; reuse for all gen with same composition.
    unique_sampled_Z = list(set(sampled_Z))
    print(f"\nrelaxing {len(unique_sampled_Z)} train baselines (one per unique composition) ...")
    t0 = time.time()
    train_baseline_E = {}
    for j, Z in enumerate(unique_sampled_Z):
        r = relax_one(train_by_comp[Z][0], relaxer, args.max_steps, args.fmax)
        train_baseline_E[Z] = r["E"] if r["converged"] else None
        if (j + 1) % 20 == 0:
            print(f"  {j+1}/{len(unique_sampled_Z)}  ({time.time()-t0:.1f}s)", flush=True)
    n_baseline_ok = sum(1 for v in train_baseline_E.values() if v is not None)
    print(f"  train baselines converged: {n_baseline_ok}/{len(unique_sampled_Z)}")

    print(f"\nrelaxing {args.n_gen} generated structures ...")
    t0 = time.time()
    gen_results = []
    for i, s in enumerate(gen_structs):
        r = relax_one(s, relaxer, args.max_steps, args.fmax)
        gen_results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n_gen}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  done in {time.time()-t0:.1f}s")

    # 5. Compute metrics
    n_valid = sum(1 for r in gen_results if r["valid"])
    n_converged = sum(1 for r in gen_results if r["converged"])

    # Stability proxy: converged + within ΔE of train baseline (per same composition)
    stable_flags = [False] * args.n_gen
    for i, r in enumerate(gen_results):
        Z = sampled_Z[i]
        base_E = train_baseline_E.get(Z)
        if r["converged"] and base_E is not None and r["E"] is not None:
            if abs(r["E"] - base_E) <= args.stable_dE:
                stable_flags[i] = True
    n_stable = sum(stable_flags)

    # Uniqueness among S~: pairwise on relaxed structures within same composition
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    unique_flags = [False] * args.n_gen
    by_comp_stable = defaultdict(list)  # Z -> list of (i, relaxed_struct)
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
                except Exception:
                    pass
            if not dup:
                unique_flags[i] = True
                kept.append((i, s))
    n_unique = sum(unique_flags)

    # Novelty: among unique-stable, fraction not matching any train of same composition
    novel_flags = [False] * args.n_gen
    for i in range(args.n_gen):
        if not unique_flags[i]:
            continue
        s_gen_relaxed = gen_results[i]["relaxed"]
        pool = train_by_comp[sampled_Z[i]]
        is_novel = True
        for ts in pool:
            try:
                if matcher.fit(s_gen_relaxed, ts):
                    is_novel = False; break
            except Exception:
                pass
        novel_flags[i] = is_novel
    n_novel = sum(novel_flags)

    out = {
        "n_gen": args.n_gen,
        "fmax_threshold": args.fmax,
        "stable_dE_threshold": args.stable_dE,
        "max_steps": args.max_steps,
        "counts": {
            "valid": n_valid, "converged": n_converged,
            "stable_proxy": n_stable, "unique_of_stable": n_unique,
            "novel_of_unique_stable": n_novel,
        },
        "rates": {
            "V": n_valid / args.n_gen,
            "converged": n_converged / args.n_gen,
            "S_proxy": n_stable / args.n_gen,
            "U_given_S": n_unique / max(1, n_stable),
            "Nv_given_US": n_novel / max(1, n_unique),
            "S_proxy_U_Nv": n_novel / args.n_gen,
        },
        "energies": {
            "gen_E_mean": float(np.mean([r["E"] for r in gen_results if r["E"] is not None])) if any(r["E"] is not None for r in gen_results) else None,
            "gen_fmax_mean": float(np.mean([r["fmax_final"] for r in gen_results if r["fmax_final"] is not None])),
            "train_baseline_n_ok": n_baseline_ok,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== DNG-relax summary ===")
    print(f"  Validity (V):                {100*out['rates']['V']:.1f}%")
    print(f"  CHGNet converged:            {100*out['rates']['converged']:.1f}%")
    print(f"  Stability proxy (S~):        {100*out['rates']['S_proxy']:.1f}%  (conv. + |ΔE| ≤ {args.stable_dE} eV/atom)")
    print(f"  U | S~:                      {100*out['rates']['U_given_S']:.1f}%")
    print(f"  Nv | S~·U:                   {100*out['rates']['Nv_given_US']:.1f}%")
    print(f"  S~·U·Nv / n_gen:             {100*out['rates']['S_proxy_U_Nv']:.1f}%")
    print(f"  Mean gen final force:        {out['energies']['gen_fmax_mean']:.3f} eV/Å")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
