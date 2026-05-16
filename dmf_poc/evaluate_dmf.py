"""Task 5 — DMF inference + DiffCSP-compatible serialization.

Loads `cache/dmf_mp20_50k.pt`, the MP-20 test set, generates `n_samples` total
crystals conditioned on test compositions (sampled with replacement for the
target count), and writes them to a `.pt` file in the same format as DiffCSP's
`eval_gen.pt`:

  {
    'eval_setting': argparse.Namespace,
    'frac_coords': (sum_N, 3) float32   — flat frac coords for all atoms
    'num_atoms':   (B,)      int64      — per-crystal N
    'atom_types':  (sum_N, MAX_ATOMIC_NUM) float32 — one-hot
    'lengths':     (B, 3)    float32    — a, b, c
    'angles':      (B, 3)    float32    — alpha, beta, gamma (degrees)
  }

Output file passes the same `baseline_metrics.py` we used for DiffCSP-mp_gen.
"""

from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from p_tqdm import p_map

sys.path.insert(0, str(Path(__file__).parent))
from cspnet_dmf import CSPNetDMF, MAX_ATOMIC_NUM


def _parse_test_cif(cif: str):
    try:
        s = Structure.from_str(cif, fmt="cif").get_reduced_structure()
        Z = [site.specie.Z for site in s.sites]
        return Z
    except Exception:
        return None


def lattice_matrix_to_lengths_angles(L: torch.Tensor):
    """L: (B, 3, 3) rows = lattice vectors -> (lengths (B,3), angles (B,3) in degrees)."""
    lengths = L.norm(dim=-1)                          # (B, 3)
    a = L[:, 0]; b = L[:, 1]; c = L[:, 2]
    def angle(u, v):
        cos = (u * v).sum(-1) / (u.norm(dim=-1).clamp_min(1e-8) * v.norm(dim=-1).clamp_min(1e-8))
        cos = cos.clamp(-1.0, 1.0)
        return torch.acos(cos) * (180.0 / torch.pi)
    alpha = angle(b, c)
    beta  = angle(a, c)
    gamma = angle(a, b)
    angles = torch.stack([alpha, beta, gamma], dim=-1)  # (B, 3)
    return lengths, angles


def make_one_hot_types(Z_list, device):
    """List of (N_i,) tensors -> flat (sum_N, MAX_ATOMIC_NUM) one-hot."""
    flat = torch.cat([torch.as_tensor(z, device=device, dtype=torch.long) for z in Z_list])
    flat = flat.clamp(0, MAX_ATOMIC_NUM - 1)
    out = torch.zeros(flat.shape[0], MAX_ATOMIC_NUM, device=device)
    out.scatter_(1, flat.unsqueeze(1), 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "cache/dmf_mp20_50k.pt"))
    ap.add_argument("--test_csv", default=str(Path.home() / "diffcsp/data/mp_20/test.csv"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "cache/eval_gen.pt"))
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--lat_prior_mean", type=float, default=5.0,
                    help="mean of lattice prior: I * lat_prior_mean")
    ap.add_argument("--lat_prior_std", type=float, default=1.0,
                    help="std of additive Gaussian on lat_in")
    ap.add_argument("--n_steps", type=int, default=1,
                    help="iterative forward passes (1=one-shot, K>1=feed prev output as next input)")
    ap.add_argument("--per_atom_z", action="store_true")
    ap.add_argument("--film", action="store_true")
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
    print(f"loaded ckpt iters={ck['n_iters']}")

    # load test compositions
    df = pd.read_csv(args.test_csv)
    print(f"parsing {len(df)} test CIFs ...")
    t0 = time.time()
    Zs = p_map(_parse_test_cif, df["cif"].tolist())
    ok = [z for z in Zs if z is not None]
    print(f"  parsed {len(ok)}/{len(df)} ok in {time.time()-t0:.1f}s")

    # group by N for efficient batched inference
    from collections import defaultdict
    by_N = defaultdict(list)
    for z in ok:
        by_N[len(z)].append(z)
    print(f"  N range: [{min(by_N)}, {max(by_N)}]")

    # weighted sampling: respect test-distribution of N
    rng = np.random.default_rng(args.seed)
    Ns = sorted(by_N.keys())
    weights = np.array([len(by_N[n]) for n in Ns], dtype=float)
    weights /= weights.sum()

    target = args.n_samples
    # per-crystal accumulators (we know how to truncate cleanly later)
    L_per_crystal = []        # list of (3,3) tensors
    F_per_crystal = []        # list of (N,3) tensors
    Z_per_crystal = []        # list of list-of-ints (composition)
    t0 = time.time()
    n_batches = 0
    while len(L_per_crystal) < target:
        N = int(rng.choice(Ns, p=weights))
        group = by_N[N]
        B = min(args.batch_size, max(2, len(group)))
        idx = rng.integers(0, len(group), size=B)
        Z_batch = [group[i] for i in idx]
        if args.per_atom_z:
            z = torch.randn(B * N, model.latent_dim, device=dev)
        else:
            z = torch.randn(B, model.latent_dim, device=dev)
        frac_in = torch.rand(B * N, 3, device=dev)
        lat_in = torch.randn(B, 3, 3, device=dev) * args.lat_prior_std + torch.eye(3, device=dev) * args.lat_prior_mean
        num_atoms_t = torch.full((B,), N, device=dev, dtype=torch.long)
        node2graph = torch.arange(B, device=dev).repeat_interleave(N)
        atom_types = make_one_hot_types(Z_batch, dev)
        with torch.no_grad():
            L_state, F_state = lat_in, frac_in
            for _ in range(args.n_steps):
                L_hat, F_hat = model(z, atom_types, F_state, L_state, num_atoms_t, node2graph)
                L_state, F_state = L_hat, F_hat
        F_per_batch = F_hat.view(B, N, 3).cpu()
        L_per_batch = L_hat.cpu()
        for i in range(B):
            if len(L_per_crystal) >= target:
                break
            L_per_crystal.append(L_per_batch[i])
            F_per_crystal.append(F_per_batch[i])
            Z_per_crystal.append(Z_batch[i])
        n_batches += 1
        if n_batches % 20 == 0:
            print(f"  generated {len(L_per_crystal)}/{target} crystals")

    print(f"  total wall {time.time()-t0:.1f}s, n_samples={len(L_per_crystal)}")

    # ---- stack tensors in DiffCSP-compatible flat format ----
    nums = torch.tensor([F.shape[0] for F in F_per_crystal], dtype=torch.long)
    L_cat = torch.stack(L_per_crystal, dim=0)               # (B, 3, 3)
    F_cat = torch.cat(F_per_crystal, dim=0)                 # (sum_N, 3)
    Z_flat_list = []
    for z in Z_per_crystal:
        Z_flat_list.extend(z)
    Z_flat = torch.tensor(Z_flat_list, dtype=torch.long)
    assert Z_flat.shape[0] == F_cat.shape[0]

    # lattice -> lengths, angles
    lengths, angles = lattice_matrix_to_lengths_angles(L_cat)
    # one-hot atom types in flat format
    atom_types = torch.zeros(Z_flat.shape[0], MAX_ATOMIC_NUM)
    atom_types.scatter_(1, Z_flat.clamp(0, MAX_ATOMIC_NUM - 1).unsqueeze(1), 1.0)

    out = {
        "eval_setting": Namespace(model_path=str(Path(args.ckpt).parent),
                                  dataset="mp_20", label="dmf_mp20"),
        "frac_coords": F_cat.float(),
        "num_atoms": nums.long(),
        "atom_types": atom_types.float(),
        "lengths": lengths.float(),
        "angles": angles.float(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\nsaved -> {args.out}")

    # ---------- success criteria ----------
    print("\n=== success criteria ===")
    finite = all(torch.isfinite(t).all() for t in [F_cat, lengths, angles])
    f_min, f_max = float(F_cat.min()), float(F_cat.max())
    f_ok = (0.0 <= f_min) and (f_max < 1.0001)  # small tolerance on upper bound from mod 1
    lengths_ok = (lengths.min() > 0.5) and (lengths.max() < 100.0)
    angles_ok = (angles.min() > 1.0) and (angles.max() < 179.0)
    shape_ok = (lengths.shape[0] == nums.shape[0] == target) and \
               (F_cat.shape[0] == atom_types.shape[0] == int(nums.sum()))

    print(f"  S1 finite values:    {'PASS' if finite else 'FAIL'}")
    print(f"  S2 F in [0,1):       min={f_min:.4f} max={f_max:.4f}  {'PASS' if f_ok else 'FAIL'}")
    print(f"  S3 lengths plausible: min={float(lengths.min()):.2f} max={float(lengths.max()):.2f}  {'PASS' if lengths_ok else 'FAIL'}")
    print(f"  S4 angles plausible: min={float(angles.min()):.2f} max={float(angles.max()):.2f}  {'PASS' if angles_ok else 'FAIL'}")
    print(f"  S5 shape consistency: {'PASS' if shape_ok else 'FAIL'}")

    ok_all = finite and f_ok and lengths_ok and angles_ok and shape_ok
    print(f"\n{'ALL CRITERIA PASS' if ok_all else 'SOME CRITERIA FAILED'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
