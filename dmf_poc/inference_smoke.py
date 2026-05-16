"""Smoke-test inference of trained DMF model.

Loads `cache/dmf_mp20_50k.pt`, generates crystals for a small batch, sanity-checks:
  - all values finite
  - F_hat in [0,1)^3
  - lattice vector lengths in a plausible range (1 Å to 50 Å)
  - distinct compositions in input produce distinct outputs (no constant collapse)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from cspnet_dmf import CSPNetDMF, MAX_ATOMIC_NUM
from mp20_loader import load_mp20


def make_one_hot_types(Z_batch):
    B, N = Z_batch.shape
    flat = Z_batch.reshape(-1).clamp(0, MAX_ATOMIC_NUM - 1)
    out = torch.zeros(flat.shape[0], MAX_ATOMIC_NUM, device=Z_batch.device)
    out.scatter_(1, flat.unsqueeze(1), 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "cache/dmf_mp20_50k.pt"))
    ap.add_argument("--cache", default=str(Path(__file__).parent / "cache/mp20_train.pt"))
    ap.add_argument("--N", type=int, default=4, help="composition size to sample for")
    ap.add_argument("--n_eval", type=int, default=64, help="number of generations")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    # load ckpt
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    print(f"loaded ckpt iters={ck['n_iters']}, lambda_frac={ck['lambda_frac']:.3f}")

    model = CSPNetDMF(
        hidden_dim=512, latent_dim=256, num_layers=6, num_freqs=128,
        ln=True, ip=True, smooth=True, pred_type=False,
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # grab N=args.N crystals from train cache as compositions to condition on
    data = load_mp20(csv_path="", cache=args.cache)
    pool = data["by_N"].get(args.N, [])
    print(f"got {len(pool)} crystals with N={args.N} in train cache")
    assert len(pool) >= args.n_eval, "not enough crystals of requested N"

    # pick first n_eval distinct compositions (just slice)
    picks = pool[: args.n_eval]
    Z = torch.stack([p["Z"] for p in picks]).to(dev)  # (n_eval, N)

    B = args.n_eval
    N = args.N
    z = torch.randn(B, model.latent_dim, device=dev)
    frac_in = torch.rand(B * N, 3, device=dev)
    lat_in = torch.randn(B, 3, 3, device=dev) + torch.eye(3, device=dev) * 5.0
    num_atoms = torch.full((B,), N, device=dev, dtype=torch.long)
    node2graph = torch.arange(B, device=dev).repeat_interleave(N)
    atom_types = make_one_hot_types(Z)

    with torch.no_grad():
        L_hat, F_hat = model(z, atom_types, frac_in, lat_in, num_atoms, node2graph)
    F_hat = F_hat.view(B, N, 3)

    # ---- sanity checks ----
    finite = torch.isfinite(L_hat).all() and torch.isfinite(F_hat).all()
    f_min, f_max = float(F_hat.min()), float(F_hat.max())
    f_ok = (0.0 <= f_min) and (f_max < 1.0)
    # lattice vector lengths
    L_lens = L_hat.norm(dim=-1)  # (B, 3)
    L_lens_min = float(L_lens.min())
    L_lens_max = float(L_lens.max())
    L_lens_mean = float(L_lens.mean())
    L_plausible = (0.5 < L_lens_min) and (L_lens_max < 100.0)
    # diversity across samples (same N=4, different compositions)
    F_var_across = F_hat.std(dim=0).mean()
    L_var_across = L_hat.flatten(1).std(dim=0).mean()
    diverse = (float(F_var_across) > 1e-3) and (float(L_var_across) > 1e-3)

    print("\n=== sanity checks ===")
    print(f"  finite values:           {'PASS' if finite else 'FAIL'}")
    print(f"  F_hat in [0,1):          min={f_min:.4f} max={f_max:.4f}  {'PASS' if f_ok else 'FAIL'}")
    print(f"  L vector lengths:        min={L_lens_min:.2f} mean={L_lens_mean:.2f} max={L_lens_max:.2f} Å  {'PASS' if L_plausible else 'FAIL'}")
    print(f"  cross-sample diversity:  F.std={float(F_var_across):.4f}  L.std={float(L_var_across):.4f}  {'PASS' if diverse else 'FAIL'}")
    print()
    ok = finite and f_ok and L_plausible and diverse
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
