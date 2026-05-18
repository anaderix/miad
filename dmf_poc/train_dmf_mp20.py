"""Task 4c — DMF training on real MP-20.

Wires together:
  - mp20_loader.same_N_iterator   (Task 4b)
  - CSPNetDMF                     (Task 3)
  - v_drift.compute_V             (Task 2)
  - friction-scaled MSE loss      (Task 4a)

Validation pass (the default for --max_iter 5000):
  - Loss decreases by >=20% from first 200 to last 200 iters (smoothed)
  - Training runs without NaN/Inf
  - Final F_hat in [0,1)^3

Full training pass: --max_iter 100000 (run in background).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

# allow direct script execution from any cwd
sys.path.insert(0, str(Path(__file__).parent))
from cspnet_dmf import CSPNetDMF, MAX_ATOMIC_NUM
from mp20_loader import load_mp20, same_N_iterator
from v_drift import compute_V, torus_diff
from mem_bank import CrystalMemoryBank


def make_one_hot_types(Z_batch: torch.Tensor) -> torch.Tensor:
    """Z_batch (B, N) of atomic numbers -> (B*N, MAX_ATOMIC_NUM) one-hot."""
    B, N = Z_batch.shape
    flat = Z_batch.reshape(-1).clamp(0, MAX_ATOMIC_NUM - 1)
    out = torch.zeros(flat.shape[0], MAX_ATOMIC_NUM, device=Z_batch.device)
    out.scatter_(1, flat.unsqueeze(1), 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path.home() / "diffcsp/data/mp_20/train.csv"))
    ap.add_argument("--cache", default=str(Path(__file__).parent / "cache/mp20_train.pt"))
    ap.add_argument("--max_iter", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda_frac", type=float, default=None,
                    help="weight of frac loss (None = auto from first batch)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_path", default=str(Path(__file__).parent / "cache/dmf_mp20_last.pt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lat_prior_mean", type=float, default=5.0)
    ap.add_argument("--lat_prior_std", type=float, default=1.0)
    ap.add_argument("--temps_L", default="0.5,1.0,2.0",
                    help="comma-separated lattice kernel temperatures")
    ap.add_argument("--temps_F", default="0.02,0.05,0.2",
                    help="comma-separated frac kernel temperatures")
    ap.add_argument("--n_steps", type=int, default=1,
                    help="K-step rollout per training iteration; loss is computed on final step output")
    ap.add_argument("--per_atom_z", action="store_true",
                    help="sample z per atom (N_total, latent) instead of per crystal (B, latent)")
    ap.add_argument("--film", action="store_true",
                    help="enable FiLM-style composition conditioning in each CSPLayer")
    ap.add_argument("--repulsion", type=float, default=0.0,
                    help="strength of anti-mode-seeking term in V; 0 = off, ~0.3-0.5 reasonable")
    ap.add_argument("--mem_bank", action="store_true",
                    help="enable per-N memory bank to augment V kernel positives")
    ap.add_argument("--mem_per_N", type=int, default=256,
                    help="max crystals per N in memory bank")
    ap.add_argument("--mem_extra", type=int, default=128,
                    help="extra positives sampled from bank per batch")
    ap.add_argument("--no_friction", action="store_true",
                    help="disable friction γ schedule; use direct drift target (as in reference DM impl)")
    ap.add_argument("--chem_temp", type=float, default=1e9,
                    help="chemistry weighting in V_frac kernel; large=off, small=strict. For Z: τ=4 → ΔZ±2 ≈0.5. For cov: τ=0.5 → Δr_cov±0.5Å ≈0.5.")
    ap.add_argument("--chem_metric", default="Z", choices=["Z", "cov"],
                    help="atom-pair chemistry distance: 'Z' (atomic number) or 'cov' (covalent radius lookup).")
    ap.add_argument("--chem_temp_per_N", default=None,
                    help="per-N chem_temp schedule, e.g. 'N<=5:2,N>=6:8'. Overrides --chem_temp.")
    ap.add_argument("--chem_temp_linear", default=None,
                    help="continuous τ(N) = a + b·N, format 'a,b'. e.g. '0,1' → τ=N. Overrides --chem_temp & --chem_temp_per_N.")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    torch.manual_seed(args.seed)

    # data
    print("loading MP-20 train ...")
    data = load_mp20(args.csv, niggli=True, primitive=False, cache=args.cache)
    print(f"  n_total = {data['n_total']}")

    # model
    model = CSPNetDMF(
        hidden_dim=512, latent_dim=256, num_layers=6, num_freqs=128,
        ln=True, ip=True, smooth=True, pred_type=False, film=args.film,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params = {n_params:,}")

    # infinite stream of batches
    def stream():
        seed = args.seed
        while True:
            for b in same_N_iterator(data, args.batch_size, seed=seed):
                yield b
            seed += 1

    temps_L = [float(x) for x in args.temps_L.split(",")]
    temps_F = [float(x) for x in args.temps_F.split(",")]
    print(f"  temps_L={temps_L}, temps_F={temps_F}")

    mem = CrystalMemoryBank(max_per_N=args.mem_per_N, device=dev) if args.mem_bank else None
    if mem is not None:
        print(f"  memory bank enabled: max_per_N={args.mem_per_N}, extra_samples={args.mem_extra}")
    loss_hist = []
    lambda_frac = args.lambda_frac
    t_start = time.time()
    it = 0
    model.train()

    for batch in stream():
        if it >= args.max_iter:
            break
        N = batch["N"]
        B = batch["B"]
        L_pos = batch["L"].to(dev)          # (B, 3, 3)
        F_pos = batch["F"].to(dev)          # (B, N, 3)
        Z = batch["Z"].to(dev)              # (B, N)

        # noise inputs
        if args.per_atom_z:
            z = torch.randn(B * N, model.latent_dim, device=dev)
        else:
            z = torch.randn(B, model.latent_dim, device=dev)
        frac_in = torch.rand(B * N, 3, device=dev)
        # Lattice prior: small random + mean lattice ~ 5*I (will be learned away)
        lat_in = torch.randn(B, 3, 3, device=dev) * args.lat_prior_std + torch.eye(3, device=dev) * args.lat_prior_mean

        num_atoms = torch.full((B,), N, device=dev, dtype=torch.long)
        node2graph = torch.arange(B, device=dev).repeat_interleave(N)
        atom_types = make_one_hot_types(Z)

        # K-step rollout; gradient flows through all forwards
        L_state, F_state = lat_in, frac_in
        for _ in range(args.n_steps):
            L_hat, F_hat = model(z, atom_types, F_state, L_state, num_atoms, node2graph)
            L_state, F_state = L_hat, F_hat
        F_hat_bn = F_hat.view(B, N, 3)

        with torch.no_grad():
            # memory bank: add this batch's GT, then sample extra positives
            L_aug, F_aug = L_pos, F_pos
            if mem is not None:
                mem.add(L_pos, F_pos, N)
                sampled = mem.sample(N, args.mem_extra)
                if sampled is not None:
                    L_mem, F_mem = sampled
                    L_aug = torch.cat([L_pos, L_mem], dim=0)
                    F_aug = torch.cat([F_pos, F_mem], dim=0)
            # Z_per_crystal for chemistry-aware kernel
            Z_gen = Z  # (B, N) — gen's compositions == current batch
            Z_pos = Z  # (B, N) — same compositions in target (current batch is the target neighborhood)
            # τ schedule overrides (continuous wins over step wins over scalar)
            chem_temp_eff = args.chem_temp
            if args.chem_temp_linear:
                a, b = [float(x) for x in args.chem_temp_linear.split(",")]
                chem_temp_eff = max(1e-3, a + b * N)
            elif args.chem_temp_per_N:
                lo_tau = hi_tau = None; cutoff = 5
                for rule in args.chem_temp_per_N.split(","):
                    op, tau = rule.strip().split(":")
                    tau = float(tau)
                    if "<=" in op:
                        cutoff = int(op.split("<=")[1]); lo_tau = tau
                    elif ">=" in op:
                        hi_tau = tau
                if lo_tau is not None and hi_tau is not None:
                    chem_temp_eff = lo_tau if N <= cutoff else hi_tau
            V_L, V_F = compute_V(L_hat.detach(), F_hat_bn.detach(), L_aug, F_aug,
                                 temperatures_L=temps_L, temperatures_F=temps_F,
                                 repulsion=args.repulsion,
                                 Z_gen=Z_gen, Z_pos=Z_pos,
                                 chem_temp=chem_temp_eff,
                                 chem_metric=args.chem_metric)
            gamma = 0.0 if args.no_friction else it / max(1, args.max_iter - 1)
            target_L = L_hat.detach() + (1.0 - gamma) * V_L
            target_F = (F_hat_bn.detach() + (1.0 - gamma) * V_F) % 1.0

        loss_L = (L_hat - target_L).pow(2).mean()
        loss_F = torus_diff(F_hat_bn, target_F).pow(2).mean()

        # auto-calibrate lambda from the first batch's raw loss magnitudes
        if lambda_frac is None:
            with torch.no_grad():
                lambda_frac = float(loss_L.detach() / loss_F.detach().clamp_min(1e-8))
            print(f"  auto-calibrated lambda_frac = {lambda_frac:.3f}")

        loss = loss_L + lambda_frac * loss_F
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        loss_hist.append(float(loss.detach()))

        if it % args.log_every == 0 or it == args.max_iter - 1:
            dt = time.time() - t_start
            print(f"iter {it:6d} | loss={loss.item():.4f}  loss_L={loss_L.item():.4f}  "
                  f"loss_F={loss_F.item():.5f}  γ={gamma:.3f}  N={N} B={B}  "
                  f"throughput={(it+1)/max(dt,1e-3):.1f} it/s")
        it += 1

    # save checkpoint
    Path(args.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "lambda_frac": lambda_frac,
        "n_iters": it,
        "loss_hist": loss_hist,
        "args": vars(args),
    }, args.ckpt_path)
    print(f"\nsaved -> {args.ckpt_path}")

    # ---------- success criteria ----------
    def smooth(xs, w):
        return sum(xs[-w:]) / max(1, min(w, len(xs)))

    L_start = sum(loss_hist[:200]) / max(1, min(200, len(loss_hist)))
    L_end = smooth(loss_hist, 200)
    finite = all(torch.isfinite(torch.tensor(loss_hist)).tolist())
    f_min = float(F_hat_bn.min())
    f_max = float(F_hat_bn.max())

    print("\n=== success criteria ===")
    s1 = L_end < 0.8 * L_start
    s2 = finite
    s3 = (0 <= f_min) and (f_max < 1)
    print(f"  S1 loss drop >=20%:  start={L_start:.4f} end={L_end:.4f}  {'PASS' if s1 else 'FAIL'}")
    print(f"  S2 no NaN/Inf:       {'PASS' if s2 else 'FAIL'}")
    print(f"  S3 F_hat in [0,1):   min={f_min:.4f} max={f_max:.4f}  {'PASS' if s3 else 'FAIL'}")

    ok = s1 and s2 and s3
    print(f"\n{'ALL CRITERIA PASS' if ok else 'SOME CRITERIA FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
