"""Task 4a — Synthetic training of CSPNetDMF with V-drift + friction loss.

Goal: validate that the training loop *mechanics* work — loss decreases,
drift norm decreases, samples converge to the synthetic target distribution.
This is independent of MP-20 data loading (which is Task 4b).

Setup:
- Single fixed composition: 4 atoms of one element (Z=14 = Si)
- Synthetic target: lattices ~ 5*I + N(0, 0.3) ; fracs ~ N(0.5, 0.05) mod 1
- 2000 iterations, batch=64, friction γ_t linear 0->1
- DMF input: z_graph ~ N(0,1)^256, frac_in ~ Uniform[0,1), lattices_in ~ N(0,1)+5*I

Success criteria:
  S1. Smoothed loss at end is < 50% of smoothed loss in first 100 iters.
  S2. Convergence: |trace(L̂_final) - target_trace| < 25% of |trace(L̂_iter0) - target_trace|.
  S3. Generated lattice mean trace converges toward 15 (= trace(5*I)).
  S4. Generated frac mean converges toward 0.5 (with torus tolerance).
  S5. Training runs without NaN/Inf.
"""

import sys
from pathlib import Path

import torch
from torch import nn

from cspnet_dmf import CSPNetDMF, MAX_ATOMIC_NUM
from v_drift import compute_V, torus_diff


def make_target_batch(B, N, device):
    """Synthetic GT: tight cluster of crystals near a known mode."""
    L = torch.eye(3, device=device).expand(B, 3, 3) * 5.0 \
        + torch.randn(B, 3, 3, device=device) * 0.3
    F = (torch.full((B, N, 3), 0.5, device=device)
         + torch.randn(B, N, 3, device=device) * 0.05) % 1.0
    return L, F


def make_atom_types(B, N, Z, device):
    """One-hot atom_types for B crystals each of N atoms all element Z."""
    types_idx = torch.full((B * N,), Z, device=device, dtype=torch.long)
    types = torch.zeros(B * N, MAX_ATOMIC_NUM, device=device)
    types.scatter_(1, types_idx.unsqueeze(1), 1.0)
    return types


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    torch.manual_seed(42)

    # config
    B, N = 64, 4
    Z_si = 14
    MAX_ITER = 2000
    LR = 5e-4
    LAMBDA_FRAC = 1.0  # weight of frac loss vs lattice loss

    model = CSPNetDMF(
        hidden_dim=512, latent_dim=256, num_layers=6, num_freqs=128,
        ln=True, ip=True, smooth=True, pred_type=False,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    num_atoms = torch.full((B,), N, device=dev, dtype=torch.long)
    node2graph = torch.cat([torch.full((N,), i, device=dev, dtype=torch.long) for i in range(B)])
    atom_types = make_atom_types(B, N, Z_si, dev)

    loss_hist = []
    drift_hist = []
    diag_iter = []
    diag_meanL_trace = []
    diag_meanF = []

    model.train()
    for it in range(MAX_ITER):
        # target batch (re-sample every step so the model sees the full target dist)
        L_pos, F_pos = make_target_batch(B, N, dev)

        # noise inputs
        z = torch.randn(B, model.latent_dim, device=dev)
        frac_in = torch.rand(B * N, 3, device=dev)
        lat_in = torch.randn(B, 3, 3, device=dev) + torch.eye(3, device=dev) * 5.0

        L_hat, F_hat = model(z, atom_types, frac_in, lat_in, num_atoms, node2graph)

        with torch.no_grad():
            # reshape F to (B, N, 3) for V
            F_hat_bn = F_hat.view(B, N, 3)
            V_L, V_F = compute_V(L_hat.detach(), F_hat_bn.detach(), L_pos, F_pos)

            gamma = it / (MAX_ITER - 1)
            V_L_eff = (1.0 - gamma) * V_L
            V_F_eff = (1.0 - gamma) * V_F

            target_L = L_hat.detach() + V_L_eff
            target_F = (F_hat_bn.detach() + V_F_eff) % 1.0

        # loss: MSE on lattice; torus-MSE on frac
        loss_L = (L_hat - target_L).pow(2).mean()
        # torus loss: use torus_diff to compute shortest difference, then squared
        frac_diff = torus_diff(F_hat_bn, target_F)
        loss_F = frac_diff.pow(2).mean()
        loss = loss_L + LAMBDA_FRAC * loss_F

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it}: {loss.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip to avoid blow-up early
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        loss_hist.append(loss.item())
        # drift norm: sum of norms over modalities
        drift_hist.append(float(V_L.pow(2).mean().sqrt() + V_F.pow(2).mean().sqrt()))

        if it % 200 == 0 or it == MAX_ITER - 1:
            with torch.no_grad():
                mean_trace = float(L_hat.diagonal(dim1=-2, dim2=-1).sum(-1).mean())
                mean_F = float(F_hat_bn.mean())
            diag_iter.append(it)
            diag_meanL_trace.append(mean_trace)
            diag_meanF.append(mean_F)
            print(f"iter {it:5d} | loss={loss.item():.5f}  loss_L={loss_L.item():.5f}  "
                  f"loss_F={loss_F.item():.5f}  drift={drift_hist[-1]:.4f}  "
                  f"γ={gamma:.3f}  mean_trace(L̂)={mean_trace:.3f}  mean(F̂)={mean_F:.3f}")

    # ---------- evaluate success criteria ----------
    print("\n=== success criteria ===")

    def smooth(xs, w=100):
        return sum(xs[-w:]) / max(1, len(xs[-w:]))

    L0 = sum(loss_hist[:100]) / 100
    L1 = smooth(loss_hist, 100)
    final_trace = diag_meanL_trace[-1]
    initial_trace = diag_meanL_trace[0]
    final_F = diag_meanF[-1]

    target_trace = 15.0  # trace(5*I) = 15
    target_F_mean = 0.5

    s1 = L1 < 0.5 * L0
    # S2: convergence ratio. Initial output must be far, final must be close.
    initial_dist = abs(initial_trace - target_trace)
    final_dist = abs(final_trace - target_trace)
    s2 = (initial_dist > 5.0) and (final_dist < 0.25 * initial_dist)
    s3 = abs(final_trace - target_trace) < 5.0
    # frac target on torus — allow tolerance both via mean and via toroidal mod
    s4 = min(abs(final_F - target_F_mean), abs(final_F + 1 - target_F_mean), abs(final_F - 1 - target_F_mean)) < 0.15
    s5 = all(torch.isfinite(torch.tensor(loss_hist)).tolist())

    print(f"  S1 loss drop:       L0={L0:.4f} -> L1={L1:.4f}  (need L1 < 0.5*L0={0.5*L0:.4f})  {'PASS' if s1 else 'FAIL'}")
    print(f"  S2 convergence:     |dist iter0|={initial_dist:.2f} -> |dist final|={final_dist:.2f}  (need >5 init, <25% final)  {'PASS' if s2 else 'FAIL'}")
    print(f"  S3 trace(L̂)→15:     final={final_trace:.3f}  (need |Δ|<5)  {'PASS' if s3 else 'FAIL'}")
    print(f"  S4 mean(F̂)→0.5:     final={final_F:.3f}  (torus tol 0.15)  {'PASS' if s4 else 'FAIL'}")
    print(f"  S5 finite loss:     {'PASS' if s5 else 'FAIL'}")

    all_pass = s1 and s2 and s3 and s4 and s5
    print(f"\n{'ALL TESTS PASS' if all_pass else 'SOME CRITERIA FAILED'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
