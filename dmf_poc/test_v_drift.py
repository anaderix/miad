"""Sanity tests for v_drift.compute_V on (lattice, frac).

Acceptance criteria from plan task 2:
  T1. V -> 0 when generated batch == target batch.
  T2. V points toward a clearly biased target distribution.
  T3. Translation invariance on frac: shifting all gen+pos by same delta keeps
      ||V_F|| equal; V_L unchanged.
  T4. Numerical stability for small tau (no NaN/Inf).
  T5. Memory: B=256, N=20 fits comfortably (we report peak GPU mem if cuda).
"""

import time
import torch

from v_drift import compute_V, compute_V_frac, compute_V_lattice, torus_diff


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().abs().max())


def t1_identical_small_tau(device):
    """V -> 0 when gen == pos AND tau is small enough to make kernel one-hot at self.

    Note: at finite tau, V points toward batch centroid even when gen==pos. The
    V=0 property only holds in the tau->0 (peaked) limit. This validates the
    kernel construction itself, separate from temperature choice.
    """
    torch.manual_seed(0)
    B, N = 8, 6
    L = torch.randn(B, 3, 3, device=device)
    F = torch.rand(B, N, 3, device=device)
    V_L, V_F = compute_V(L, F, L.clone(), F.clone(),
                        temperatures_L=[1e-6], temperatures_F=[1e-8])
    assert _norm(V_L) < 1e-4, f"V_L not ~0: {_norm(V_L)}"
    assert _norm(V_F) < 1e-4, f"V_F not ~0: {_norm(V_F)}"
    print(f"  T1  PASS  ||V_L||_inf={_norm(V_L):.2e}  ||V_F||_inf={_norm(V_F):.2e}")


def t1b_iid_batch_mean(device):
    """For large batch, gen i.i.d. from same dist as pos -> batch-mean V ≈ 0."""
    torch.manual_seed(10)
    B, N = 256, 5
    L_gen = torch.randn(B, 3, 3, device=device)
    L_pos = torch.randn(B, 3, 3, device=device)
    F_gen = torch.rand(B, N, 3, device=device)
    F_pos = torch.rand(B, N, 3, device=device)
    V_L, V_F = compute_V(L_gen, F_gen, L_pos, F_pos)
    # mean drift over batch — should be small for matched distributions
    m_L = V_L.mean(dim=0).abs().max()
    m_F = V_F.mean(dim=0).abs().max()
    # tolerance: per-sample V is normalized to unit-ish magnitude, so 1/sqrt(B)
    # would be ~0.06 for B=256. Allow 0.2 as a loose bound.
    assert float(m_L) < 0.2, f"V_L batch mean too large: {float(m_L):.3f}"
    assert float(m_F) < 0.2, f"V_F batch mean too large: {float(m_F):.3f}"
    print(f"  T1b PASS batch-mean |V_L|={float(m_L):.3f}  |V_F|={float(m_F):.3f}  (<0.2)")


def t2_directional_lat(device):
    """V_L points from gen toward a single-mode target."""
    torch.manual_seed(1)
    B = 16
    # generated lattices near origin
    L_gen = torch.randn(B, 3, 3, device=device) * 0.1
    # target lattices all near +5*I
    L_pos = torch.eye(3, device=device).expand(B, 3, 3) * 5.0 \
            + torch.randn(B, 3, 3, device=device) * 0.05
    V_L = compute_V_lattice(L_gen, L_pos, [0.5, 1.0, 2.0])
    # mean V should have positive trace (pull toward +5*I)
    mean_trace = V_L.diagonal(dim1=-2, dim2=-1).sum(-1).mean()
    assert float(mean_trace) > 0, f"V_L trace not positive: {float(mean_trace):.3f}"
    print(f"  T2L PASS mean trace(V_L)={float(mean_trace):.3f}  (>0 expected)")


def t2_directional_frac(device):
    """V_F drift points toward target frac coords (torus-aware)."""
    torch.manual_seed(2)
    B, N = 16, 4
    # generated near 0.1
    F_gen = torch.full((B, N, 3), 0.1, device=device) + torch.rand(B, N, 3, device=device) * 0.02
    # target near 0.4 — within torus the shortest path is +0.3 (not -0.7)
    F_pos = torch.full((B, N, 3), 0.4, device=device) + torch.rand(B, N, 3, device=device) * 0.02
    V_F = compute_V_frac(F_gen, F_pos, [0.02, 0.05, 0.2])
    # mean V should be positive (+0.3 direction)
    mean_V = V_F.mean()
    assert float(mean_V) > 0, f"V_F mean not positive: {float(mean_V):.3f}"
    print(f"  T2F PASS mean(V_F)={float(mean_V):.3f}  (>0 expected, shortest tor path is +0.3)")


def t2_directional_frac_wraparound(device):
    """V_F respects shortest-path on torus: gen=0.05, pos=0.95 should drift NEGATIVE."""
    torch.manual_seed(3)
    B, N = 16, 4
    F_gen = torch.full((B, N, 3), 0.05, device=device) + torch.rand(B, N, 3, device=device) * 0.01
    F_pos = torch.full((B, N, 3), 0.95, device=device) + torch.rand(B, N, 3, device=device) * 0.01
    V_F = compute_V_frac(F_gen, F_pos, [0.02, 0.05, 0.2])
    mean_V = V_F.mean()
    # shortest tor path: 0.05 -> 0.95 is via -0.1 (i.e. wrap around), so V should be negative
    assert float(mean_V) < 0, f"V_F wraparound not negative: {float(mean_V):.3f}"
    print(f"  T2W PASS mean(V_F)={float(mean_V):.3f}  (<0 expected via wraparound)")


def t3_translation_invariance(device):
    """V_F under uniform delta-shift on both gen and pos has same MAGNITUDE."""
    torch.manual_seed(4)
    B, N = 16, 5
    F_gen = torch.rand(B, N, 3, device=device)
    F_pos = torch.rand(B, N, 3, device=device)
    V_base = compute_V_frac(F_gen, F_pos, [0.05])
    # shift everything by same delta
    delta = torch.rand(1, 1, 3, device=device)
    V_shift = compute_V_frac(
        (F_gen + delta) % 1.0,
        (F_pos + delta) % 1.0,
        [0.05],
    )
    diff = (V_base - V_shift).abs().max()
    assert float(diff) < 1e-5, f"translation invariance broken: max diff {float(diff):.2e}"
    print(f"  T3  PASS max|V_base - V_shift|={float(diff):.2e}")


def t4_small_tau_stability(device):
    """Tiny tau -> kernel becomes nearly one-hot. Must not NaN/Inf."""
    torch.manual_seed(5)
    B, N = 32, 8
    L_gen = torch.randn(B, 3, 3, device=device)
    F_gen = torch.rand(B, N, 3, device=device)
    L_pos = torch.randn(B, 3, 3, device=device)
    F_pos = torch.rand(B, N, 3, device=device)
    V_L, V_F = compute_V(
        L_gen, F_gen, L_pos, F_pos,
        temperatures_L=[1e-4],
        temperatures_F=[1e-4],
    )
    assert torch.isfinite(V_L).all(), "V_L has non-finite values"
    assert torch.isfinite(V_F).all(), "V_F has non-finite values"
    print(f"  T4  PASS finite for tau=1e-4")


def t5_memory_scale(device):
    """B=256, N=20 batch — measure peak memory if on cuda."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(6)
    B, N = 256, 20
    L_gen = torch.randn(B, 3, 3, device=device)
    F_gen = torch.rand(B, N, 3, device=device)
    L_pos = torch.randn(B, 3, 3, device=device)
    F_pos = torch.rand(B, N, 3, device=device)
    t0 = time.time()
    V_L, V_F = compute_V(L_gen, F_gen, L_pos, F_pos)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.time() - t0
    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
    print(f"  T5  PASS B=256 N=20  time={dt*1000:.1f} ms  peak={peak_mb}MB"
          if peak_mb else f"  T5  PASS B=256 N=20  time={dt*1000:.1f} ms (cpu)")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    t1_identical_small_tau(dev)
    t1b_iid_batch_mean(dev)
    t2_directional_lat(dev)
    t2_directional_frac(dev)
    t2_directional_frac_wraparound(dev)
    t3_translation_invariance(dev)
    t4_small_tau_stability(dev)
    t5_memory_scale(dev)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
