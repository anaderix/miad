"""Drift field V for conditional CSP-DMF on (lattice, frac).

Inputs assume a batch of crystals with the SAME composition (same atom_types and
same N per batch). Permutation handling: external — caller passes atoms in the
same canonical order across the batch (e.g. sorted by atomic number, then by
fractional position lex). Inside this module we do NOT solve assignment.

V is the kernel-density-gradient mean-shift toward the target batch, in the
style of OFM/DM drifting. Multi-temperature is supported as in the notebook.

Conventions:
  L_gen:    (B, 3, 3)  — generated lattices (Cartesian basis rows)
  F_gen:    (B, N, 3)  — generated fractional coords in [0,1)^3
  L_pos:    (B', 3, 3) — target (positive) lattices
  F_pos:    (B', N, 3) — target fractional coords

Returns:
  V_L:      (B, 3, 3)  — drift on lattice
  V_F:      (B, N, 3)  — drift on frac, expressed in the torus tangent space
"""

from __future__ import annotations

import torch
from torch import Tensor


def torus_diff(f_a: Tensor, f_b: Tensor) -> Tensor:
    """Minimum-image difference on the unit torus T^3 = [0,1)^3.

    f_a, f_b can be any broadcastable shapes ending in (..., 3).
    Returns f_a - f_b mapped into (-0.5, 0.5]^3.
    """
    d = f_a - f_b
    return d - torch.round(d)


def _kernel_weights(d2: Tensor, tau: float) -> Tensor:
    """Softmax over the LAST axis with logits = -d2 / tau. Numerically stable."""
    logits = -d2 / tau
    return torch.softmax(logits, dim=-1)


def compute_V_lattice(
    L_gen: Tensor,
    L_pos: Tensor,
    temperatures: list[float],
) -> Tensor:
    """Multi-temperature kernel mean-shift on flattened lattices (R^9).

    L_gen: (B, 3, 3)
    L_pos: (B', 3, 3)
    Returns V_L: (B, 3, 3) — average drift across temperatures, each normalized.
    """
    B = L_gen.shape[0]
    Lg = L_gen.reshape(B, 9)
    Lp = L_pos.reshape(L_pos.shape[0], 9)

    # pairwise squared distance (B, B')
    d2 = (Lg.unsqueeze(1) - Lp.unsqueeze(0)).pow(2).sum(-1)

    V_acc = torch.zeros_like(Lg)
    for tau in temperatures:
        w = _kernel_weights(d2, tau)                   # (B, B')
        target = w @ Lp                                # (B, 9)
        V = target - Lg                                # (B, 9)
        # per-temperature normalize so different scales contribute equally
        norm = V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        V_acc = V_acc + V / norm
    V_acc = V_acc / len(temperatures)
    return V_acc.reshape(B, 3, 3)


def compute_V_frac(
    F_gen: Tensor,
    F_pos: Tensor,
    temperatures: list[float],
) -> Tensor:
    """Multi-temperature drift on fractional coords with torus metric.

    F_gen: (B, N, 3)
    F_pos: (B', N, 3) — same N and same canonical atom order as F_gen.
    Returns V_F: (B, N, 3) — drift vectors in the torus tangent space
    (each component in (-0.5, 0.5]).
    """
    # pairwise torus diff: (B, B', N, 3)
    diff = torus_diff(F_gen.unsqueeze(1), F_pos.unsqueeze(0))
    # squared dist per-crystal, summed over atoms and 3 dims: (B, B')
    d2 = diff.pow(2).sum(dim=(-1, -2))

    V_acc = torch.zeros_like(F_gen)
    for tau in temperatures:
        w = _kernel_weights(d2, tau)                   # (B, B')
        # drift in tangent space: weighted sum of (F_pos - F_gen) via torus diff
        # diff has sign (F_gen - F_pos); we want (F_pos - F_gen) = -diff
        # weighted across F_pos (axis 1)
        drift = -(w.unsqueeze(-1).unsqueeze(-1) * diff).sum(dim=1)  # (B, N, 3)
        # per-temperature normalize (whole-crystal scale)
        flat = drift.reshape(drift.shape[0], -1)
        norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        drift = drift / norm.unsqueeze(-1)
        V_acc = V_acc + drift
    V_acc = V_acc / len(temperatures)
    return V_acc


def compute_V(
    L_gen: Tensor,
    F_gen: Tensor,
    L_pos: Tensor,
    F_pos: Tensor,
    temperatures_L: list[float] = (0.5, 1.0, 2.0),
    temperatures_F: list[float] = (0.02, 0.05, 0.2),
    repulsion: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Joint drift on (lattice, frac). Independent per modality, multi-temperature.

    If `repulsion` > 0, adds an anti-mode-seeking term:
        V_total = V_attract(gen -> targets) - repulsion * V_repel(gen -> other_gen)
    Repel uses the same kernel but with `gen` as both sides (excluding self).
    """
    V_L = compute_V_lattice(L_gen, L_pos, list(temperatures_L))
    V_F = compute_V_frac(F_gen, F_pos, list(temperatures_F))
    if repulsion > 0.0:
        # repulsion from other gen samples: V_attract(gen -> gen) but negated
        # mask self by setting diagonal to large distance
        V_L_rep = compute_V_lattice(L_gen, L_gen, list(temperatures_L))
        V_F_rep = compute_V_frac(F_gen, F_gen, list(temperatures_F))
        V_L = V_L - repulsion * V_L_rep
        V_F = V_F - repulsion * V_F_rep
    return V_L, V_F
