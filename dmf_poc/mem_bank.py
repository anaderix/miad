"""Per-N ring-buffer memory bank of real MP-20 crystals.

Inspired by /home/anaderi/FlowMatching/DM/memory_bank.py (ImageNet drifting).
For each N (number of atoms per cell), keep up to `max_per_N` real-data
(L, F) pairs in a ring buffer. Each training step:
  1. Adds the current batch's GT crystals to the bank under their N.
  2. Samples `n_extra` crystals from the bank for the same N as augmented
     positives in V kernel target.

This breaks out of the "64-positives-per-V-evaluation" bottleneck where
singleton compositions don't get meaningful drift signal.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch


class CrystalMemoryBank:
    """Per-N ring buffer of (L, F) tensors."""

    def __init__(self, max_per_N: int = 256, device: str | torch.device = "cpu"):
        self.max_per_N = int(max_per_N)
        self.device = torch.device(device)
        # bank: N -> dict(L=(M,3,3), F=(M,N,3))
        self._bank: dict[int, dict] = {}
        self._ptr: dict[int, int] = defaultdict(int)
        self._count: dict[int, int] = defaultdict(int)

    def _ensure(self, N: int):
        if N not in self._bank:
            self._bank[N] = {
                "L": torch.zeros(self.max_per_N, 3, 3, dtype=torch.float32, device=self.device),
                "F": torch.zeros(self.max_per_N, N, 3, dtype=torch.float32, device=self.device),
            }

    def add(self, L: torch.Tensor, F: torch.Tensor, N: int) -> None:
        """L: (B, 3, 3), F: (B, N, 3) — per-batch crystals (real data)."""
        assert F.shape[1] == N
        self._ensure(N)
        B = L.shape[0]
        L_cpu = L.detach().to(self.device, copy=True, non_blocking=True)
        F_cpu = F.detach().to(self.device, copy=True, non_blocking=True)
        for i in range(B):
            idx = self._ptr[N]
            self._bank[N]["L"][idx] = L_cpu[i]
            self._bank[N]["F"][idx] = F_cpu[i]
            self._ptr[N] = (idx + 1) % self.max_per_N
            if self._count[N] < self.max_per_N:
                self._count[N] += 1

    def sample(self, N: int, n_samples: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return (L_mem, F_mem) of shape (n_samples, 3, 3), (n_samples, N, 3) or None if empty."""
        if N not in self._bank or self._count[N] == 0:
            return None
        valid = self._count[N]
        n = min(n_samples, valid)
        if n < n_samples:
            # sample with replacement when not enough valid entries
            idx = np.random.choice(valid, n_samples, replace=True)
        else:
            idx = np.random.choice(valid, n_samples, replace=False)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        L_mem = self._bank[N]["L"][idx_t]
        F_mem = self._bank[N]["F"][idx_t]
        return L_mem, F_mem

    def stats(self) -> str:
        rows = [(N, self._count[N]) for N in sorted(self._bank)]
        return "  ".join(f"N{N}={c}" for N, c in rows) or "(empty)"
