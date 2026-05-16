"""Task 4b — MP-20 data loader for DMF training.

Loads DiffCSP-format MP-20 CSV (with a `cif` column), parses each CIF via
pymatgen, applies Niggli reduction + primitive cell (DiffCSP convention from
its hparams.yaml), caches to a tensor file for fast subsequent loads.

Output per crystal:
  - L  (3, 3)    Cartesian lattice basis (rows = vectors)
  - F  (N, 3)    fractional coordinates in [0,1)^3
  - Z  (N,)      atomic numbers (int)
  - N  (scalar)  number of atoms

Provides a `same_N_iterator` that yields random batches of crystals with the
SAME N, which is what V_frac/V_lattice (Task 2) expects (it indexes atoms
positionally, so different N would mismatch).
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core import Structure


def _parse_cif(cif_str: str, niggli: bool = True, primitive: bool = False):
    """Parse one CIF -> (L, F, Z, N).  Returns None on failure."""
    try:
        s = Structure.from_str(cif_str, fmt="cif")
        if primitive:
            s = s.get_primitive_structure()
        if niggli:
            s = s.get_reduced_structure()
        L = np.array(s.lattice.matrix, dtype=np.float32)
        F = np.array(s.frac_coords, dtype=np.float32) % 1.0
        Z = np.array([site.specie.Z for site in s.sites], dtype=np.int64)
        return L, F, Z
    except Exception as e:  # noqa: BLE001
        return None


def load_mp20(csv_path: str, niggli: bool = True, primitive: bool = False, cache: str | None = None):
    """Parse the entire MP-20 CSV. Returns dict of lists keyed by N.

    cache: path to a .pt file. If exists, load. If None, compute fresh; if
    provided as a path and missing, compute and save.
    """
    if cache and Path(cache).exists():
        print(f"loading cache: {cache}")
        return torch.load(cache, weights_only=False)

    df = pd.read_csv(csv_path)
    print(f"parsing {len(df)} CIFs from {csv_path}")
    t0 = time.time()
    parsed = p_map(_parse_cif, df["cif"].tolist())
    n_ok = sum(1 for p in parsed if p is not None)
    print(f"  parsed {n_ok}/{len(df)} ok in {time.time()-t0:.1f}s")

    by_N: dict[int, list[dict]] = defaultdict(list)
    for p in parsed:
        if p is None:
            continue
        L, F, Z = p
        N = int(F.shape[0])
        by_N[N].append({
            "L": torch.from_numpy(L),
            "F": torch.from_numpy(F),
            "Z": torch.from_numpy(Z),
            "N": N,
        })

    data = {"by_N": dict(by_N), "n_total": n_ok, "csv_path": csv_path}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
        print(f"saved cache -> {cache}")
    return data


def same_N_iterator(data, batch_size: int, seed: int = 0) -> Iterator[dict]:
    """Yield batches of size `batch_size`, all crystals in batch share same N.

    Smaller groups produce smaller batches (last batch may be partial). Groups
    are shuffled, and within each group, crystals are shuffled before batching.
    """
    rng = random.Random(seed)
    groups = list(data["by_N"].keys())
    rng.shuffle(groups)
    for N in groups:
        group = data["by_N"][N][:]
        rng.shuffle(group)
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            if len(batch) < 2:
                continue  # V needs at least 2 to be meaningful
            L = torch.stack([c["L"] for c in batch], dim=0)
            F = torch.stack([c["F"] for c in batch], dim=0)
            Z = torch.stack([c["Z"] for c in batch], dim=0)
            yield {"L": L, "F": F, "Z": Z, "N": N, "B": len(batch)}


def main():
    """Smoke-test: load MP-20 train split, dump stats, iterate one epoch."""
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(Path.home() / "diffcsp/data/mp_20/train.csv"))
    p.add_argument("--cache", default=str(Path.home() / "miad/dmf_poc/cache/mp20_train.pt"))
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    t0 = time.time()
    data = load_mp20(args.csv, niggli=True, primitive=False, cache=args.cache)
    t_load = time.time() - t0
    print(f"\n=== stats ===")
    print(f"  n_total = {data['n_total']}")
    print(f"  load wall = {t_load:.1f}s")
    Ns = sorted(data["by_N"].keys())
    print(f"  N range: [{min(Ns)}, {max(Ns)}]")
    print(f"  N histogram (top 10 most common):")
    counts = sorted([(N, len(v)) for N, v in data["by_N"].items()], key=lambda x: -x[1])
    for N, c in counts[:10]:
        print(f"    N={N:3d}: {c}")

    # iterate one epoch
    print(f"\n=== iterate one epoch, batch_size={args.batch_size} ===")
    t0 = time.time()
    n_batches = 0
    n_yielded = 0
    constant_N = True
    for batch in same_N_iterator(data, args.batch_size):
        if batch["L"].shape[1:] != (3, 3):
            raise RuntimeError(f"bad L shape: {batch['L'].shape}")
        if batch["F"].shape[1] != batch["N"]:
            constant_N = False
        n_batches += 1
        n_yielded += batch["B"]
    t_iter = time.time() - t0
    print(f"  batches: {n_batches}, total crystals yielded: {n_yielded}")
    print(f"  epoch wall: {t_iter:.2f}s")
    print(f"  constant N within batch: {constant_N}")


if __name__ == "__main__":
    main()
