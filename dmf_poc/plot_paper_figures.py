"""Generate paper-ready figures from existing JSON metrics.

Figures:
  fig1_k_coverage.png    — match-rate vs K, DMF variants + DiffCSP
  fig2_per_N.png         — per-N comparison: DiffCSP K=5 vs DMF K=200 (compute parity)
  fig3_per_N_K_curve.png — match-rate vs K for selected N strata (N=4, 6, 8, 10)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent / "cache"


def load_json(p):
    return json.loads(Path(p).read_text())


def fig1_k_coverage():
    curve = load_json(ROOT / "match_at_k_curve.json")  # has diffcsp + DMF variants @ K=1..20
    fig, ax = plt.subplots(figsize=(7, 5))

    # DiffCSP curve
    d = curve["diffcsp"]["match_at_k"]
    ks = sorted(int(k) for k in d)
    ax.plot(ks, [d[str(k)] for k in ks], "o-", color="C3", linewidth=2,
            markersize=7, label="DiffCSP (1000 NFE/sample)")

    # DMF λ=1.0 curve (from K=1..20 + extend with K=30, 50, 75, 100, 200)
    dmf = curve["repel10"]["match_at_k"]
    ks_short = sorted(int(k) for k in dmf)
    # extend with extra points
    extra = {30: 0.275, 50: 0.32, 75: 0.365, 100: 0.375, 200: 0.41}
    extra[500] = 0.45  # K=500 result
    ks_full = ks_short + sorted(extra.keys())
    vals_full = [dmf[str(k)] for k in ks_short] + [extra[k] for k in sorted(extra.keys())]
    ax.plot(ks_full, vals_full, "s-", color="C0", linewidth=2,
            markersize=7, label="DMF + repulsion λ=1.0 (1 NFE/sample)")

    # other DMF variants for context (K=1..20 only)
    for name, color in [("default", "gray"), ("repel05", "C1")]:
        v = curve[name]["match_at_k"]
        ks_v = sorted(int(k) for k in v)
        ax.plot(ks_v, [v[str(k)] for k in ks_v], "o--", color=color, alpha=0.5,
                markersize=4,
                label=f"DMF {name.replace('05', ' λ=0.5').replace('default', '(λ=0)')}")

    ax.set_xlabel("K (samples per composition)", fontsize=12)
    ax.set_ylabel("match rate @K (StructureMatcher)", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 600)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Match-rate vs K — DiffCSP saturates, DMF+repulsion keeps gaining",
                 fontsize=11)
    plt.tight_layout()
    out = ROOT.parent / "fig1_k_coverage.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


def fig2_per_N_compute_parity():
    """DiffCSP K=5 vs DMF K=200 at compute parity (5000 NFE/comp), per-N."""
    # data hardcoded from compute_parity report
    Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 18, 20]
    diffcsp_K5 = [1.00, 1.00, 0.875, 0.75, 0.833, 0.80, 0.64, 0.40, 0.885, 0.692, 0.60, 0.556, 0.50, 0.333, 0.722]
    dmf_K200   = [1.00, 1.00, 1.000, 1.00, 0.944, 0.80, 0.68, 0.00, 0.038, 0.000, 0.00, 0.000, 0.00, 0.000, 0.000]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(Ns))
    w = 0.4
    ax.bar(x - w/2, diffcsp_K5, w, label="DiffCSP K=5 (5000 NFE/comp)", color="C3", alpha=0.8)
    ax.bar(x + w/2, dmf_K200,   w, label="DMF K=200 (200 NFE/comp — 25× less)", color="C0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("N (atoms per primitive cell)", fontsize=12)
    ax.set_ylabel("match rate", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Compute-parity per-N: DMF wins for N=4-8 at 25× lower compute", fontsize=12)

    # highlight DMF-win region
    for i, n in enumerate(Ns):
        if 4 <= n <= 8 and dmf_K200[i] >= diffcsp_K5[i]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="green")

    plt.tight_layout()
    out = ROOT.parent / "fig2_per_N_compute_parity.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


def fig3_per_N_K_curve():
    """For selected N strata, plot match-rate vs K (using K=20, K=100, K=200, K=500)."""
    # collected from various reports
    N_curves = {
        4:  {20: 0.80, 100: 1.00, 200: 1.00, 500: 1.00},
        6:  {20: 0.51, 100: 0.83, 200: 0.94, 500: 1.00},
        8:  {20: 0.22, 100: 0.52, 200: 0.68, 500: 0.84},
        10: {20: 0.00, 100: 0.00, 200: 0.038, 500: 0.077},
        12: {20: 0.00, 100: 0.00, 200: 0.000, 500: 0.000},
    }
    # DiffCSP reference (K=20 values from earlier compute-parity table)
    diffcsp_ref = {4: 0.99, 6: 0.84, 8: 0.84, 10: 0.90, 12: 0.85}

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {4: "C0", 6: "C1", 8: "C2", 10: "C3", 12: "C4"}
    for N, curve in N_curves.items():
        ks = sorted(curve.keys())
        ax.plot(ks, [curve[k] for k in ks], "o-", color=colors[N], linewidth=2, markersize=7,
                label=f"DMF N={N}")
        # DiffCSP reference as horizontal line
        ax.axhline(diffcsp_ref[N], color=colors[N], linestyle=":", alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlim(15, 600)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("K (samples per composition)", fontsize=12)
    ax.set_ylabel("DMF match rate @K", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    ax.set_title("Per-N K-coverage: DMF saturates near DiffCSP for N≤8, fails N≥10\n(dotted = DiffCSP K=20 baseline)",
                 fontsize=11)
    plt.tight_layout()
    out = ROOT.parent / "fig3_per_N_K_curve.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


def main():
    fig1_k_coverage()
    fig2_per_N_compute_parity()
    fig3_per_N_K_curve()


if __name__ == "__main__":
    main()
