"""Task 6a — Side-by-side comparison of DiffCSP-mp_gen vs DMF gen metrics.

Reads two JSON files produced by `baseline_metrics.py` and emits a markdown
table with values, absolute delta, and relative delta. Also notes which side
is "better" per metric (higher-is-better for valid/cov, lower-is-better for
wdist/amsd/amcd).

Usage:
  python compare_metrics.py \\
      --left  /path/to/diffcsp_metrics.json --left-label "DiffCSP-mp_gen" \\
      --right /path/to/dmf_metrics.json     --right-label "DMF-50k"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# direction: True = higher is better, False = lower is better
DIRECTION = {
    "comp_valid": True,
    "struct_valid": True,
    "valid": True,
    "cov_recall": True,
    "cov_precision": True,
    "wdist_density": False,
    "wdist_num_elems": False,
    "amsd_recall": False,
    "amsd_precision": False,
    "amcd_recall": False,
    "amcd_precision": False,
}


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def winner(left, right, higher_is_better):
    if left is None or right is None:
        return ""
    if abs(left - right) < 1e-6:
        return "≈"
    if higher_is_better:
        return "L" if left > right else "R"
    return "L" if left < right else "R"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--left-label", default="LEFT")
    ap.add_argument("--right-label", default="RIGHT")
    ap.add_argument("--out", default=None, help="optional output markdown file")
    args = ap.parse_args()

    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())

    keys = sorted(set(left) | set(right), key=lambda k: (-1 if k in DIRECTION else 0, k))

    lines = []
    lines.append(f"# Metrics comparison — {args.left_label} vs {args.right_label}\n")
    lines.append(f"- Left:  `{args.left}` ({args.left_label})")
    lines.append(f"- Right: `{args.right}` ({args.right_label})\n")
    lines.append(f"| Metric | {args.left_label} | {args.right_label} | Δ (R−L) | Δ% | Better |")
    lines.append("|---|---:|---:|---:|---:|:---:|")

    for k in keys:
        l = left.get(k)
        r = right.get(k)
        d = None
        dp = None
        if isinstance(l, (int, float)) and isinstance(r, (int, float)):
            d = r - l
            if abs(l) > 1e-9:
                dp = 100.0 * d / abs(l)
        hib = DIRECTION.get(k)
        win = winner(l, r, hib) if hib is not None else ""
        lines.append(
            f"| `{k}` | {fmt(l)} | {fmt(r)} | "
            f"{('+' if (d or 0) >= 0 else '')}{fmt(d) if d is not None else ''} | "
            f"{('+' if (dp or 0) >= 0 else '')}{fmt(dp) if dp is not None else ''}% | "
            f"{win} |"
        )

    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).write_text(out + "\n")
        print(f"\nsaved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
