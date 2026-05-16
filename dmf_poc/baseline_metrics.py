"""Compute partial DiffCSP gen metrics (skipping property wdist).

Re-uses DiffCSP's own evaluators (GenEval, get_crystal_array_list) so numbers
are byte-identical to what compute_metrics.py would print, minus the
prop_wdist (which requires a separately-trained DimeNet++ that's not in the
public checkpoint folder).

Usage:
  cd ~/diffcsp && source .env && source .venv/bin/activate
  export PYTHONPATH=$HOME/diffcsp:$PYTHONPATH
  python ~/miad/dmf_poc/baseline_metrics.py \\
      --root_path ~/diffcsp/checkpoints/mp_gen --gt_file data/mp_20/test.csv
"""

import argparse
import json
import sys
from pathlib import Path

# allow running from any cwd as long as diffcsp/ on PYTHONPATH
sys.path.insert(0, str(Path.home() / "diffcsp" / "scripts"))
import pandas as pd
from p_tqdm import p_map
from compute_metrics import GenEval, get_crystal_array_list, Crystal, get_gt_crys_ori


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", required=True)
    p.add_argument("--gt_file", required=True)
    p.add_argument("--out", default=None, help="json output path")
    args = p.parse_args()

    root = Path(args.root_path)
    gen_file = root / "eval_gen.pt"
    print(f"loading: {gen_file}")
    crys_array_list, _ = get_crystal_array_list(str(gen_file), batch_idx=-2)
    print(f"  n_gen_crystals = {len(crys_array_list)}")

    print(f"loading GT: {args.gt_file}")
    csv = pd.read_csv(args.gt_file)
    gt_crys = p_map(get_gt_crys_ori, csv["cif"])
    print(f"  n_gt = {len(gt_crys)}")

    print("wrapping in Crystal objects (this triggers validity check) ...")
    gen_crys = [Crystal(c) for c in crys_array_list]

    # eval_model_name keys into COV_Cutoffs (used for coverage); 'mp20' fits MP-20.
    # It does NOT trigger model loading for our skipped prop_wdist path.
    evaluator = GenEval(gen_crys, gt_crys, eval_model_name="mp20")

    metrics = {}
    print("compute validity ...")
    metrics.update(evaluator.get_validity())
    print("compute density wdist ...")
    metrics.update(evaluator.get_density_wdist())
    print("SKIP prop_wdist (eval-model missing from public checkpoint folder)")
    print("compute num_elem wdist ...")
    metrics.update(evaluator.get_num_elem_wdist())
    print("compute coverage ...")
    metrics.update(evaluator.get_coverage())

    print()
    print("=== METRICS ===")
    print(json.dumps(metrics, indent=2, default=str))

    out = args.out or str(root / "metrics_partial.json")
    Path(out).write_text(json.dumps(metrics, indent=2, default=str))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
