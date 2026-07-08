"""
Stage 2 of the reservoir anchor correlation plan (see kirklocal/plan_reservoir_eva.md).

Reads the correlation output produced by explore_reservoir_eva.py (Stage 1), applies a
threshold, and writes accepted EVA-site -> anchor-CP assignments to
data/configs/basins.json for the target basin. Only the target basin's
"reservoir_anchors" key is overwritten; everything else in basins.json is left alone.

Re-run after adjusting --threshold to update the assignments.

Usage:
    python workflow/III_WRAP_Execution/assign_reservoir_anchors.py --basin Colorado
"""

import argparse
import json

from toolkit import repo_data_path

BASINS_PATH = repo_data_path / "configs" / "basins.json"
EXPLORATION_OUTPUT_DIR = repo_data_path.parent / "kirklocal" / "reservoir_exploration" / "outputs"

DEFAULT_THRESHOLD = 0.3


def main():
    parser = argparse.ArgumentParser(description="Assign reservoir EVA-site anchors for a basin")
    parser.add_argument("--basin", required=True, help="Basin name as it appears in basins.json (e.g. Colorado)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help=f"Minimum |r| required to accept an anchor (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    with open(BASINS_PATH, "r") as f:
        basins = json.load(f)

    if args.basin not in basins:
        raise SystemExit(f"Error: basin '{args.basin}' not found in {BASINS_PATH}")

    anchors_csv = EXPLORATION_OUTPUT_DIR / args.basin / "eva_best_anchors.csv"
    if not anchors_csv.exists():
        raise SystemExit(
            f"Error: no Stage 1 correlation output found at {anchors_csv}\n"
            f"Run explore_reservoir_eva.py for basin '{args.basin}' first."
        )

    import pandas as pd
    summary = pd.read_csv(anchors_csv, index_col="eva_site")

    accepted = {}
    skipped = []
    for eva_site, row in summary.iterrows():
        r_value = row["r_value"]
        if abs(r_value) >= args.threshold:
            accepted[eva_site] = {
                "anchor_cp": row["best_cp"],
                "anchor_correlation": round(float(r_value), 4),
            }
        else:
            skipped.append((eva_site, row["best_cp"], r_value))

    basins[args.basin]["reservoir_anchors"] = accepted

    with open(BASINS_PATH, "w") as f:
        json.dump(basins, f, indent=2)
        f.write("\n")

    print(f"Assigned {len(accepted)}/{len(summary)} EVA site anchors for basin '{args.basin}' "
          f"(threshold |r| >= {args.threshold})")
    print(f"Wrote {BASINS_PATH}")

    if skipped:
        print(f"\nSkipped {len(skipped)} sites below threshold (will use historical climatology):")
        for eva_site, best_cp, r_value in skipped:
            print(f"  {eva_site}: best candidate was {best_cp} (r={r_value:.3f})")


if __name__ == "__main__":
    main()
