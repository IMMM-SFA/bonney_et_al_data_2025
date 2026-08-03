import json
from pathlib import Path

import pandas as pd

from toolkit import repo_data_path

### Settings ###
THRESHOLD = 0.3  # Minimum |r| required to accept an anchor

### Path Configuration ###
BASINS_PATH = repo_data_path / "configs" / "basins.json"
EXPLORATION_OUTPUT_DIR = Path(__file__).parent / "outputs" / "reservoir_exploration"

### Functions ###


def assign_basin_anchors(basin_name: str, summary: pd.DataFrame, threshold: float) -> dict:
    """Threshold Stage 1's per-EVA-site correlation summary into accepted anchor assignments."""
    accepted = {}
    skipped = []
    for eva_site, row in summary.iterrows():
        r_value = row["r_value"]
        if abs(r_value) >= threshold:
            accepted[eva_site] = {
                "anchor_cp": row["best_cp"],
                "anchor_correlation": round(float(r_value), 4),
            }
        else:
            skipped.append((eva_site, row["best_cp"], r_value))

    print(f"Assigned {len(accepted)}/{len(summary)} EVA site anchors for basin '{basin_name}' "
          f"(threshold |r| >= {threshold})")
    if skipped:
        print(f"  Skipped {len(skipped)} sites below threshold (will use historical climatology):")
        for eva_site, best_cp, r_value in skipped:
            print(f"    {eva_site}: best candidate was {best_cp} (r={r_value:.3f})")

    return accepted


### Main ###


def main():
    with open(BASINS_PATH, "r") as f:
        basins = json.load(f)

    for basin_name in basins:
        anchors_csv = EXPLORATION_OUTPUT_DIR / basin_name / "eva_best_anchors.csv"
        if not anchors_csv.exists():
            print(f"Skipping '{basin_name}': no Stage 1 correlation output found at {anchors_csv}. "
                  f"Run explore_reservoir_eva.py first.")
            continue

        summary = pd.read_csv(anchors_csv, index_col="eva_site")
        basins[basin_name]["reservoir_anchors"] = assign_basin_anchors(basin_name, summary, THRESHOLD)

    with open(BASINS_PATH, "w") as f:
        json.dump(basins, f, indent=2)
        f.write("\n")
    print(f"\nWrote {BASINS_PATH}")


if __name__ == "__main__":
    main()
