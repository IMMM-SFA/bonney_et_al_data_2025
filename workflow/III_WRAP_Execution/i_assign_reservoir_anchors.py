import json

from toolkit import repo_data_path
from toolkit.wrap.io import evp_to_df, flo_to_df
from toolkit.wrap.reservoir_anchors import best_anchors, compute_annual_raw_correlations, resolve_eva_flo_paths

### Settings ###
THRESHOLD = 0.3  # Minimum |r| required to accept an anchor

### Path Configuration ###
BASINS_PATH = repo_data_path / "configs" / "basins.json"

### Functions ###


def assign_basin_anchors(basin_name: str, basin_config: dict, threshold: float) -> dict:
    """Correlate this basin's EVA sites against its FLO control points and threshold the
    best-correlated pick per site into accepted anchor assignments."""
    eva_path, flo_path = resolve_eva_flo_paths(basin_config)
    eva_df = evp_to_df(str(eva_path))
    flo_df = flo_to_df(str(flo_path))
    summary = best_anchors(compute_annual_raw_correlations(eva_df, flo_df))

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

    for basin_name, basin_config in basins.items():
        basins[basin_name]["reservoir_anchors"] = assign_basin_anchors(basin_name, basin_config, THRESHOLD)

    with open(BASINS_PATH, "w") as f:
        json.dump(basins, f, indent=2)
        f.write("\n")
    print(f"\nWrote {BASINS_PATH}")


if __name__ == "__main__":
    main()
