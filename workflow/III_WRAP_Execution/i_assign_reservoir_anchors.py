import json

from toolkit import repo_data_path
from toolkit.wrap.io import evp_to_df, flo_to_df
from toolkit.wrap.reservoir_anchors import best_anchors, compute_annual_loglog_r2, resolve_eva_flo_paths

### Settings ###
# None

### Path Configuration ###
BASINS_PATH = repo_data_path / "configs" / "basins.json"

### Functions ###


def assign_basin_anchors(basin_name: str, basin_config: dict) -> dict:
    """Fit log(EVA) ~ log(flow) between this basin's EVA sites and its FLO control points,
    and anchor each site to its single best-fitting (highest R^2) control point."""
    eva_path, flo_path = resolve_eva_flo_paths(basin_config)
    eva_df = evp_to_df(str(eva_path))
    flo_df = flo_to_df(str(flo_path))
    summary = best_anchors(compute_annual_loglog_r2(eva_df, flo_df))

    anchors = {
        eva_site: {
            "anchor_cp": row["best_cp"],
            "anchor_r_squared": round(float(row["score"]), 4),
        }
        for eva_site, row in summary.iterrows()
    }

    print(f"Assigned {len(anchors)} EVA site anchors for basin '{basin_name}'")

    return anchors


### Main ###


def main():
    with open(BASINS_PATH, "r") as f:
        basins = json.load(f)

    for basin_name, basin_config in basins.items():
        basins[basin_name]["reservoir_anchors"] = assign_basin_anchors(basin_name, basin_config)

    with open(BASINS_PATH, "w") as f:
        json.dump(basins, f, indent=2)
        f.write("\n")
    print(f"\nWrote {BASINS_PATH}")


if __name__ == "__main__":
    main()
