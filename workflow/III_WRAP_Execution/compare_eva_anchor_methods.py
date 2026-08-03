import json
from pathlib import Path

import numpy as np
import pandas as pd

from toolkit import repo_data_path
from toolkit.wrap.io import evp_to_df, flo_to_df

BASINS_PATH = repo_data_path / "configs" / "basins.json"
OUTPUT_ROOT = (Path(__file__).parent / "outputs" / "reservoir_exploration").resolve()


def resolve_eva_flo_paths(basin_config: dict):
    flo_path = repo_data_path / basin_config["flo_file"]
    basin_dir = flo_path.parent
    matches = [p for p in basin_dir.iterdir() if p.suffix.lower() == ".eva"]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one .eva file in {basin_dir}, found {matches}")
    return matches[0], flo_path


def annual_sums(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(float).resample("YS").sum()


def colleague_best_cp(eva_annual: pd.DataFrame, flo_annual: pd.DataFrame) -> pd.DataFrame:
    """Best CP per EVA site by highest log-log R^2 (colleague's method), dropping
    non-positive years for each pair the same way the original script does."""
    common_index = eva_annual.index.intersection(flo_annual.index)
    eva_annual = eva_annual.loc[common_index]
    flo_annual = flo_annual.loc[common_index]

    eva_sites = [c for c in eva_annual.columns if isinstance(c, str) and c.startswith("EV")]
    cps = list(flo_annual.columns)

    log_flo = np.log(flo_annual.where(flo_annual > 0))
    log_eva = np.log(eva_annual.where(eva_annual > 0))

    r2 = pd.DataFrame(index=eva_sites, columns=cps, dtype=float)
    for site in eva_sites:
        y = log_eva[site]
        for cp in cps:
            paired = pd.DataFrame({"x": log_flo[cp], "y": y}).dropna()
            if len(paired) < 3:
                continue
            r2.loc[site, cp] = paired["x"].corr(paired["y"]) ** 2

    # Some EVA sites (e.g. malformed/placeholder IDs) have no valid CP pair at all --
    # idxmax on an all-NaN row raises, so resolve those separately and leave them NaN.
    valid_rows = r2.dropna(how="all")
    best_cp = pd.Series(index=r2.index, dtype=object)
    best_r2 = pd.Series(index=r2.index, dtype=float)
    best_cp.loc[valid_rows.index] = valid_rows.idxmax(axis=1)
    best_r2.loc[valid_rows.index] = valid_rows.max(axis=1)

    return pd.DataFrame({"colleague_best_cp": best_cp, "colleague_r2": best_r2})


def compare_basin(basin_name: str, basin_config: dict) -> pd.DataFrame:
    eva_path, flo_path = resolve_eva_flo_paths(basin_config)
    eva_df = evp_to_df(str(eva_path))
    flo_df = flo_to_df(str(flo_path))

    colleague = colleague_best_cp(annual_sums(eva_df), annual_sums(flo_df))

    ours_path = OUTPUT_ROOT / basin_name / "eva_best_anchors.csv"
    if not ours_path.exists():
        raise SystemExit(
            f"Error: no anchor output found at {ours_path}. Run explore_reservoir_eva.py "
            f"--basin {basin_name} first."
        )
    ours = pd.read_csv(ours_path, index_col="eva_site")[["best_cp", "r_value"]].rename(
        columns={"best_cp": "our_anchor_cp", "r_value": "our_r"}
    )

    comparison = ours.join(colleague, how="outer")
    comparison["agree"] = comparison["our_anchor_cp"] == comparison["colleague_best_cp"]
    return comparison.reindex(comparison["our_r"].abs().sort_values(ascending=False).index)


def main():
    with open(BASINS_PATH, "r") as f:
        basins = json.load(f)

    for basin_name, basin_config in basins.items():
        print(f"\n=== {basin_name} ===")
        comparison = compare_basin(basin_name, basin_config)
        output_path = OUTPUT_ROOT / basin_name / "eva_anchor_method_comparison.csv"
        comparison.to_csv(output_path)

        n_sites = len(comparison)
        n_agree = int(comparison["agree"].sum())
        print(f"Agreement: {n_agree}/{n_sites} ({n_agree / n_sites:.0%})")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
