import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from toolkit import repo_data_path
from toolkit.wrap.io import evp_to_df, flo_to_df

BASINS_PATH = repo_data_path / "configs" / "basins.json"
OUTPUT_ROOT = (Path(__file__).parent / "outputs" / "reservoir_exploration").resolve()

WEAK_ANCHOR_THRESHOLD = 0.3


def resolve_wam_paths(basin_config: dict):
    """Derive (dat_path, eva_path, flo_path) for a basin from its basins.json entry.

    flo_file's exact case is trusted as given; the sibling .dat/.eva files are found by
    case-insensitive suffix match in the same directory, since basin WAM directories mix
    filename casing (e.g. Trinity's trin3.dat vs Trin3.flo/Trin3.eva).
    """
    flo_path = repo_data_path / basin_config["flo_file"]
    basin_dir = flo_path.parent

    def find_sibling(suffix: str) -> Path:
        matches = [p for p in basin_dir.iterdir() if p.suffix.lower() == suffix]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {suffix} file in {basin_dir}, found {matches}"
            )
        return matches[0]

    return find_sibling(".dat"), find_sibling(".eva"), flo_path


def parse_eva_site_to_cp(dat_path, eva_path) -> dict:
    """Parse C3.dat for each EVA site's own control point."""
    quad_comment_re = re.compile(r"REPRESENTED BY EVAP @ CP\s+(\S+)")
    documented = {}
    with open(dat_path, "rt") as f:
        for line in f:
            match = quad_comment_re.search(line)
            if match:
                documented[f"EV{match.group(1)}"] = match.group(1)

    cp_ids = set()
    with open(dat_path, "rt") as f:
        for line in f:
            if line.startswith("CP") and not line.startswith("CP "):
                cp_id = line[2:10].strip()
                if cp_id:
                    cp_ids.add(cp_id)

    with open(eva_path, "rt") as f:
        eva_sites = sorted({
            line.split()[0] for line in f
            if line.strip() and line[0] != "*"
        })

    lookup = {}
    for site in eva_sites:
        candidate = site[2:]
        if candidate in cp_ids:
            lookup[site] = candidate
        else:
            print(f"WARNING: no matching CP found in {dat_path.name} for EVA site {site}")

    return lookup


def compute_correlations(eva_df: pd.DataFrame, flo_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between every EVA site and every CP, shape (n_eva_sites, n_cps)."""
    common_index = eva_df.index.intersection(flo_df.index)
    eva_aligned = eva_df.loc[common_index].astype(float)
    flo_aligned = flo_df.loc[common_index].astype(float)

    eva_sites = [c for c in eva_aligned.columns if isinstance(c, str) and c.startswith("EV")]
    cps = list(flo_aligned.columns)

    corr = pd.DataFrame(index=eva_sites, columns=cps, dtype=float)
    for site in eva_sites:
        site_vals = eva_aligned[site]
        for cp in cps:
            corr.loc[site, cp] = site_vals.corr(flo_aligned[cp])

    return corr


def signed_log1p(x):
    """log1p that preserves sign, for quantities (like net evap) that can be negative."""
    return np.sign(x) * np.log1p(np.abs(x))


def annualize(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(float).resample("YS").sum()


# Flow is non-negative but can be exactly 0 in a given year at small CPs, so log1p
# rather than log. Net evap can be negative (wet years, precip > evap), so a signed
# log1p rather than a plain log.
ANNUAL_TRANSFORMS = {
    "annual_raw": (lambda f: f, lambda e: e),
    "annual_log_flow": (lambda f: np.log1p(f), lambda e: e),
    "annual_log_evap": (lambda f: f, lambda e: signed_log1p(e)),
    "annual_log_both": (lambda f: np.log1p(f), lambda e: signed_log1p(e)),
}


def compute_annual_correlations(eva_df: pd.DataFrame, flo_df: pd.DataFrame) -> dict:
    """Pearson r between every EVA site and every CP, annualized, under each transform
    in ANNUAL_TRANSFORMS. Returns {transform_label: correlation DataFrame}."""
    eva_annual = annualize(eva_df)
    flo_annual = annualize(flo_df)
    common_index = eva_annual.index.intersection(flo_annual.index)
    eva_annual = eva_annual.loc[common_index]
    flo_annual = flo_annual.loc[common_index]

    eva_sites = [c for c in eva_annual.columns if isinstance(c, str) and c.startswith("EV")]
    cps = list(flo_annual.columns)

    results = {}
    for label, (flow_transform, evap_transform) in ANNUAL_TRANSFORMS.items():
        flo_t = flow_transform(flo_annual)
        eva_t = evap_transform(eva_annual)
        corr = pd.DataFrame(index=eva_sites, columns=cps, dtype=float)
        for site in eva_sites:
            site_vals = eva_t[site]
            for cp in cps:
                corr.loc[site, cp] = site_vals.corr(flo_t[cp])
        results[label] = corr

    return results


def best_anchors(corr: pd.DataFrame) -> pd.DataFrame:
    """For each EVA site, the CP with the highest |r|, flagged weak if below threshold."""
    records = []
    for site in corr.index:
        row = corr.loc[site]
        best_cp = row.abs().idxmax()
        r_value = row[best_cp]
        records.append({
            "eva_site": site,
            "best_cp": best_cp,
            "r_value": r_value,
            "weak_anchor": abs(r_value) < WEAK_ANCHOR_THRESHOLD,
        })
    summary = pd.DataFrame(records).set_index("eva_site")
    return summary.sort_values("r_value", ascending=False, key=abs)


def compare_schemes(monthly_corr: pd.DataFrame, annual_corrs: dict) -> pd.DataFrame:
    """Best |r| per EVA site under monthly_raw and each annual transform, plus which
    scheme wins per site."""
    schemes = {"monthly_raw": monthly_corr, **annual_corrs}
    best_r = pd.DataFrame({
        label: corr.abs().max(axis=1) for label, corr in schemes.items()
    })
    best_r["best_scheme"] = best_r.idxmax(axis=1)
    return best_r.sort_values("monthly_raw", ascending=False)


def plot_heatmap(corr: pd.DataFrame, output_path, title="Pearson r: EVA site net evaporation vs FLO control point flow"):
    plt.figure(figsize=(14, 16))
    sns.heatmap(corr.astype(float), cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel("Control point")
    plt.ylabel("EVA site")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_spot_checks(summary: pd.DataFrame, eva_df: pd.DataFrame, flo_df: pd.DataFrame, output_dir, n=10):
    top = summary.reindex(summary["r_value"].abs().sort_values(ascending=False).index).head(n)
    common_index = eva_df.index.intersection(flo_df.index)

    _, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    for ax, (site, row) in zip(axes, top.iterrows()):
        cp = row["best_cp"]
        r = row["r_value"]
        eva_series = eva_df.loc[common_index, site].astype(float)
        flo_series = flo_df.loc[common_index, cp].astype(float)

        ax2 = ax.twinx()
        ax.plot(common_index, eva_series, color="tab:blue", label=f"{site} net evap", alpha=0.8)
        ax2.plot(common_index, flo_series, color="tab:orange", label=f"{cp} flow", alpha=0.6)
        ax.set_title(f"{site} vs {cp} (r={r:.2f})")
        ax.set_ylabel("Net evap", color="tab:blue")
        ax2.set_ylabel("Flow", color="tab:orange")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "top10_spot_check_timeseries.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Explore EVA-site vs FLO-CP correlations for a basin")
    parser.add_argument("--basin", required=True, help="Basin name as it appears in basins.json (e.g. Colorado)")
    args = parser.parse_args()

    with open(BASINS_PATH, "r") as f:
        basins = json.load(f)

    if args.basin not in basins:
        raise SystemExit(f"Error: basin '{args.basin}' not found in {BASINS_PATH}")

    dat_path, eva_path, flo_path = resolve_wam_paths(basins[args.basin])
    output_dir = OUTPUT_ROOT / args.basin
    output_dir.mkdir(parents=True, exist_ok=True)

    own_cp_lookup = parse_eva_site_to_cp(dat_path, eva_path)
    pd.Series(own_cp_lookup, name="own_cp").rename_axis("eva_site").to_csv(
        output_dir / "eva_site_own_cp_lookup.csv"
    )

    eva_df = evp_to_df(str(eva_path))
    flo_df = flo_to_df(str(flo_path))

    monthly_corr = compute_correlations(eva_df, flo_df)
    monthly_corr.to_csv(output_dir / "eva_cp_correlation_matrix_monthly_raw.csv")
    plot_heatmap(
        monthly_corr, output_dir / "eva_cp_correlation_heatmap_monthly_raw.png",
        title="Pearson r: EVA site vs FLO CP, monthly_raw",
    )
    plot_spot_checks(best_anchors(monthly_corr), eva_df, flo_df, output_dir, n=10)

    annual_corrs = compute_annual_correlations(eva_df, flo_df)
    for label, annual_corr in annual_corrs.items():
        annual_corr.to_csv(output_dir / f"eva_cp_correlation_matrix_{label}.csv")
        plot_heatmap(
            annual_corr, output_dir / f"eva_cp_correlation_heatmap_{label}.png",
            title=f"Pearson r: EVA site vs FLO CP, {label}",
        )

    comparison = compare_schemes(monthly_corr, annual_corrs)
    comparison.to_csv(output_dir / "eva_scheme_comparison.csv")

    scheme_cols = ["monthly_raw"] + list(annual_corrs.keys())
    print(comparison[scheme_cols].describe().loc[["mean", "50%", "min", "max"]].rename(index={"50%": "median"}).to_string())
    print(comparison["best_scheme"].value_counts().to_string())
    for col in scheme_cols:
        print(f"{col}: >=0.3: {(comparison[col] >= 0.3).sum()}, >=0.5: {(comparison[col] >= 0.5).sum()}")

    # annual_raw is used for anchor selection (Stage 2) and Stage 3's analog-year
    # matching: it captures nearly all of the annual-vs-monthly improvement without a
    # log-transformed anchor CP, which would complicate Stage 3's raw-value transfer.
    summary = best_anchors(annual_corrs["annual_raw"])
    summary.to_csv(output_dir / "eva_best_anchors.csv")
    print(f"{summary['weak_anchor'].sum()}/{len(summary)} weak anchors (|r| < {WEAK_ANCHOR_THRESHOLD})")


if __name__ == "__main__":
    main()
