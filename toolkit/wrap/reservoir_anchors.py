import pandas as pd

from toolkit import repo_data_path


def resolve_eva_flo_paths(basin_config: dict):
    """Derive (eva_path, flo_path) for a basin. The sibling .eva file is matched
    case-insensitively, since WAM dirs mix filename casing (e.g. Trinity's Trin3.eva vs trin3.dat)."""
    flo_path = repo_data_path / basin_config["flo_file"]
    basin_dir = flo_path.parent
    matches = [p for p in basin_dir.iterdir() if p.suffix.lower() == ".eva"]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one .eva file in {basin_dir}, found {matches}")
    return matches[0], flo_path


def annualize(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(float).resample("YS").sum()


def compute_annual_raw_correlations(eva_df: pd.DataFrame, flo_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between every EVA site and every CP on annual sums; shape (n_eva_sites, n_cps)."""
    eva_annual = annualize(eva_df)
    flo_annual = annualize(flo_df)
    common_index = eva_annual.index.intersection(flo_annual.index)
    eva_annual = eva_annual.loc[common_index]
    flo_annual = flo_annual.loc[common_index]

    eva_sites = [c for c in eva_annual.columns if isinstance(c, str) and c.startswith("EV")]
    cps = list(flo_annual.columns)

    corr = pd.DataFrame(index=eva_sites, columns=cps, dtype=float)
    for site in eva_sites:
        site_vals = eva_annual[site]
        for cp in cps:
            corr.loc[site, cp] = site_vals.corr(flo_annual[cp])

    return corr


def best_anchors(corr: pd.DataFrame) -> pd.DataFrame:
    """For each EVA site, the CP with the highest |r|. Columns: best_cp, r_value."""
    records = []
    for site in corr.index:
        row = corr.loc[site]
        best_cp = row.abs().idxmax()
        records.append({"eva_site": site, "best_cp": best_cp, "r_value": row[best_cp]})
    summary = pd.DataFrame(records).set_index("eva_site")
    return summary.sort_values("r_value", ascending=False, key=abs)
