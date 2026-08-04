import numpy as np
import pandas as pd
import statsmodels.api as sm

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


def _loglog_r2(x: pd.Series, y: pd.Series) -> float:
    """R^2 of an OLS fit of log(y) ~ log(x), restricted to years where both are positive."""
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    df_xy = df_xy[(df_xy["x"] > 0) & (df_xy["y"] > 0)]

    design = sm.add_constant(np.log(df_xy["x"]))
    response = np.log(df_xy["y"])
    return sm.OLS(response, design).fit().rsquared


def compute_annual_loglog_r2(eva_df: pd.DataFrame, flo_df: pd.DataFrame) -> pd.DataFrame:
    """R^2 of log(EVA) ~ log(flow) between every EVA site and every CP on annual sums,
    restricted to years where both are positive; shape (n_eva_sites, n_cps)."""
    eva_annual = annualize(eva_df)
    flo_annual = annualize(flo_df)
    common_index = eva_annual.index.intersection(flo_annual.index)
    eva_annual = eva_annual.loc[common_index]
    flo_annual = flo_annual.loc[common_index]

    eva_sites = [c for c in eva_annual.columns if isinstance(c, str) and c.startswith("EV")]
    cps = list(flo_annual.columns)

    r_squared = pd.DataFrame(index=eva_sites, columns=cps, dtype=float)
    for site in eva_sites:
        site_vals = eva_annual[site]
        for cp in cps:
            r_squared.loc[site, cp] = _loglog_r2(flo_annual[cp], site_vals)

    return r_squared


def best_anchors(scores: pd.DataFrame) -> pd.DataFrame:
    """For each EVA site, the CP with the highest score. Columns: best_cp, score."""
    records = []
    for site in scores.index:
        row = scores.loc[site]
        best_cp = row.idxmax()
        records.append({"eva_site": site, "best_cp": best_cp, "score": row[best_cp]})
    summary = pd.DataFrame(records).set_index("eva_site")
    return summary.sort_values("score", ascending=False)
