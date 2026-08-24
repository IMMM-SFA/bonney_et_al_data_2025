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


def compute_annual_scores(eva_df: pd.DataFrame, flo_df: pd.DataFrame, score_fn) -> pd.DataFrame:
    """Shared scaffolding for per-(EVA site, CP) annual scoring: annualize both, align on
    their common years, then apply score_fn(flow_annual, eva_annual) to every (site, cp)
    pair. shape (n_eva_sites, n_cps)."""
    eva_annual = annualize(eva_df)
    flo_annual = annualize(flo_df)
    common_index = eva_annual.index.intersection(flo_annual.index)
    eva_annual = eva_annual.loc[common_index]
    flo_annual = flo_annual.loc[common_index]

    eva_sites = [c for c in eva_annual.columns if isinstance(c, str) and c.startswith("EV")]
    cps = list(flo_annual.columns)

    scores = pd.DataFrame(index=eva_sites, columns=cps, dtype=float)
    for site in eva_sites:
        site_vals = eva_annual[site]
        for cp in cps:
            scores.loc[site, cp] = score_fn(flo_annual[cp], site_vals)

    return scores


def _semilog_r2(x: pd.Series, y: pd.Series) -> float:
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    df_xy = df_xy[df_xy["x"] > 0]
    if len(df_xy) < 3:
        return np.nan

    design = sm.add_constant(np.log(df_xy["x"]))
    response = df_xy["y"]
    return sm.OLS(response, design).fit().rsquared


def compute_annual_semilog_r2(eva_df: pd.DataFrame, flo_df: pd.DataFrame) -> pd.DataFrame:
    """R^2 of EVA ~ log(flow) between every EVA site and every CP on annual sums,
    restricted to years where flow is positive; shape (n_eva_sites, n_cps)."""
    return compute_annual_scores(eva_df, flo_df, _semilog_r2)


def best_anchors(scores: pd.DataFrame) -> pd.DataFrame:
    """For each EVA site, the CP with the highest-magnitude score (abs() is a no-op for
    non-negative scores like R^2, and correctly ranks signed scores like Pearson r).
    Columns: best_cp, score. NaN score if every candidate CP was NaN for that site."""
    records = []
    for site in scores.index:
        row = scores.loc[site]
        if row.isna().all():
            records.append({"eva_site": site, "best_cp": np.nan, "score": np.nan})
            continue
        best_cp = row.abs().idxmax()
        records.append({"eva_site": site, "best_cp": best_cp, "score": row[best_cp]})
    summary = pd.DataFrame(records).set_index("eva_site")
    return summary.sort_values("score", ascending=False, key=abs)
