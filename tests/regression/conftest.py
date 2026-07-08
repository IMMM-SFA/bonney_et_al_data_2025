"""
Fixtures for WRAP pipeline regression tests.

These tests require wine64, data/WRAP/SIM.exe, and data/WRAP/basin_wams/colo-full/.
Run with --generate-baseline on first use to create local baseline files.
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from toolkit import repo_data_path
from toolkit.wrap.io import df_to_flo, out_to_dfs, flo_to_df
from toolkit.wrap.wraputils import fix_cols
from toolkit.wrap.processing import process_diversion_csv, process_reservoir_csv
from tests.regression._paths import BASIN_NAME, FLO_FILE
from tests.regression._helpers import DIVERSION_COLUMNS, RESERVOIR_COLUMNS


@pytest.fixture(scope="session")
def generate_baseline(pytestconfig):
    return pytestconfig.getoption("--generate-baseline")


@pytest.fixture(scope="session")
def basin_config():
    basins_path = Path(repo_data_path) / "configs" / "basins.json"
    if not basins_path.exists():
        pytest.skip(f"basins.json not found at {basins_path}")
    with open(basins_path) as f:
        return json.load(f)[BASIN_NAME]


@pytest.fixture(scope="session")
def historical_flo():
    if not FLO_FILE.exists():
        pytest.skip(f"Historical FLO not found: {FLO_FILE}")
    return flo_to_df(str(FLO_FILE))


@pytest.fixture(scope="session")
def run_wrap_pipeline(basin_config, historical_flo):
    """Returns a callable that runs N realizations through the full WRAP pipeline.

    Signature:
        run_wrap_pipeline(streamflow, streamflow_index, streamflow_columns,
                          slot, flo_dir) -> dict[str, np.ndarray]

    streamflow : ndarray shape (n_realizations, n_months, n_sites)
    Returns a dict keyed "div_{var}" and "res_{var}" with stacked arrays of
    shape (n_realizations, n_months, n_columns).
    """
    def _run(streamflow, streamflow_index, streamflow_columns, slot, flo_dir):
        all_div = {}
        all_res = {}

        for i in range(streamflow.shape[0]):
            synth_flow = pd.DataFrame(
                streamflow[i],
                index=pd.to_datetime(streamflow_index),
                columns=streamflow_columns,
            )
            flo_ref = historical_flo.copy()
            flo_ref.index = synth_flow.index
            fix_cols(basin_config, synth_flow, flo_ref)

            flo_path = Path(flo_dir) / f"regression_{i:04d}.FLO"
            df_to_flo(synth_flow, flo_path)

            flo_name = slot.run(flo_path)
            out_file = slot.slot_dir / f"{flo_name}.OUT"
            mss_file = slot.slot_dir / f"{flo_name}.MSS"
            dfs = out_to_dfs(out_file)

            for key, df in process_diversion_csv(
                dfs["diversions"],
                column_names=DIVERSION_COLUMNS,
                compute_shortage_ratio=True,
            ).items():
                all_div.setdefault(key, []).append(df.values)

            for key, df in process_reservoir_csv(
                dfs["reservoirs"], column_names=RESERVOIR_COLUMNS
            ).items():
                all_res.setdefault(key, []).append(df.values)

            for p in (out_file, mss_file, flo_path):
                p.unlink()

        return {
            **{f"div_{k}": np.stack(v) for k, v in all_div.items()},
            **{f"res_{k}": np.stack(v) for k, v in all_res.items()},
        }

    return _run
