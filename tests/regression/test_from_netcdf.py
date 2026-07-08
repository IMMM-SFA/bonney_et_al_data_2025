"""
Regression test: 10 realizations from an existing synthetic streamflow NetCDF
through the full WRAP pipeline, compared against a stored baseline.

Catches regressions in fix_cols, df_to_flo, WRAPExecutionSlot, out_to_dfs,
and process_diversion_csv / process_reservoir_csv.

Prerequisites:
  - Run workflow stages I–II to produce the synthetic NetCDF.
  - Run once with --generate-baseline to create the local baseline file.
  - wine64 + data/WRAP/SIM.exe must be present.
"""
import pytest

from toolkit.data.io import load_netcdf_format
from toolkit.wrap.execution_slot import LocalWRAPExecutionSlot
from ._paths import SYNTHETIC_NC_PATH, WAM_PATH, WRAP_SIM_PATH, WRAP_EXEC_PATH, BASELINE_DIR, N_REALIZATIONS
from ._helpers import requires_wrap, compare_or_save

BASELINE_PATH = BASELINE_DIR / "from_netcdf.npz"
SLOT_DIR = WRAP_EXEC_PATH / "regression_slot_netcdf"


@requires_wrap
def test_wrap_pipeline_from_netcdf(run_wrap_pipeline, generate_baseline, tmp_path):
    """Run 10 pre-generated realizations through WRAP and compare against baseline."""
    if not SYNTHETIC_NC_PATH.exists():
        pytest.skip(f"Synthetic NetCDF not found: {SYNTHETIC_NC_PATH}")
    if not WAM_PATH.exists():
        pytest.skip(f"WAM path not found: {WAM_PATH}")

    data = load_netcdf_format(str(SYNTHETIC_NC_PATH))
    streamflow = data["streamflow"][:N_REALIZATIONS]
    streamflow_index = data["streamflow_index"]
    streamflow_columns = data["streamflow_columns"]

    slot = LocalWRAPExecutionSlot(SLOT_DIR, WAM_PATH, WRAP_SIM_PATH)
    slot.teardown()
    slot.setup()
    try:
        outputs = run_wrap_pipeline(
            streamflow, streamflow_index, streamflow_columns,
            slot, tmp_path,
        )
    finally:
        slot.teardown()

    saved = compare_or_save(outputs, BASELINE_PATH, generate_baseline)
    if not saved:
        pytest.skip("Baseline written. Re-run without --generate-baseline to compare.")
