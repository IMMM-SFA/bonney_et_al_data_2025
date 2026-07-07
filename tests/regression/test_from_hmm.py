"""
Regression test: generate 10 realizations from a trained BHMM model, run them
through the full WRAP pipeline, and compare against a stored baseline.

Covers the complete pipeline from HMM generation onward: posterior sampling,
annual streamflow synthesis, KNN disaggregation, FLO writing, WRAP execution,
and output parsing. REGRESSION_SEED pins the random state so the same 10
realizations are produced on every run.

Prerequisites:
  - Run workflow stages I–II to train and save the model.
  - Run once with --generate-baseline to create the local baseline file.
  - wine64 + data/WRAP/SIM.exe must be present.
"""
import pytest

from toolkit.hmm.model import BayesianStreamflowHMM
from toolkit.wrap.execution_slot import LocalWRAPExecutionSlot
from ._paths import (
    MODEL_PATH, WAM_PATH, WRAP_SIM_PATH, WRAP_EXEC_PATH,
    BASELINE_DIR, N_REALIZATIONS, REGRESSION_SEED,
)
from ._helpers import requires_wrap, compare_or_save

BASELINE_PATH = BASELINE_DIR / "from_hmm.npz"
SLOT_DIR = WRAP_EXEC_PATH / "regression_slot_hmm"


@requires_wrap
def test_wrap_pipeline_from_hmm(run_wrap_pipeline, historical_flo, basin_config, generate_baseline, tmp_path):
    """Generate 10 realizations from a trained BHMM and run them through WRAP."""
    if not MODEL_PATH.with_suffix(".nc").exists():
        pytest.skip(f"Trained model not found: {MODEL_PATH}.nc")
    if not WAM_PATH.exists():
        pytest.skip(f"WAM path not found: {WAM_PATH}")

    model = BayesianStreamflowHMM.load(str(MODEL_PATH))
    outflow_index = historical_flo.columns.tolist().index(basin_config["gage_name"])

    synthetic = model.generate_synthetic_streamflow(
        start_year=2020,
        historical_monthly_data=historical_flo.values,
        n_ensembles=N_REALIZATIONS,
        random_seed=REGRESSION_SEED,
        site_names=historical_flo.columns.tolist(),
        time_index=historical_flo.index.tolist(),
        outflow_index=outflow_index,
    )
    streamflow = synthetic["streamflow"]
    streamflow_index = synthetic["streamflow_index"]
    streamflow_columns = synthetic["streamflow_columns"]

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

    # Include the generated streamflow so HMM-generation regressions are
    # distinguishable from WRAP-pipeline regressions in the baseline diff.
    outputs["streamflow"] = streamflow

    saved = compare_or_save(outputs, BASELINE_PATH, generate_baseline)
    if not saved:
        pytest.skip("Baseline written. Re-run without --generate-baseline to compare.")
