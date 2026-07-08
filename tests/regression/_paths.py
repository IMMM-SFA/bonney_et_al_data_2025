"""Path constants for WRAP regression tests."""
from pathlib import Path
from toolkit import repo_data_path, outputs_path

FILTER_NAME = "all_models"
BASIN_NAME = "Colorado"
N_REALIZATIONS = 10
REGRESSION_SEED = 20250707

FLO_FILE = Path(repo_data_path) / "WRAP" / "basin_wams" / "colo-full" / "C3.FLO"
WAM_PATH = FLO_FILE.parent
WRAP_SIM_PATH = Path(repo_data_path) / "WRAP" / "SIM.exe"
WRAP_EXEC_PATH = Path(repo_data_path) / "WRAP" / "wrap_execution_directories"

SYNTHETIC_NC_PATH = (
    outputs_path / "bayesian_hmm" / FILTER_NAME / BASIN_NAME.lower()
    / f"{FILTER_NAME}_{BASIN_NAME.lower()}_synthetic_dataset.nc"
)
MODEL_PATH = (
    outputs_path / "bayesian_hmm" / FILTER_NAME / BASIN_NAME.lower()
    / f"{BASIN_NAME}_{FILTER_NAME}_model"
)

BASELINE_DIR = Path(__file__).parent / "baseline"
