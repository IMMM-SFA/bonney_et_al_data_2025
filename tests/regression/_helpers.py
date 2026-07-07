"""Non-fixture helpers for WRAP regression tests."""
import shutil
import numpy as np
import pytest

from ._paths import WRAP_SIM_PATH, BASELINE_DIR

# ── WRAP availability ─────────────────────────────────────────────────────────

_wrap_available = shutil.which("wine64") is not None and WRAP_SIM_PATH.exists()

requires_wrap = pytest.mark.skipif(
    not _wrap_available,
    reason="WRAP requires wine64 in PATH and data/WRAP/SIM.exe",
)

# ── Column lists (must match wrap_pipeline in wraputils.py) ──────────────────

DIVERSION_COLUMNS = [
    "diversion_or_energy_shortage",
    "diversion_or_energy_target",
]
RESERVOIR_COLUMNS = [
    "reservoir_water_surface_elevation",
    "reservoir_storage_capacity",
    "inflows_to_reservoir_from_stream_flow_depletions",
    "inflows_to_reservoir_from_releases_from_other_reservoirs",
    "reservoir_net_evaporation_precipitation_volume",
    "energy_generated",
    "reservoir_releases_accessible_to_hydroelectric_power_turbines",
    "reservoir_releases_not_accessible_to_hydroelectric_power_turbines",
]


# ── Baseline comparison ───────────────────────────────────────────────────────

def compare_or_save(outputs, baseline_path, generate_baseline_flag):
    """Save outputs as a compressed baseline or compare against an existing one.

    Returns True if a comparison was performed, False if a baseline was written.
    """
    if generate_baseline_flag or not baseline_path.exists():
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(baseline_path, **outputs)
        return False
    baseline = dict(np.load(baseline_path))
    missing = set(outputs) - set(baseline)
    extra = set(baseline) - set(outputs)
    assert not missing, f"Keys in outputs but not in baseline: {missing}"
    assert not extra, f"Keys in baseline but not in outputs: {extra}"
    for key, actual in outputs.items():
        np.testing.assert_array_equal(
            actual, baseline[key],
            err_msg=f"Regression failure: '{key}' does not match baseline",
        )
    return True
