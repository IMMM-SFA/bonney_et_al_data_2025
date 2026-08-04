"""
Shared output-path construction for basin/ensemble-filter workflow outputs.
"""
from pathlib import Path

from toolkit import outputs_path


def basin_filter_dir(filter_name: str, basin_name: str) -> Path:
    """Per-basin, per-filter output directory under outputs/bayesian_hmm/."""
    return outputs_path / "bayesian_hmm" / filter_name / basin_name.lower()


def synthetic_dataset_path(filter_name: str, basin_name: str) -> Path:
    """Path to the synthetic streamflow (+ WRAP outputs, once appended) NetCDF
    for one basin/filter combination."""
    return basin_filter_dir(filter_name, basin_name) / f"{filter_name}_{basin_name.lower()}_synthetic_dataset.nc"
