import os
import pandas as pd
from pathlib import Path
from toolkit.wrap.io import out_to_dfs, df_to_flo
from toolkit.emulator.processing import process_diversion_csv, process_reservoir_csv


def clean_folders(
    diversions_directory=None,
    reservoirs_directory=None,
    flo_directory=None,
):
    """Remove CSV and FLO files from output directories before a fresh run.

    Execution slot directories are managed separately via WRAPExecutionSlot.teardown().
    """
    if diversions_directory is not None:
        for file in next(os.walk(diversions_directory))[2]:
            if ".csv" in file:
                os.remove(os.path.join(diversions_directory, file))

    if reservoirs_directory is not None:
        for file in next(os.walk(reservoirs_directory))[2]:
            if ".csv" in file:
                os.remove(os.path.join(reservoirs_directory, file))

    if flo_directory is not None:
        for file in next(os.walk(flo_directory))[2]:
            if "FLO" in file:
                os.remove(os.path.join(flo_directory, file))


def split_into_sublists(lst, n):
    """Split a list into n sublists of roughly equal size for multiprocessing."""
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    sublists = [lst[i*chunk_size+min(i,remainder):(i+1)*chunk_size+min(i+1,remainder)] for i in range(n)]
    assert [item for sublist in sublists for item in sublist] == lst
    return sublists


def fix_cols(basin_config, synth_flow, default_flo):
    """Splice in historical flow for gages outside the basin model boundary."""
    for gage in basin_config.get("external_gages", []):
        synth_flow[gage] = default_flo[gage].astype(float)
    return synth_flow


def wrap_pipeline(
    slot,
    flo_files,
    diversions_csvs_path,
    reservoirs_csvs_path,
    synthetic_flo_output_path,
):
    """For each FLO file: run WRAP via slot, process the OUT file into CSVs, and clean up.

    Parameters
    ----------
    slot : WRAPExecutionSlot
        Pre-setup execution slot for this worker process.
    flo_files : list[str]
        FLO file names (not full paths) to process sequentially in this slot.
    """
    count = 0
    for flo_file in flo_files:
        flo_path = Path(synthetic_flo_output_path) / flo_file
        flo_name = slot.run(flo_path)

        out_file = slot.slot_dir / f"{flo_name}.OUT"
        mss_file = slot.slot_dir / f"{flo_name}.MSS"
        dfs = out_to_dfs(out_file)

        diversion_data = process_diversion_csv(
            dfs["diversions"],
            column_names=["diversion_or_energy_shortage", "diversion_or_energy_target"],
            compute_shortage_ratio=True)
        for key, value in diversion_data.items():
            value.to_csv(Path(diversions_csvs_path) / f"{flo_name}_{key}.csv")

        reservoir_data = process_reservoir_csv(
            dfs["reservoirs"],
            column_names=[
                "reservoir_water_surface_elevation",
                "reservoir_storage_capacity",
                "inflows_to_reservoir_from_stream_flow_depletions",
                "inflows_to_reservoir_from_releases_from_other_reservoirs",
                "reservoir_net_evaporation_precipitation_volume",
                "energy_generated",
                "reservoir_releases_accessible_to_hydroelectric_power_turbines",
                "reservoir_releases_not_accessible_to_hydroelectric_power_turbines",
            ])
        for key, value in reservoir_data.items():
            value.to_csv(Path(reservoirs_csvs_path) / f"{flo_name}_{key}.csv")

        out_file.unlink()
        mss_file.unlink()
        count += 1
        print(count, flo_file, f"slot: {slot.slot_dir.name}")


def process_ensemble_member(args):
    """Worker function to process a single ensemble member"""
    ens, streamflow, streamflow_index, streamflow_columns, basin_config, flo_df, synthetic_flo_output_path = args

    data = streamflow[ens, :, :]
    synth_flow = pd.DataFrame(
        data,
        index=streamflow_index,
        columns=streamflow_columns,
    )

    synth_flow.index = pd.to_datetime(synth_flow.index)
    flo_df.index = synth_flow.index
    fix_cols(basin_config, synth_flow, flo_df)

    out_name = synthetic_flo_output_path / f"synthflow_{ens:02d}.FLO"
    df_to_flo(synth_flow, out_name)

    return f"Completed ensemble member {ens+1}"
