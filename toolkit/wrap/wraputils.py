import os
import zipfile
import shutil
import pandas as pd
from toolkit.wrap.wrapdriver import WRAPDriver
from toolkit.wrap.io import out_to_csvs, df_to_flo
from toolkit.emulator.processing import process_diversion_csv, process_reservoir_csv

def clean_folders(
    execution_directory=None,
    diversions_directory=None,
    reservoirs_directory=None,
    flo_directory=None,
    out_files_directory=None):
    """Removes OUT, MSS, and FLO files from subdirectories

    Parameters
    ----------
    execution_directory : str
        directory that contains multiple directories named "execution_folder_i" 
        where i is some integer.
    """

    # clean execution folder
    if execution_directory is not None:
        for directory in next(os.walk(execution_directory))[1]:
            for file in next(os.walk(os.path.join(execution_directory,directory)))[2]:
                if "OUT" in file or "MSS" in file or "FLO" in file:
                    os.remove(os.path.join(execution_directory,directory, file))
                
    # clean diversions folder
    if diversions_directory is not None:
        for file in next(os.walk(diversions_directory))[2]:
            if ".csv" in file: 
                os.remove(os.path.join(diversions_directory, file))
                
    # clean reservoirs folder
    if reservoirs_directory is not None:
        for file in next(os.walk(reservoirs_directory))[2]:
            if ".csv" in file: 
                os.remove(os.path.join(reservoirs_directory, file))
                
    # clean flo folder
    if flo_directory is not None:
        for file in next(os.walk(flo_directory))[2]:
            if "FLO" in file: 
                os.remove(os.path.join(flo_directory, file))
                
    # clean out files folder
    if out_files_directory is not None:
        for file in next(os.walk(out_files_directory))[2]:
            if "OUT" in file: 
                os.remove(os.path.join(out_files_directory, file))

def split_into_sublists(lst, n):
    """Basic function to split a list into n sublists of roughly equal size.
    Used to prepare flofile list for multiprocessing.
    """
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    sublists = [lst[i*chunk_size+min(i,remainder):(i+1)*chunk_size+min(i+1,remainder)] for i in range(n)]
    assert [item for sublist in sublists for item in sublist] == lst
    return sublists

def fix_cols(basin_name, synth_flow, default_flo):
    """Fix column names for synthetic flow"""
    if basin_name == "Colorado":
        # add in historical flow at these two gage sites since they are outside of the CRB
        synth_flow["INL10000"] = default_flo["INL10000"].astype(float)
        synth_flow["INL20000"] = default_flo["INL20000"].astype(float)
    return synth_flow


# Define a pipeline function to be utilized by multiprocessing
def wrap_pipeline(
    flo_files,
    wrap_execution_folder, 
    wrap_sim_path, 
    diversions_csvs_path,
    reservoirs_csvs_path,
    out_zip_path,
    synthetic_flo_output_path,
    out_files_path=None):
    """For each FLO file: copy FLO file to execution folder, run wrap, 
    process the OUT file, compresses the original OUT file, and delete the 
    OUT, MSS, and FLO file for the run.

    Parameters
    ----------
    flo_files : list[str]
        list of .FLO file paths
    wrap_execution_folder : str
        folder that pipeline will run wrap inside of
        outputs are left in this folder as well.
        Needs to contain the 5 configuration files for
        running WRAP.
    """
    driver = WRAPDriver(wrap_sim_path)
    count = 0
    for flo_file in flo_files:
        # copy flo file to execution folder
        flo_name = flo_file.split(".")[0]
        flo_file = os.path.join(synthetic_flo_output_path, flo_file)
        
        # execute wrap
        driver.execute(flo_file=flo_file,
                    execution_folder=wrap_execution_folder)
        
        # process .OUT file
        out_file = os.path.join(wrap_execution_folder, f"{flo_name}.OUT")
        mss_file = os.path.join(wrap_execution_folder, f"{flo_name}.MSS")
        out_to_csvs(out_file, wrap_execution_folder, csvs_to_write=["diversions", "reservoirs"])
        
        # process diversion file
        diversions_path = os.path.join(wrap_execution_folder, f"{flo_name}_diversions.csv")
        diversions_df = pd.read_csv(diversions_path)
        diversion_data = process_diversion_csv(
            diversions_df, 
            column_names=["diversion_or_energy_shortage", "diversion_or_energy_target"], 
            compute_shortage_ratio=True)
        for key, value in diversion_data.items():
            value.to_csv(os.path.join(diversions_csvs_path, f"{flo_name}_{key}.csv"))
        
        # process reservoir file
        reservoirs_path = os.path.join(wrap_execution_folder, f"{flo_name}_reservoirs.csv")
        reservoirs_df = pd.read_csv(reservoirs_path)
        reservoir_data = process_reservoir_csv(
            reservoirs_df,
            column_names=[
                "reservoir_water_surface_elevation",
                "reservoir_storage_capacity",
                "inflows_to_reservoir_from_stream_flow_depletions",
                "inflows_to_reservoir_from_releases_from_other_reservoirs",
                "reservoir_net_evaporation_precipitation_volume",
                "energy_generated",
                "reservoir_releases_accessible_to_hydroelectric_power_turbines",
                "reservoir_releases_not_accessible_to_hydroelectric_power_turbines"
            ])
        for key, value in reservoir_data.items():
            value.to_csv(os.path.join(reservoirs_csvs_path, f"{flo_name}_{key}.csv"))
        
        # copy OUT file to out_files_path if specified
        # if out_files_path is not None:
        #     out_dest = os.path.join(out_files_path, f"{flo_name}.OUT")
        #     shutil.copy2(out_file, out_dest)
        
        # # compress out file
        # zip_file = os.path.join(out_zip_path, f"{flo_name}.OUT.zip")
        # with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as myzip:
        #     myzip.write(out_file)

        # delete files
        os.remove(out_file)
        os.remove(mss_file)
        os.remove(diversions_path)
        os.remove(reservoirs_path)
        count += 1
        print(count, flo_file, f"process: {str(wrap_execution_folder)[-1]}")
        
def process_ensemble_member(args):
    """Worker function to process a single ensemble member"""
    ens, streamflow, streamflow_index, streamflow_columns, basin_name, flo_df, synthetic_flo_output_path = args
    
    # Get synthetic flow for this ensemble
    data = streamflow[ens, :, :]
    synth_flow = pd.DataFrame(
        data, 
        index=streamflow_index,
        columns=streamflow_columns,
    )
    
    synth_flow.index = pd.to_datetime(synth_flow.index)
    flo_df.index = synth_flow.index
    # Address missing columns
    fix_cols(basin_name, synth_flow, flo_df)
    
    out_name = synthetic_flo_output_path / f"synthflow_{ens:02d}.FLO"
    df_to_flo(synth_flow, out_name)
    
    return f"Completed ensemble member {ens+1}"