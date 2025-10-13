import shutil
import os
import subprocess
from datetime import datetime
from os.path import join


class WRAPDriver:
    def __init__(self, wrap_exe_path):
        self.wrap_exe_path = wrap_exe_path
        
    def execute(self, base_name="C3", flo_file=None, execution_folder=None):
        # run wrap using wrap files in input folder and
        # put output files in output_folder
        # parent_files = os.listdir(parent_folder)
        if flo_file is not None:
            flo_name = os.path.basename(flo_file).split(".")[0]
            shutil.copyfile(flo_file, join(execution_folder, f"{base_name}.FLO"))
        
        # execute wrap
        commandline = f"(echo {base_name} && echo {base_name}) | taskset -c 0-63 wine64 {self.wrap_exe_path}"
        cmdmsg = subprocess.run(
            commandline, 
            cwd=execution_folder, 
            shell=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL)
        # Print periodic status updates
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        statusmsg = base_name + " is done!"
        print(statusmsg)
        if flo_file is not None:
            os.rename(join(execution_folder, f"{base_name}.OUT"), join(execution_folder, f"{flo_name}.OUT"))
            os.rename(join(execution_folder, f"{base_name}.MSS"), join(execution_folder, f"{flo_name}.MSS"))
            os.remove(join(execution_folder, f"{base_name}.FLO"))