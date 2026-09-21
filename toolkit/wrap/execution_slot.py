import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


class WRAPExecutionSlot(ABC):
    """Abstract base: owns the lifecycle of one WRAP execution directory.

    Subclasses implement _invoke_wrap() to run WRAP via local Wine or a
    Singularity container. All file staging and output renaming is handled here.

    One slot corresponds to one parallel worker process. Static WAM config files
    (.dat, .dis, .eva, .fad, .his) are copied once in setup(); per-realization
    inputs (.FLO, and eventually per-realization .EVA) are staged in run().
    """

    _STATIC_EXTENSIONS = {".dat", ".dis", ".eva", ".fad", ".his"}

    def __init__(self, slot_dir, wam_path, base_name="C3", dat_suffix=".dat"):
        """
        Parameters
        ----------
        dat_suffix : str, default ".dat"
            Selects which .DAT variant to stage as the WAM's input file, for basins
            with more than one available.
        """
        self.slot_dir = Path(slot_dir)
        self.wam_path = Path(wam_path)
        self.base_name = base_name
        self.dat_suffix = dat_suffix

    def setup(self):
        """Create the slot directory and populate it with static WAM config files."""
        self.slot_dir.mkdir(parents=True, exist_ok=True)

        dat_target_name = f"{self.base_name}{self.dat_suffix}".lower()
        dat_matches = [f for f in self.wam_path.iterdir() if f.name.lower() == dat_target_name]
        if len(dat_matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 file named {dat_target_name!r} "
                f"(case-insensitive) in {self.wam_path}, found {len(dat_matches)}: {dat_matches}"
            )
        shutil.copy2(dat_matches[0], self.slot_dir / f"{self.base_name}.dat")

        for f in self.wam_path.iterdir():
            if f.suffix.lower() in self._STATIC_EXTENSIONS and f.suffix.lower() != ".dat":
                shutil.copy2(f, self.slot_dir / f.name)

    def teardown(self):
        """Remove the slot directory and all its contents."""
        if self.slot_dir.exists():
            shutil.rmtree(self.slot_dir)

    def run(self, flo_path, eva_df=None) -> str:
        """Stage inputs, invoke WRAP, rename outputs.

        Returns the stem of the FLO file, used to locate the renamed .OUT/.MSS.
        eva_df is a seam for the future per-realization EVA feature.
        """
        flo_path = Path(flo_path)
        flo_name = flo_path.stem
        shutil.copy2(flo_path, self.slot_dir / f"{self.base_name}.FLO")
        if eva_df is not None:
            from toolkit.wrap.io import df_to_evp
            df_to_evp(eva_df, self.slot_dir / f"{self.base_name}.EVA")
        self._invoke_wrap()
        now = datetime.now()
        out_path = self.slot_dir / f"{self.base_name}.OUT"
        if not out_path.exists():
            raise RuntimeError(
                f"WRAP produced no {out_path.name} in {self.slot_dir} for {flo_name}; "
                "SIM.exe crashed or was never invoked"
            )
        print(f"[{now.strftime('%H:%M:%S')}] {self.base_name} done ({self.slot_dir.name})")
        out_path.rename(self.slot_dir / f"{flo_name}.OUT")
        (self.slot_dir / f"{self.base_name}.MSS").rename(self.slot_dir / f"{flo_name}.MSS")
        (self.slot_dir / f"{self.base_name}.FLO").unlink()
        return flo_name

    @abstractmethod
    def _invoke_wrap(self):
        """Execute WRAP for one realization. Called by run() after inputs are staged."""


class LocalWRAPExecutionSlot(WRAPExecutionSlot):
    """Runs WRAP using wine64 installed directly on the host."""

    def __init__(self, slot_dir, wam_path, wrap_exe_path, base_name="C3", dat_suffix=".dat"):
        super().__init__(slot_dir, wam_path, base_name, dat_suffix)
        self.wrap_exe_path = Path(wrap_exe_path)

    def _invoke_wrap(self):
        # SIM.exe writes prompts to the Windows console (CONOUT$). Wine only creates a console
        # when a standard stream is a terminal, so under nohup/Slurm (all streams redirected)
        # the Fortran runtime aborts with "error during write, unit -1, file CONOUT$". Giving
        # stdout/stderr a pseudo-terminal keeps the console available; stdin stays a pipe so
        # the two base-name answers are followed by EOF as before.
        master, slave = os.openpty()
        try:
            proc = subprocess.Popen(
                ["wine64", str(self.wrap_exe_path)],
                cwd=self.slot_dir,
                stdin=subprocess.PIPE,
                stdout=slave,
                stderr=slave,
                env={**os.environ, "WINEDEBUG": os.environ.get("WINEDEBUG", "-all")},
            )
        finally:
            os.close(slave)
        proc.stdin.write(f"{self.base_name}\n{self.base_name}\n".encode())
        proc.stdin.close()
        try:
            while True:  # drain console output so SIM.exe never blocks on a full pty buffer
                try:
                    if not os.read(master, 65536):
                        break
                except OSError:
                    break
        finally:
            os.close(master)
        proc.wait()


class SingularityWRAPExecutionSlot(WRAPExecutionSlot):
    """Runs WRAP inside a persistent Singularity instance (one instance per slot).

    The slot directory is bind-mounted into the container at /wrap/slot.
    SIM.exe must be baked into the image at /wrap/SIM.exe.

    setup() starts the instance; teardown() stops it and removes the slot directory.
    Between calls to run(), the instance stays alive so there is no per-realization
    container startup cost.
    """

    _WRAP_EXE = "/wrap/SIM.exe"

    def __init__(self, slot_dir, wam_path, sif_path, base_name="C3", dat_suffix=".dat"):
        super().__init__(slot_dir, wam_path, base_name, dat_suffix)
        self.sif_path = Path(sif_path)
        # Singularity instance names must be alphanumeric + underscores
        self._instance_name = re.sub(r"[^a-zA-Z0-9]", "_", self.slot_dir.name)

    def setup(self):
        super().setup()
        subprocess.run(
            [
                "singularity", "instance", "start",
                "--bind", f"{self.slot_dir}:/wrap/slot",
                str(self.sif_path),
                self._instance_name,
            ],
            check=True,
        )

    def teardown(self):
        subprocess.run(
            ["singularity", "instance", "stop", self._instance_name],
            check=False,  # tolerate already-stopped instance
        )
        super().teardown()

    def _invoke_wrap(self):
        subprocess.run(
            [
                "singularity", "exec", f"instance://{self._instance_name}",
                "bash", "-c",
                f"cd /wrap/slot && (echo {self.base_name} && echo {self.base_name}) | wine64 {self._WRAP_EXE}",
            ],
            check=True,
        )
