import shutil
import subprocess
from datetime import datetime
from pathlib import Path


class WRAPExecutionSlot:
    """Owns the lifecycle of one WRAP execution directory (execution_folder_i).

    One slot corresponds to one parallel worker process. Static WAM config files
    (.dat, .dis, .eva, .fad, .his) are copied once in setup(); per-realization
    inputs (.FLO, and eventually per-realization .EVA) are staged in run().
    """

    _STATIC_EXTENSIONS = {".dat", ".dis", ".eva", ".fad", ".his"}

    def __init__(self, slot_dir, wam_path, wrap_exe_path, base_name="C3"):
        self.slot_dir = Path(slot_dir)
        self.wam_path = Path(wam_path)
        self.wrap_exe_path = Path(wrap_exe_path)
        self.base_name = base_name

    def setup(self):
        """Create the slot directory and populate it with static WAM config files."""
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        for f in self.wam_path.iterdir():
            if f.suffix.lower() in self._STATIC_EXTENSIONS:
                shutil.copy2(f, self.slot_dir / f.name)

    def teardown(self):
        """Remove the slot directory and all its contents."""
        if self.slot_dir.exists():
            shutil.rmtree(self.slot_dir)

    def run(self, flo_path, eva_df=None) -> str:
        """Stage FLO (and optionally a per-realization EVA), execute WRAP, rename outputs.

        Returns the stem of the FLO file (used to locate the renamed .OUT/.MSS).
        The eva_df parameter is a no-op seam for the future per-realization EVA feature.
        """
        flo_path = Path(flo_path)
        flo_name = flo_path.stem
        shutil.copy2(flo_path, self.slot_dir / f"{self.base_name}.FLO")
        if eva_df is not None:
            from toolkit.wrap.io import df_to_evp
            df_to_evp(eva_df, self.slot_dir / f"{self.base_name}.EVA")
        subprocess.run(
            f"(echo {self.base_name} && echo {self.base_name}) | wine64 {self.wrap_exe_path}",
            cwd=self.slot_dir,
            shell=True,
        )
        now = datetime.now()
        print(f"[{now.strftime('%H:%M:%S')}] {self.base_name} done ({self.slot_dir.name})")
        (self.slot_dir / f"{self.base_name}.OUT").rename(self.slot_dir / f"{flo_name}.OUT")
        (self.slot_dir / f"{self.base_name}.MSS").rename(self.slot_dir / f"{flo_name}.MSS")
        (self.slot_dir / f"{self.base_name}.FLO").unlink()
        return flo_name
