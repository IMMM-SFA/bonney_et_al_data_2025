__version__ = '0.1.0'

__all__ = [
    'repo_data_path',
    'outputs_path',
]

import json
from os.path import abspath, dirname, join
from pathlib import Path

repo_data_path = Path(join(dirname(abspath(str(__file__))), "..", "data"))
outputs_path = Path(join(dirname(abspath(str(__file__))), "..", "outputs"))

