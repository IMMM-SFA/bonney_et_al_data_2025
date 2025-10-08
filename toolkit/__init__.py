__version__ = '0.1.0'

__all__ = [
    'repo_data_path',
    'outputs_path',
]

import json
from os.path import abspath, dirname, join

repo_data_path = join(dirname(abspath(str(__file__))), "..", "data")
outputs_path = join(dirname(abspath(str(__file__))), "..", "outputs")

