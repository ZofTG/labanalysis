"""testing utils module"""

#! IMPORTS


import sys
from os.path import dirname, join

import numpy as np

sys.path += [join(dirname(dirname(dirname(__file__))), "src")]

from labanalysis import *

__all__ = ["test_utils"]


#! FUNCTION


def test_utils():
    """test the regression module"""
    raw = np.arange(51)
    splitted = split_data(raw, {"A": 0.5, "B": 0.25, "C": 0.25}, 5)  # type: ignore
    print(f"RAW: {raw}")
    print(f"SPLITTED: {splitted}")


if __name__ == "__main__":
    test_utils()
