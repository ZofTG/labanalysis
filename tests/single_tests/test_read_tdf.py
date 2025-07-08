"""read_tdf function testing module"""

#! IMPORTS

import sys
from os.path import dirname, join

sys.path += [join(dirname(dirname(dirname(__file__))))]

import pytest

from src.labanalysis.io.read.btsbioengineering import read_tdf

#! MAIN

if __name__ == "__main__":

    # get the file
    files_path = join(dirname(__file__), "assets")
    tdf_file_path = join(files_path, "gaitanalysis_data", "run_test_0.tdf")

    # check the file is read correctly
    with pytest.raises(RuntimeError):
        tdf = read_tdf(tdf_file_path)
    print("read_tdf worked as expected.")
