"""read_tdf function testing module"""

#! IMPORTS

import sys
from os.path import dirname, join, abspath

sys.path += [join(dirname(dirname(dirname(abspath(__file__)))))]

from src.labanalysis.io.read.btsbioengineering import read_tdf

#! MAIN


def run_test():
    # get the file
    files_path = join(dirname(dirname(__file__)), "assets")
    tdf_file_path = join(files_path, "gaitanalysis_data", "run_test_0.tdf")

    # check the file is read correctly
    try:
        tdf = read_tdf(tdf_file_path)
    except Exception as exc:
        raise RuntimeError("read_tdf generated an error") from exc
    print("read_tdf worked as expected.")


if __name__ == "__main__":
    run_test()
