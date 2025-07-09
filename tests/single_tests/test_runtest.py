"""read_tdf function testing module"""

#! IMPORTS

import sys
from os.path import abspath, dirname, join

sys.path += [dirname(dirname(dirname(abspath(__file__))))]

from src.labanalysis import (
    RunningTest,
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)

#! MAIN


def run_test():
    # get the file
    files_path = join(dirname(dirname(__file__)), "assets")
    tdf_file_path = join(files_path, "gaitanalysis_data", "run_test_0.tdf")

    # check the file is read correctly
    try:
        run_test = RunningTest.from_tdf_file(
            tdf_file_path,
            "kinematics",
            "lHeel",
            "rHeel",
            "lToe",
            "rToe",
            "lMid",
            "rMid",
            "fRes",
            DEFAULT_MINIMUM_CONTACT_GRF_N,
            DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
            "Y",
            "Z",
        )
        run_test_fig, run_test_df = run_test.results()
    except Exception as exc:
        raise RuntimeError(
            "RunningTest generation failed with kinematics algorithm"
        ) from exc
    print("read_tdf worked as expected.")


if __name__ == "__main__":
    run_test()
