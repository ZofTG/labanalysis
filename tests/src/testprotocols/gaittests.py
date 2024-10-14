"""gait analysis testing module"""

#! IMPORTS


import sys
from os.path import dirname, join

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from src.labanalysis import *

__all__ = ["test_gaits"]


#! FUNCTION


def test_run():
    """test the run test"""
    print("\nTEST RUN")
    file = join(dirname(__file__), "gaitanalysis_data", "run_test.tdf")
    test = GaitTest.from_tdf_file(
        file=file,
        grf_label="fRes",
        rheel_label="r_heel",
        lheel_label="l_heel",
        rtoe_label="r_toe",
        ltoe_label="l_toe",
        lmid_label="l_met",
        rmid_label="r_met",
        height_thresh=0.02,
        force_thresh=30,
    )
    print("STEPS SUMMARY")
    print(test.steps_summary())


def test_gaits():
    """test the jumptests module"""
    test_run()


if __name__ == "__main__":
    test_gaits()
