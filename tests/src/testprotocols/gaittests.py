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
    path = join(dirname(__file__), "gaitanalysis_data")
    files = [join(path, "run_test.tdf"), join(path, "run_test_2.tdf")]
    for file in files:
        frame = StateFrame.from_tdf_file(file)
        run_test = RunningTest(
            frame=frame,
            algorithm="kinematics",
            left_heel="l_heel",
            right_heel="r_heel",
            left_toe="l_toe",
            right_toe="r_toe",
            left_meta_head="l_met",
            right_meta_head="r_met",
            grf="fRes",
        )
        fig, dfr = run_test.summary()
        fig, dfr = run_test.results()
        fig_name = file.replace(".tdf", "_kinematics.html")
        fig.write_html(fig_name)
        run_test.set_algorithm("kinetics")
        fig, dfr = run_test.results()
        fig_name = file.replace(".tdf", "_kinetics.html")
        fig.write_html(fig_name)
    print("RUNNING TEST COMPLETED")


def test_walk():
    """test the run test"""
    print("\nTEST WALK")
    # TODO: add walk test
    print("WALK TEST NOT IMPLEMENTED")
    print("WALK TEST COMPLETED")


def test_gaits():
    """test the jumptests module"""
    test_run()
    test_walk()


if __name__ == "__main__":
    test_gaits()
