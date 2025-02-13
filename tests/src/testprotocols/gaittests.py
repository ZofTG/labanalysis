"""gait analysis testing module"""

#! IMPORTS


import sys
from os.path import dirname, join, sep

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from src.labanalysis import *

__all__ = ["test_gaits"]


#! FUNCTION


def test_run():
    """test the run test"""
    path = join(dirname(__file__), "gaitanalysis_data")
    files = [i for i in get_files(path, ".tdf", False) if "run_test" in i]
    figures = {}
    for file in files:
        print(f"\nTEST {file}")
        test = RunningTest.from_tdf_file(
            file=file,
            algorithm="kinematics",
            left_heel="l_heel",
            right_heel="r_heel",
            left_toe="l_toe",
            right_toe="r_toe",
            left_meta_head="l_met",
            right_meta_head="r_met",
            grf="fRes",
        )
        for algorithm in ["kinematics", "kinetics"]:
            test.set_algorithm(algorithm)  # type: ignore
            name = file.rsplit(sep, 1)[-1].rsplit(".", 1)[0]

            # summary plots
            fig, dfr = test.summary()
            for key, val in fig.items():
                title = val.layout.title.text + f" ({test.algorithm})"  # type: ignore
                title = " ".join([name, title])
                val.update_layout(title=title)
                figures[title] = val

            # results plot
            fig, dfr = test.results()
            title = [name, fig.layout.title.text + f" ({test.algorithm})"]  # type: ignore
            title = " ".join(title)
            fig.update_layout(title=title)
            figures[title] = fig

    # store the figures
    for key, val in figures.items():
        val.write_html(join(path, key + ".html"))
    print("RUNNING TEST COMPLETED")


def test_walk():
    """test the walk test"""
    path = join(dirname(__file__), "gaitanalysis_data")
    files = [i for i in get_files(path, ".tdf", False) if "walk_test" in i]
    for file in files:
        print(f"\nTEST {file}")
        frame = StateFrame.from_tdf_file(file)
        test = WalkingTest(
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
        for algorithm in ["kinetics", "kinematics"]:
            test.set_algorithm(algorithm)  # type: ignore
            fig, dfr = test.summary()
            for key, val in fig.items():
                title = val.layout.title.text + f" ({test.algorithm})"  # type: ignore
                val.update_layout(title=title)
                val.show()
            fig, dfr = test.results()
            title = fig.layout.title.text + f" ({test.algorithm})"  # type: ignore
            fig.update_layout(title=title)
            fig.show()
    print("WALKING TEST COMPLETED")


def test_gaits():
    """test the gaittests module"""
    test_run()
    test_walk()


if __name__ == "__main__":
    test_gaits()
