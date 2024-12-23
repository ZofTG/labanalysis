"""gait analysis testing module"""

#! IMPORTS


import sys
from os.path import dirname, join

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

import pandas as pd
from src.labanalysis import *

__all__ = ["test_gaits"]


#! FUNCTION


def test_run():
    """test the run test"""
    print("\nTEST RUN")
    path = join(dirname(__file__), "gaitanalysis_data")
    file = join(path, "run_test.tdf")
    frame = StateFrame.from_tdf_file(file)
    mcols = []
    for i in frame.markers.columns:
        lbl, axis, unit = i
        root = lbl.split("_")[-2]
        root += lbl.split("_")[-1].capitalize().replace("Met", "Mid")
        mcols += [(root, axis, unit)]
    frame._markers.columns = pd.MultiIndex.from_tuples(mcols)
    valid_markers = ["lHeel", "lToe", "lMid", "rHeel", "rToe", "rMid"]
    mcols = [i for i in frame._markers if i[0] in valid_markers]
    frame._markers = frame._markers[mcols]
    test = RunningTest(frame=frame)
    fig, dfr = test.results()
    grf_fig = file.replace(".tdf", "_grf_fig.html")
    fig.write_html(grf_fig)
    frame._forceplatforms = pd.DataFrame()
    test = RunningTest(frame=frame)
    fig, dfr = test.results()
    mrk_fig = file.replace(".tdf", "_markers_fig.html")
    fig.write_html(mrk_fig)
    print(dfr)


def test_walk():
    """test the run test"""
    print("\nTEST WALK")
    file = join(dirname(__file__), "gaitanalysis_data", "walk_test.tdf")
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
    df, fig = test.results()
    print(df)
    fig.show()


def test_gaits():
    """test the jumptests module"""
    test_run()
    test_walk()


if __name__ == "__main__":
    test_gaits()
