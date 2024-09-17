"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname, join

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

import matplotlib.pyplot as plt
import matplotlib
from labanalysis import *

matplotlib.use("TkAgg", force=True)

__all__ = ["test_jumps"]


#! FUNCTION


def test_squatjump():
    """test the squatjump test"""
    path = join(dirname(__file__), "squat_jump_data")
    jumps = [
        SquatJump.from_tdf_file(join(path, f"squat_jump_{i + 1}.tdf")) for i in range(3)
    ]
    baseline = StaticUprightStance.from_tdf_file(join(path, "baseline.tdf"))
    test = SquatJumpTest(baseline, *jumps)
    print(test.summary_table)
    fig = test.summary_plot
    plt.show()


def test_jumps():
    """test the jumptests module"""
    test_squatjump()


if __name__ == "__main__":
    test_jumps()
