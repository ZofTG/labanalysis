"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname, join

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from labanalysis import *

__all__ = ["test_jumps"]


#! FUNCTION


def test_squatjump():
    """test the squatjump test"""
    print("\nTEST SQUAT JUMP")
    path = join(dirname(__file__), "squat_jump_data")
    jumps = [
        SquatJump.from_tdf_file(join(path, f"squat_jump_{i + 1}.tdf")) for i in range(3)
    ]
    baseline = StaticUprightStance.from_tdf_file(join(path, "baseline.tdf"))
    test = SquatJumpTest(baseline, *jumps)
    print(test.summary_table)
    fig = test.summary_plot
    fig.show()


def test_countermovementjump():
    """test the counter movement test"""
    print("\nTEST COUNTERMOVEMENT JUMP")
    path = join(dirname(__file__), "counter_movement_jump_data")
    jumps = [
        CounterMovementJump.from_tdf_file(
            join(path, f"counter_movement_jump_{i + 1}.tdf")
        )
        for i in range(3)
    ]
    baseline = StaticUprightStance.from_tdf_file(join(path, "baseline.tdf"))
    test = CounterMovementJumpTest(baseline, *jumps)
    print(test.summary_table)
    fig = test.summary_plot
    fig.show()


def test_jumps():
    """test the jumptests module"""
    test_countermovementjump()
    test_squatjump()


if __name__ == "__main__":
    test_jumps()
