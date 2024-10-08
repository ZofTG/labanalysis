"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname, join

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from labanalysis import *

__all__ = ["test_jumps"]


#! FUNCTION


def test_squatjump():
    """test the squat jump test"""
    print("\nTEST SQUAT JUMP")
    path = join(dirname(__file__), "squat_jump_data")
    files = [join(path, f"squat_jump_{i + 1}.tdf") for i in range(3)]
    jumps = [SquatJump.from_tdf_file(file) for file in files]
    baseline = UprightStance.from_tdf_file(join(path, "baseline.tdf"))
    SquatJumpTest(baseline, jumps).summary_plot.show()


def test_sidejump():
    """test the single leg jump test"""
    print("\nTEST SIDE JUMP")
    path = join(dirname(__file__), "side_jump_data")
    left_files = [join(path, f"side_jump_sx_{i + 1}.tdf") for i in range(3)]
    left_jumps = [SideJump.from_tdf_file(file, "Left") for file in left_files]
    right_files = [join(path, f"side_jump_dx_{i + 1}.tdf") for i in range(3)]
    right_jumps = [SideJump.from_tdf_file(file, "Right") for file in right_files]
    baseline = UprightStance.from_tdf_file(join(path, "baseline.tdf"))
    SideJumpTest(baseline, left_jumps, right_jumps).summary_plot.show()


def test_countermovementjump():
    """test the counter movement test"""
    print("\nTEST COUNTERMOVEMENT JUMP")
    path = join(dirname(__file__), "counter_movement_jump_data")
    files = [join(path, f"counter_movement_jump_{i + 1}.tdf") for i in range(3)]
    jumps = [CounterMovementJump.from_tdf_file(file) for file in files]
    baseline = UprightStance.from_tdf_file(join(path, "baseline.tdf"))
    CounterMovementJumpTest(baseline, jumps).summary_plot.show()


def test_singlelegjump():
    """test the single leg jump test"""
    print("\nTEST SINGLE LEG JUMP")
    path = join(dirname(__file__), "single_leg_jump_data")
    left_files = [join(path, f"single_leg_jump_sx_{i + 1}.tdf") for i in range(3)]
    left_jumps = [SingleLegJump.from_tdf_file(file, "Left") for file in left_files]
    right_files = [join(path, f"single_leg_jump_dx_{i + 1}.tdf") for i in range(3)]
    right_jumps = [SingleLegJump.from_tdf_file(file, "Right") for file in right_files]
    baseline = UprightStance.from_tdf_file(join(path, "baseline.tdf"))
    SingleLegJumpTest(baseline, left_jumps, right_jumps).summary_plot.show()


def test_jumps():
    """test the jumptests module"""
    print("\nJUMP TESTS STARTED")
    test_singlelegjump()
    test_countermovementjump()
    test_sidejump()
    test_squatjump()
    print("\nJUMP TESTS COMPLETED")


if __name__ == "__main__":
    test_jumps()
