"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname, join

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from src.labanalysis import *

__all__ = ["test_jumps"]


#! FUNCTION


def test_jumps():
    """test the jumptests module"""
    print("\nJUMP TESTS STARTED")

    print("\tREADING UPRIGHT STANCE DATA")
    path = join(dirname(__file__), "squat_jump_data")
    baseline = UprightStance.from_tdf_file(join(path, "baseline.tdf"))

    print("\tREADING SQUAT JUMP DATA")
    sj_files = [join(path, f"squat_jump_{i + 1}.tdf") for i in range(3)]
    sj_jumps = [SquatJump.from_tdf_file(file) for file in sj_files]
    sj_test = SquatJumpTest(baseline, sj_jumps)

    print("\tREADING COUNTER MOVEMENT JUMP DATA")
    path = join(dirname(__file__), "counter_movement_jump_data")
    cmj_files = [join(path, f"counter_movement_jump_{i + 1}.tdf") for i in range(3)]
    cmj_jumps = [CounterMovementJump.from_tdf_file(file) for file in cmj_files]
    cmj_test = CounterMovementJumpTest(baseline, cmj_jumps)

    print("\tREADING SIDE JUMP DATA")
    path = join(dirname(__file__), "side_jump_data")
    sdj_lt_files = [join(path, f"side_jump_sx_{i + 1}.tdf") for i in range(3)]
    sdj_left = [SideJump.from_tdf_file(file, "Left") for file in sdj_lt_files]
    sdj_rt_files = [join(path, f"side_jump_dx_{i + 1}.tdf") for i in range(3)]
    sdj_right = [SideJump.from_tdf_file(file, "Right") for file in sdj_rt_files]
    sdj_test = SideJumpTest(baseline, sdj_left, sdj_right)

    print("\tREADING SINGLE LEG JUMP DATA")
    path = join(dirname(__file__), "single_leg_jump_data")
    slj_lt_files = [join(path, f"single_leg_jump_sx_{i + 1}.tdf") for i in range(3)]
    slj_left = [SingleLegJump.from_tdf_file(file, "Left") for file in slj_lt_files]
    slj_rt_files = [join(path, f"single_leg_jump_dx_{i + 1}.tdf") for i in range(3)]
    slj_right = [SingleLegJump.from_tdf_file(file, "Right") for file in slj_rt_files]
    slj_test = SingleLegJumpTest(baseline, slj_left, slj_right)

    print("\tGENERATING THE TEST BATTERY")
    jumps_battery = TestBattery(sj_test, cmj_test, sdj_test, slj_test)
    for name, fig in jumps_battery.results_plots.items():
        fig.update_layout(title=name + " RESULTS")
        fig.show()
    for name, fig in jumps_battery.summary_plots.items():
        fig.update_layout(title=name + " SUMMARY")
        fig.show()

    print("\nJUMP TESTS COMPLETED")


#! MAIN

if __name__ == "__main__":
    test_jumps()
