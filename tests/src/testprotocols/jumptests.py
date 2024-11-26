"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname, join

import numpy as np
import pandas as pd

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
    test_file = join(path, "") + ".squatjumptest"
    sj_test.save(test_file)
    sj_test = SquatJumpTest.load(test_file)
    sj_res_fig, sj_res_df = sj_test.results()
    sj_sum_fig, sj_sum_df = sj_test.summary()

    print("\tREADING COUNTER MOVEMENT JUMP DATA")
    path = join(dirname(__file__), "counter_movement_jump_data")
    cmj_files = [join(path, f"counter_movement_jump_{i + 1}.tdf") for i in range(3)]
    cmj_jumps = [CounterMovementJump.from_tdf_file(file) for file in cmj_files]
    cmj_test = CounterMovementJumpTest(baseline, cmj_jumps)
    test_file = join(path, "") + ".countermovementjumptest"
    cmj_test.save(test_file)
    cmj_test = CounterMovementJumpTest.load(test_file)
    cmj_res_fig, cmj_res_df = cmj_test.results()
    cmj_sum_fig, cmj_sum_df = cmj_test.summary()

    print("\tREADING SIDE JUMP DATA")
    path = join(dirname(__file__), "side_jump_data")
    sdj_lt_files = [join(path, f"side_jump_sx_{i + 1}.tdf") for i in range(3)]
    sdj_left = [SideJump.from_tdf_file(file, "Left") for file in sdj_lt_files]
    sdj_rt_files = [join(path, f"side_jump_dx_{i + 1}.tdf") for i in range(3)]
    sdj_right = [SideJump.from_tdf_file(file, "Right") for file in sdj_rt_files]
    sdj_test = SideJumpTest(baseline, sdj_left, sdj_right)
    test_file = join(path, "") + ".sidejumptest"
    sdj_test.save(test_file)
    sdj_test = SideJumpTest.load(test_file)
    sdj_res_fig, sdj_res_df = sdj_test.results()
    sdj_sum_fig, sdj_sum_df = sdj_test.summary()

    print("\tREADING SINGLE LEG JUMP DATA")
    path = join(dirname(__file__), "single_leg_jump_data")
    slj_lt_files = [join(path, f"single_leg_jump_sx_{i + 1}.tdf") for i in range(3)]
    slj_left = [SingleLegJump.from_tdf_file(file, "Left") for file in slj_lt_files]
    slj_rt_files = [join(path, f"single_leg_jump_dx_{i + 1}.tdf") for i in range(3)]
    slj_right = [SingleLegJump.from_tdf_file(file, "Right") for file in slj_rt_files]
    slj_test = SingleLegJumpTest(baseline, slj_left, slj_right)
    test_file = join(path, "") + ".singlelegjumptest"
    slj_test.save(test_file)
    slj_test = SingleLegJumpTest.load(test_file)
    slj_res_fig, slj_res_df = slj_test.results()
    slj_sum_fig, slj_sum_df = slj_test.summary()

    print("\tGENERATING THE TEST BATTERY")
    jumps_battery = JumpTestBattery(sj_test, cmj_test, sdj_test, slj_test)
    battery_file = join(dirname(__file__), "") + ".jumptestbattery"
    jumps_battery.save(battery_file)
    jumps_battery = JumpTestBattery.load(battery_file)
    norm_data = [
        ["SquatJumpTest", "Elevation", "Good", 40, np.inf, "green"],
        ["SquatJumpTest", "Elevation", "Normal", 20, 40, "yellow"],
        ["SquatJumpTest", "Elevation", "Poor", -np.inf, 20, "red"],
        ["CounterMovementJumpTest", "Elevation", "Good", 50, np.inf, "green"],
        ["CounterMovementJumpTest", "Elevation", "Normal", 25, 50, "yellow"],
        ["CounterMovementJumpTest", "Elevation", "Poor", -np.inf, 25, "red"],
    ]
    norm_cols = ["Test", "Parameter", "Rank", "Lower", "Upper", "Color"]
    normative_data = pd.DataFrame(data=norm_data, columns=norm_cols)
    figures, normative_data = jumps_battery.summary(normative_data)
    for fig in figures.values():
        fig.show()

    print("\nJUMP TESTS COMPLETED")


#! MAIN

if __name__ == "__main__":
    test_jumps()
