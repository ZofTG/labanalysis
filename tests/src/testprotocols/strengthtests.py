"""test the strength tests"""

#! IMPORTS


import sys
from os.path import dirname, join
from os import remove

import labio

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from src.labanalysis import *

__all__ = ["test_strength"]


#! FUNCTION


def test_isokinetic_1rm():
    """test the isokinetic 1RM test"""
    print("\nTEST ISOKINETIC 1RM")
    files = get_files(join(dirname(__file__), "isokinetic_1rm_data"), ".txt")
    tests = []
    for file in files:
        side = "Right" if file.split("_")[-1].split(".")[0] == "dx" else "Left"
        prod = labio.LegPressREV.from_file(file)
        tests += [Isokinetic1RMTest(prod, side)]
    battery = Isokinetic1RMTestBattery(*tests)
    battery.summary_plots.show()
    print("TEST ISOKINETIC 1RM COMPLETED")


def test_strength():
    """test the strength module"""
    print("\nSTRENGTH TESTS")
    test_isokinetic_1rm()


if __name__ == "__main__":
    test_strength()
