"""test the postural tests"""

#! IMPORTS


import sys
from os.path import dirname, join

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from labanalysis import *

__all__ = ["test_posture"]


#! FUNCTION


def test_plank():
    """test the squatjump test"""
    print("\nTEST PLANK")
    file = join(dirname(__file__), "plank_data", "plank.tdf")
    PlankTest.from_tdf_file(file).summary_plot.show()


def test_posture():
    """test the jumptests module"""
    print("\nPOSTURAL TESTS")
    test_plank()


if __name__ == "__main__":
    test_posture()
