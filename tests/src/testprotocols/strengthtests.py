"""test the strength tests"""

#! IMPORTS


import sys
from os.path import dirname, join

import labio

sys.path += [dirname(dirname(dirname(dirname(__file__))))]

from labanalysis import *

__all__ = ["test_strength"]


#! FUNCTION


def test_isokinetic_1rm():
    """test the isokinetic 1RM test"""
    print("\nTEST ISOKINETIC 1RM")
    files = get_files(join(dirname(__file__), "isokinetic_1rm_data"), ".txt")
    for file in files:
        test = Isokinetic1RMTest.from_biostrength_file(file, labio.LegPressREV)  # type: ignore
        fig = test.summary_plot
        fig.update_layout(title=file)
        fig.show()
    print("TEST ISOKINETIC 1RM COMPLETED")


def test_strength():
    """test the strength module"""
    print("\nSTRENGTH TESTS")
    test_isokinetic_1rm()


if __name__ == "__main__":
    test_strength()
