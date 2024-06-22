"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname, join
from typing import Any
import numpy as np

sys.path += [join(dirname(dirname(dirname(__file__))), "src")]

from labanalysis import *

__all__ = ["test_plotting"]


#! FUNCTION


def test_plotting():
    """test the plotting module"""
    xarr = np.random.random(100)
    yarr = np.random.random(100)
    plot_comparisons_plotly(xarr, yarr, "XLAB", "YLAB", parametric=True).show()


if __name__ == "__main__":
    test_plotting()
