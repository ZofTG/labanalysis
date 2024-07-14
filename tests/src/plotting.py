"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname, join
import numpy as np

sys.path += [join(dirname(dirname(dirname(__file__))), "src")]

from labanalysis import *

__all__ = ["test_plotting"]


#! FUNCTION


def test_plotting():
    """test the plotting module"""
    xarr = np.random.random(100)
    yarr = np.random.random(100)
    clrs = np.array(["M" if i % 2 == 0 else "F" for i in np.arange(len(xarr))])
    fig = plot_comparisons_plotly(xarr, yarr, clrs, "XLAB", "YLAB", parametric=True)
    fig.show()


if __name__ == "__main__":
    test_plotting()
