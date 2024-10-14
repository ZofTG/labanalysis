"""test the plotting module"""

#! IMPORTS


import sys
from os.path import dirname
import numpy as np

sys.path += [dirname(dirname(dirname(__file__)))]

from src.labanalysis import *

__all__ = ["test_plotting"]


#! FUNCTION


def test_plotting():
    """test the plotting module"""
    xarr = np.random.random(100)
    yarr = np.random.random(100)
    yarr = (yarr - np.mean(yarr)) / np.std(yarr)
    beta0 = 0
    beta1 = -0.5
    std = 0.3
    yarr = (xarr * beta1 + beta0) + yarr * std
    clrs = np.array(["M" if i % 2 == 0 else "F" for i in np.arange(len(xarr))])
    fig = plot_comparisons_plotly(xarr, yarr, clrs, "XLAB", "YLAB", parametric=True)
    fig.show()


if __name__ == "__main__":
    test_plotting()
