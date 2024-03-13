"""
labanalysis package

A library containing helpful classes and functions to speed up the lab
data processing.

Libraries
---------
regression
    a library wrapping scikit-learn regression estimators.

Modules
-------
signalprocessing
    a set of functions dedicated to the processing and analysis of 1D signals

utils
    module containing several utilities that can be used for multiple purposes
"""

from .signalprocessing import *
from .utils import *
from .regression import *
