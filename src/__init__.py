"""
labanalysis package

A library containing helpful classes and functions to speed up the lab
data processing.

Libraries
---------
constants
    a library containing useful constant values.

regression
    a library wrapping scikit-learn regression estimators.

gaitanalysis
    a library for gait analysis.

equations
    a library containing predicting equations for VO2 and 1RM

Modules
-------
signalprocessing
    a set of functions dedicated to the processing and analysis of 1D signals

utils
    module containing several utilities that can be used for multiple purposes

plotting
    a set of functions for standard plots creation.
"""

from .constants import *
from .equations import *
from .frames import *
from .io import *
from .messages import *
from .plotting import *
from .regression import *
from .signalprocessing import *
from .testprotocols import *
from .utils import *
