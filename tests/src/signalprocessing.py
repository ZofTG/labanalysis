"""signal processing testing module"""

#! IMPORTS


from os.path import dirname
import sys

import numpy as np

sys.path += [dirname(dirname(dirname(__file__)))]

from src import *

__all__ = ["test_signalprocessing"]


#! FUNCTION


def test_signalprocessing():
    """test the regression module"""

    # thresholded filter
    samples = 1001
    x = np.linspace(0, 1, samples)
    freq = 2
    y = np.sin(x * freq * 2 * np.pi)
    noise = np.random.randn(samples) * 2
    filtered_median = thresholding_filt(y + noise, factor=2, robust=True, order=201)
    filtered_mean = thresholding_filt(y + noise, factor=2, robust=False, order=201)

    # fillna
    x = np.random.randn(100, 10)
    value = float(np.quantile(x.flatten(), 0.05))
    x[x <= value] = value
    y = np.copy(x)
    y[y == value] = np.nan
    assert np.all(x == fillna(y, value=value)), "fillna by value not working"
    filled_cs_ok = np.isnan(fillna(y)).sum().sum() == 0
    assert filled_cs_ok, "fillna by cubic spline not working"
    filled_lr_ok = np.isnan(fillna(y, n_regressors=4)).sum().sum() == 0
    assert filled_lr_ok, "fillna by linear regression not working"
    z = np.random.randn(100, 1).flatten()
    k = np.copy(z)
    k[np.random.permutation(np.arange(100))[:20]] = np.nan
    try:
        filled = fillna(k, value=value)
    except Exception as exc:
        raise ValueError from exc
    filled_cs_ok = np.isnan(fillna(z)).sum().sum() == 0
    assert filled_cs_ok, "fillna by cubic spline not working"


if __name__ == "__main__":
    test_signalprocessing()
