"""rslib.regression testing module"""

#! IMPORTS


import sys
from os.path import dirname
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path += [dirname(dirname(dirname(__file__)))]

from labanalysis import *

__all__ = ["test_regression"]


#! FUNCTION


def _add_noise(
    arr: np.ndarray[Any, np.dtype[np.float64 | np.int_]],
    noise: float,
):
    """
    add noise to the array

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64 | np.int_]]
        the input array

    noise: float
        the noise level

    Return
    ------
    out: np.ndarray[Any, np.dtype[np.float64 | np.int_]]
        the array with added input.
    """
    return arr + np.random.randn(*arr.shape) * np.std(arr) * noise


def test_regression():
    """test the regression module"""
    x = np.linspace(0, 100, 101)

    # multiline regression
    print("\nTESTING MULTISEGMENT REGRESSION")
    y = _add_noise(0.2 * x**0.5, 0.2)
    model = MultiSegmentRegression(degree=1, n_lines=2).fit(x, y)
    betas = model.betas
    z = model.predict(x).values.flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")
    plt.close()
    plt.plot(x, y, label="RAW")
    plt.plot(x, z, "r--", label="FITTED")
    plt.legend()
    plt.title("MULTISEGMENT REGRESSION")
    plt.show()

    # Log regression
    print("\nTESTING LOG REGRESSION")
    y = _add_noise(0.5 + np.log(x + 1) * 0.2, 0.1)
    model = PolynomialRegression(degree=1, transform=np.log).fit(x + 1, y)
    betas = model.betas
    z = model.predict(x + 1).values.flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")
    plt.close()
    plt.plot(x + 1, y, label="RAW")
    plt.plot(x + 1, z, "r--", label="FITTED")
    plt.legend()
    plt.title("LOG REGRESSION")
    plt.show()

    # polynomial regression
    print("\nTESTING POLYNOMIAL REGRESSION")
    y = _add_noise(0.5 + 0.2 * x + 0.4 * x**2, 0.1)
    model = PolynomialRegression(degree=2).fit(x, y)
    betas = model.betas
    z = model.predict(x).values.flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")
    plt.close()
    plt.plot(x, y, label="RAW")
    plt.plot(x, z, "r--", label="FITTED")
    plt.legend()
    plt.title("POLYNOMIAL REGRESSION")
    plt.show()

    # power regression
    print("\nTESTING POWER REGRESSION")
    y = abs(_add_noise(2 * x**0.5, 0.1))
    model = PowerRegression().fit(x, y)
    betas = model.betas
    z = model.predict(x).values.flatten()
    rmse = np.nanmean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")
    plt.close()
    plt.plot(x, y, label="RAW")
    plt.plot(x, z, "r--", label="FITTED")
    plt.legend()
    plt.title("POWER REGRESSION")
    plt.show()


if __name__ == "__main__":
    test_regression()
