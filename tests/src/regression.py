"""regression library testing module"""

#! IMPORTS


import sys
from os.path import dirname

import numpy as np
import pandas as pd

sys.path += [dirname(dirname(dirname(__file__)))]

from src.labanalysis import *

__all__ = ["test_regression"]


#! FUNCTION


def _add_noise(
    arr: np.ndarray,
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


def test_ols():
    """test the regression module"""
    x = np.linspace(1, 100, 101)

    # power regression
    print("\nTESTING POWER REGRESSION")
    y = abs(_add_noise(2 * x**0.5, 0.1))
    model = PowerRegression().fit(pd.DataFrame(x), pd.Series(y))
    betas = model.betas
    z = model.predict(x).values.flatten()
    rmse = np.nanmean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # multiline regression
    print("\nTESTING MULTISEGMENT REGRESSION")
    y = _add_noise(0.2 * x**0.5, 0.2)
    model = MultiSegmentRegression(degree=1, n_lines=2).fit(x, y)
    betas = model.betas
    z = model.predict(x).values.flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # Log regression
    print("\nTESTING LOG REGRESSION")
    y = _add_noise(0.5 + np.log(x + 1) * 0.2, 0.1)
    model = PolynomialRegression(degree=1, transform=np.log).fit(x + 1, y)
    betas = model.betas
    z = model.predict(x + 1).values.flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # polynomial regression
    print("\nTESTING POLYNOMIAL REGRESSION")
    y = _add_noise(0.5 + 0.2 * x + 0.4 * x**2, 0.1)
    model = PolynomialRegression(degree=2).fit(x, y)
    betas = model.betas
    z = model.predict(x).values.flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Output betas: {betas}\nRMSE: {rmse:0.3f}\n")


def test_geometry():
    """test the geometrical objects"""

    # line2d
    coefs = pd.Series({"A": 1, "B": 2, "C": 3})
    x = np.linspace(0, 100, 101)
    y = (-coefs.C - coefs.A * x) / coefs.B
    model = Line2D(has_intercept=True).fit(x=x, y=y)
    betas = model.betas
    print("Line2D")
    print(f"True coefs: {coefs.to_dict()}")
    print(f"Predicted coefs: {betas}")

    # line3d
    coefs = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4})
    x = np.linspace(0, 100, 101)
    y = np.linspace(0, 200, 101)
    z = (-coefs.A * x - coefs.B * y - coefs.D) / coefs.C
    model = Line3D(has_intercept=True).fit(x=x, y=y, z=z)
    betas = model.betas
    print("Line3D")
    print(f"True coefs: {coefs.to_dict()}")
    print(f"Predicted coefs: {betas}")


def test_regression():
    """test the regression library"""
    test_ols()
    test_geometry()


if __name__ == "__main__":
    test_regression()
