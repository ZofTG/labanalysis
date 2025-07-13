"""
signal1d module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pint

from .frames import Timeseries

ureg = pint.UnitRegistry()


__all__ = ["Signal1D"]


class Signal1D(Timeseries):
    """
    A 1D signal (single column) time series.
    """

    def __init__(
        self,
        data: np.ndarray,
        index: list[float],
        unit: str | pint.Quantity,
        column: str = "X",
    ):
        """
        Initialize a Signal1D.

        Parameters
        ----------
        data : array-like
            2D data array with one column.
        index : list of float
            Time values.
        unit : str or pint.Quantity
            Unit of measurement for the data.
        column : str, optional
            Single column label, by default "X".

        Raises
        ------
        ValueError
            If data does not have exactly one column.
        """
        data_array = np.asarray(data, float)
        if data_array.ndim == 1:
            data_array = np.atleast_2d(data_array).T
        if data_array.ndim != 2 or data_array.shape[1] != 1:
            raise ValueError("Signal1D must have exactly one column")
        if not isinstance(unit, (str, pint.Quantity)):
            raise ValueError("unit must be a str or a pint.Quantity")
        if isinstance(unit, str):
            unit = ureg(unit)
        super().__init__(data_array, index, [column], unit)

    def copy(self):
        """
        Return a deep copy of the Signal1D.

        Returns
        -------
        Signal1D
            A new Signal1D object with the same data, index, unit, and column.
        """
        return Signal1D(
            data=self._data.copy(),
            index=self.index,
            unit=self.unit,
        )
