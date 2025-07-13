"""
point3d module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pint

from .frames import Signal3D

ureg = pint.UnitRegistry()


__all__ = ["Point3D"]


class Point3D(Signal3D):
    """
    A 3D point time series, automatically converted to meters (m).
    """

    def __init__(
        self,
        data: np.ndarray,
        index: list[float],
        unit: str | pint.Quantity = "m",
        columns: list[str] = ["X", "Y", "Z"],
    ):
        """
        Initialize a Point3D.

        Parameters
        ----------
        data : array-like
            2D data array with three columns.
        index : list of float
            Time values.
        unit : str or pint.Quantity, optional
            Unit of measurement for the data, by default "m".
        columns : list, optional
            Column labels, must be 'X', 'Y', 'Z', by default ["X", "Y", "Z"].

        Raises
        ------
        ValueError
            If units are not valid or not unique.
        """
        super().__init__(data, index, unit, columns)

        # check the unit
        # check the unit and convert to uV if required
        if not self._unit.check("[length]"):
            raise ValueError("unit must represent length.")
        meters = pint.Quantity("m")
        magnitude = self._unit.to(meters).magnitude
        self *= magnitude
        self._unit = meters

    def copy(self):
        """
        Return a deep copy of the Point3D.

        Returns
        -------
        Point3D
            A new Point3D object with the same data, index, columns, and unit.
        """
        return Point3D(
            data=self._data.copy(),
            index=self.index,
            columns=self.columns,
            unit=self.unit,
        )
