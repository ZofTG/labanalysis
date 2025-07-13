"""
signal3d module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pint

from .frames import Timeseries

ureg = pint.UnitRegistry()


__all__ = ["Signal3D"]


class Signal3D(Timeseries):
    """
    A 3D signal (three columns: X, Y, Z) time series.
    """

    def __init__(
        self,
        data: np.ndarray,
        index: list[float],
        unit: pint.Quantity | str,
        columns: list[str] = ["X", "Y", "Z"],
    ):
        """
        Initialize a Signal3D.

        Parameters
        ----------
        data : array-like
            2D data array with three columns.
        index : list of float
            Time values.
        unit : str or pint.Quantity
            Unit of measurement for the data.
        columns : list, optional
            Column labels, must be 'X', 'Y', 'Z', by default ["X", "Y", "Z"].

        Raises
        ------
        ValueError
            If columns are not exactly 'X', 'Y', 'Z'.
        """
        super().__init__(data, index, columns, unit)

        # check dimensions
        if data.shape[1] != 3:
            raise ValueError("Signal3D must have exactly 3 columns.")

    def change_reference_frame(self, new_x, new_y, new_z, new_origin):
        """
        Rotate and translate each sample using the new reference frame defined by
        orthonormal versors new_x, new_y, new_z and origin new_origin.

        Parameters
        ----------
        new_x, new_y, new_z : array-like
            Orthonormal basis vectors.
        new_origin : array-like
            New origin.

        Returns
        -------
        Signal3D
            Transformed signal.

        Raises
        ------
        ValueError
            If input vectors are not valid.
        """
        R = np.array([new_x, new_y, new_z]).T  # Rotation matrix (3x3)
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3 from three 3D versors")
        if np.linalg.norm(np.dot(R.T, R) - np.eye(3)) > 1e-6:
            raise ValueError("Provided versors must be orthonormal")
        origin = np.array(new_origin)
        if origin.shape != (3,):
            raise ValueError("Origin must be a 3-element vector")

        transformed_data = (self._data - origin) @ R
        return self.__class__(
            data=transformed_data,
            index=self.index,
            columns=self.columns,
            unit=self.unit,
        )

    def copy(self):
        """
        Return a deep copy of the Signal3D.

        Returns
        -------
        Signal3D
            A new Signal3D object with the same data, index, columns, and unit.
        """
        return Signal3D(
            data=self._data.copy(),
            index=self.index,
            columns=self.columns,
            unit=self.unit,
        )
