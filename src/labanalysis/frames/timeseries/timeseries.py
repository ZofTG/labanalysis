"""
timeseries module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pandas as pd
import pint

from .frames import LabeledArray

ureg = pint.UnitRegistry()


__all__ = ["Timeseries"]


class Timeseries(LabeledArray):
    """
    A labeled array with float-valued time index, representing a time series.
    """

    _unit: pint.Quantity

    def __init__(
        self,
        data: np.ndarray,
        index: list[float],
        columns: list[str],
        unit: str | pint.Quantity,
    ):
        """
        Initialize a Timeseries.

        Parameters
        ----------
        data : array-like
            2D data array.
        index : list of float
            Time values.
        columns : list
            Column labels.
        unit : str or pint.Quantity
            Unit of measurement for the data.

        Raises
        ------
        TypeError
            If any index value is not a float.
        ValueError
            If the unit is not a string or a pint.Quantity.
        """
        if not all(isinstance(i, float) for i in index):
            raise TypeError("All index values must be floats for Timeseries")
        super().__init__(data, index, columns)

        # check the unit of measurement
        msg = "unit must be a string representing a conventional unit "
        msg += "of measurement in the SI sytem or a pint.Quantity"
        if not isinstance(unit, (str, pint.Quantity)):
            raise ValueError(msg)
        if isinstance(unit, str):
            try:
                unit = ureg(unit)
            except Exception as exc:
                raise ValueError(msg) from exc
        self._unit = unit

    @property
    def unit(self):
        """
        Get the unit of measurement.

        Returns
        -------
        str
            The unit of measurement.
        """
        return f"{self._unit.units:~}"

    def reset_time(self, inplace=False):
        """
        Reset the time index to start at zero.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify in place. If False, return a new Timeseries.

        Returns
        -------
        Timeseries or None
            If inplace is False, returns a new Timeseries with reset time.
            If inplace is True, returns None.
        """
        min_time = min(self.index)
        new_index = [i - min_time for i in self.index]
        if inplace:
            self.index = new_index
            self._row_map = {label: i for i, label in enumerate(self.index)}
            return self
        else:
            return Timeseries(
                data=self._data.copy(),
                index=new_index,
                columns=self.columns,
                unit=self.unit,
            )

    def __getitem__(self, key, indices=False):
        """
        Get item(s) from the timeseries using time-based slicing or boolean mask.

        Parameters
        ----------
        key : slice, np.ndarray, or other
            Slicing or mask.
        indices : bool, optional
            If True, interpret key as indices.

        Returns
        -------
        Timeseries or np.ndarray
            The selected data as a Timeseries object or a NumPy array.
        """
        # Time-based slicing with float range
        if (
            isinstance(key, slice)
            and isinstance(key.start, float)
            and isinstance(key.stop, float)
        ):
            mask = [(k >= key.start and k < key.stop) for k in self.index]
            return self.__class__(
                data=self._data[mask],
                index=[k for k, m in zip(self.index, mask) if m],
                columns=self.columns,
                unit=self.unit,
            )
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            return self.__class__(
                data=self._data[key],
                index=[k for k, m in zip(self.index, key) if m],
                columns=self.columns,
                unit=self.unit,
            )

        return super().__getitem__(key, indices=indices)

    def mean(self, axis=None, **kwargs):
        """
        Compute the mean along the specified axis, mimicking numpy's mean.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which the mean is computed. Default is None (mean of all elements).
            0 = mean over rows (returns 1D array of columns)
            1 = mean over columns (returns 1D array of rows)
        **kwargs : dict
            Additional keyword arguments passed to numpy.mean.

        Returns
        -------
        float or np.ndarray
            Mean value(s) as per numpy.mean.
        """
        return np.mean(self._data, axis=axis, **kwargs)

    def copy(self):
        """
        Return a deep copy of the Timeseries.

        Returns
        -------
        Timeseries
            A new Timeseries object with the same data, index, columns, and unit.
        """
        return Timeseries(
            data=self._data.copy(),
            index=self.index,
            columns=self.columns,
            unit=self.unit,
        )

    def to_dataframe(self):
        """
        Convert to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame representation.
        """
        columns = []
        for i in self.columns:
            if not isinstance(i, tuple):
                columns += [(i, self.unit)]
            else:
                columns += [(*i, self.unit)]
        columns = pd.MultiIndex.from_tuples(columns)
        return pd.DataFrame(self._data, index=self.index, columns=columns)

    @classmethod
    def from_dataframe(cls, df):
        """
        Create a LabeledArray from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        LabeledArray
            A Timeseries object created from the DataFrame.
        """

        # check the index
        try:
            index = df.index.to_numpy().astype(float)
        except Exception as exc:
            raise ValueError("'df' index must be castable to float") from exc

        # check columns
        cols = []
        units = []
        for col in df.columns.tolist():
            if not isinstance(col, tuple) or not len(col) == 2:
                raise ValueError("'df' columns must be tuples with 2 elements")
            axis, unit = col
            cols += [axis]
            units += [unit]

        # check unit of measurement
        units = np.unique(units)
        if len(units) > 1:
            raise ValueError("Timeseries must have one single unit of measurement.")
        unit = units[0]

        # check values
        try:
            vals = df.values.astype(float)
        except Exception as exc:
            raise ValueError("Timeseries values must be castable to float.") from exc
        return cls(data=vals, index=index, columns=cols, unit=unit)
