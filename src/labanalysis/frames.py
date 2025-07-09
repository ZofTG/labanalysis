"""
frames module containing useful classes for kinetic, kinematic and EMG data
analysis
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import pickle
import re
from copy import deepcopy
from os.path import exists
from socket import SocketType
from typing import Optional, Union
from warnings import warn

import numpy as np
import pandas as pd

from . import messages, signalprocessing, io

__all__ = [
    "TimeSeries",
    "Signal1D",
    "EMGSignal",
    "Signal3D",
    "Point3D",
    "ForcePlatform",
    "StateFrame",
]

#! CLASSES


class IndexSearch(pd.Index):
    """
    An enhanced version of pandas.Index that supports advanced search operations.
    Supports exact match and regular expression (regex) search across index
    values.

    Methods
    -------
    search(key)
        Search for values in the index using exact match or regex.
    """

    def search(self, key):
        """
        Search for values in the index using exact match or regex.

        Parameters
        ----------
        key : int, str, re.Pattern
            The value or regex pattern to search for.

        Returns
        -------
        list of int
            List of positions in the index that match the search criteria.

        Raises
        ------
        KeyError
            If no match is found.
        """
        if isinstance(key, (int, slice)):
            return key
        if isinstance(key, str) or isinstance(key, re.Pattern):
            pattern = re.compile(key) if isinstance(key, str) else key
            matches = [i for i, val in enumerate(self) if pattern.search(str(val))]
        else:
            matches = [i for i, val in enumerate(self) if val == key]
        if not matches:
            raise KeyError(f"{key} not found in Index.")
        return matches


class MultiIndexSearch(pd.MultiIndex):
    """
    An enhanced version of pandas.MultiIndex that supports advanced search
    operations.
    Allows searching across all levels of the MultiIndex using exact match or
    regex.

    Methods
    -------
    search(key)
        Search for values across all levels using exact match or regex.
    """

    def search(self, key):
        """
        Search for values across all levels using exact match or regex.

        Parameters
        ----------
        key : int, str, re.Pattern, tuple
            The value or regex pattern to search for.

        Returns
        -------
        list of int
            List of positions in the MultiIndex that match the search criteria.

        Raises
        ------
        KeyError
            If no match is found.
        """
        if isinstance(key, (int, slice, tuple)):
            return key

        if isinstance(key, str) or isinstance(key, re.Pattern):
            pattern = re.compile(key) if isinstance(key, str) else key
            matches = [
                i
                for i, tup in enumerate(self)
                if any(pattern.search(str(val)) for val in tup)
            ]
        else:
            matches = [i for i, tup in enumerate(self) if key in tup]

        if not matches:
            raise KeyError(f"{key} not found in any level of the MultiIndex.")

        return matches


class SmartSeries(pd.Series):
    """
    A pandas Series subclass that supports advanced index-based search
    operations.
    Works with SmartIndex and MultiIndexSearch to enable regex and value-based
    lookup across all levels of the index.

    Properties
    ----------
    loc : object
        Enhanced .loc accessor with search support.
    iloc : object
        Enhanced .iloc accessor with search support.

    Methods
    -------
    xs(key, level=None, drop_level=True)
        Cross-section lookup with support for search across all index levels.
    """

    @property
    def _constructor(self):
        return SmartSeries

    def __getitem__(self, key):
        if hasattr(self.index, "search"):
            try:
                key = self.index.search(key)  # type: ignore
            except KeyError:
                pass
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if hasattr(self.index, "search"):
            try:
                key = self.index.search(key)  # type: ignore
            except KeyError:
                pass
        return super().__setitem__(key, value)

    @property
    def loc(self):
        parent = self

        class _LocIndexer:
            def __getitem__(_, key):  # type: ignore
                if hasattr(parent.index, "search"):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                return pd.Series.loc.__getitem__(parent, key)  # type: ignore

            def __setitem__(_, key, value):  # type: ignore
                if hasattr(parent.index, "search"):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                pd.Series.loc.__setitem__(parent, key, value)  # type: ignore

        return _LocIndexer()

    @property
    def iloc(self):
        parent = self

        class _IlocIndexer:
            def __getitem__(_, key):  # type: ignore
                if hasattr(parent.index, "search") and not isinstance(
                    key, (int, slice, list)
                ):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                return pd.Series.iloc.__getitem__(parent, key)  # type: ignore

            def __setitem__(_, key, value):  # type: ignore
                if hasattr(parent.index, "search") and not isinstance(
                    key, (int, slice, list)
                ):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                pd.Series.iloc.__setitem__(parent, key, value)  # type: ignore

        return _IlocIndexer()

    def xs(self, key, level=None, drop_level=True):
        """
        Return cross-section from the Series with support for search.

        Parameters
        ----------
        key : object
            Label or regex pattern to match.
        level : object, optional
            Level(s) to match values on (default is None).
        drop_level : bool, default True
            If False, returns object with same levels as self.

        Returns
        -------
        Series
            Cross-section of the Series.
        """
        if (
            hasattr(self.index, "search")
            and level is None
            and not isinstance(key, dict)
        ):
            try:
                matches = self.index.search(key)  # type: ignore
                return self.iloc[matches]
            except KeyError:
                pass
        return super().xs(key, level=level, drop_level=drop_level)  # type: ignore


class SmartDataFrame(pd.DataFrame):
    """
    A pandas DataFrame subclass that supports advanced index-based search
    operations.
    Works with SmartIndex and MultiIndexSearch to enable regex and value-based
    lookup across all levels of the index.

    Properties
    ----------
    loc : object
        Enhanced .loc accessor with search support.
    iloc : object
        Enhanced .iloc accessor with search support.

    Methods
    -------
    xs(key, level=None, axis=0, drop_level=True)
        Cross-section lookup with support for search across all index levels.
    """

    @property
    def _constructor(self):
        return SmartDataFrame

    @property
    def _constructor_sliced(self):
        return SmartSeries

    def __getitem__(self, key):
        if hasattr(self.index, "search"):
            try:
                key = self.index.search(key)  # type: ignore
            except KeyError:
                pass
        result = super().__getitem__(key)
        if isinstance(result, pd.Series):
            result.__class__ = SmartSeries
        return result

    def __setitem__(self, key, value):
        if hasattr(self.index, "search"):
            try:
                key = self.index.search(key)  # type: ignore
            except KeyError:
                pass
        return super().__setitem__(key, value)

    @property
    def loc(self):
        parent = self

        class _LocIndexer:
            def __getitem__(_, key):  # type: ignore
                if hasattr(parent.index, "search"):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                result = pd.DataFrame.loc.__getitem__(parent, key)  # type: ignore
                if isinstance(result, pd.Series):
                    result.__class__ = SmartSeries
                return result

            def __setitem__(_, key, value):  # type: ignore
                if hasattr(parent.index, "search"):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                pd.DataFrame.loc.__setitem__(parent, key, value)  # type: ignore

        return _LocIndexer()

    @property
    def iloc(self):
        parent = self

        class _IlocIndexer:
            def __getitem__(_, key):  # type: ignore
                if hasattr(parent.index, "search") and not isinstance(
                    key, (int, slice, list)
                ):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                result = pd.DataFrame.iloc.__getitem__(parent, key)  # type: ignore
                if isinstance(result, pd.Series):
                    result.__class__ = SmartSeries
                return result

            def __setitem__(_, key, value):  # type: ignore
                if hasattr(parent.index, "search") and not isinstance(
                    key, (int, slice, list)
                ):
                    try:
                        key = parent.index.search(key)  # type: ignore
                    except KeyError:
                        pass
                pd.DataFrame.iloc.__setitem__(parent, key, value)  # type: ignore

        return _IlocIndexer()

    def xs(self, key, level=None, axis=0, drop_level=True):
        """
        Return cross-section from the DataFrame with support for search.

        Parameters
        ----------
        key : object
            Label or regex pattern to match.
        level : object, optional
            Level(s) to match values on (default is None).
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis to retrieve cross-section on.
        drop_level : bool, default True
            If False, returns object with same levels as self.

        Returns
        -------
        Series or DataFrame
            Cross-section of the DataFrame.
        """
        if (
            hasattr(self.index, "search")
            and level is None
            and not isinstance(key, dict)
        ):
            try:
                matches = self.index.search(key)  # type: ignore
                result = self.iloc[matches]
                if isinstance(result, pd.Series):
                    result.__class__ = SmartSeries
                return result
            except KeyError:
                pass
        result = super().xs(key, level=level, axis=axis, drop_level=drop_level)  # type: ignore
        if isinstance(result, pd.Series):
            result.__class__ = SmartSeries
        return result

    def copy(self):
        """
        Create a deep copy of the object.

        Returns
        -------
        TimeSeries
            A deep copy of the current object.
        """
        return deepcopy(self)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame):
        if isinstance(frame, pd.DataFrame) or issubclass(type(frame), pd.Dataframe):
            return cls(frame)


class TimeSeries(SmartDataFrame):
    """
    A base class for time series data with named dimensions and units.

    Parameters
    ----------
    **data: 1D numeric arrays
    time : array-like, optional
        Time values corresponding to the data. If None, a default range is used.
    unit : str, optional
        Unit of measurement.
    strip : bool, default=True
        Whether to strip leading/trailing NaNs.
    reset_index : bool, default=False
        Whether to reset the time index to start from zero.
    """

    def __init__(
        self,
        time: Optional[Union[np.ndarray, pd.Series, pd.DataFrame, list]] = None,
        unit: str = "",
        strip: bool = True,
        reset_index: bool = False,
        **data: Union[np.ndarray, list],
    ):
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Validate time
        if time is not None:
            time = np.asarray(time).astype(float).flatten()
            if len(time) != df.shape[0]:
                msg = f"Length of time ({len(time)}) does not match number "
                msg += f"of samples ({len(df)})."
                raise ValueError(msg)
        else:
            time = np.arange(len(df))

        # Validate data
        for key, arr in df.items():
            if not np.issubdtype(arr.dtype, np.number):  # type: ignore
                raise TypeError(f"Data for '{key}' must be numeric.")

        # build the object
        time = df.index.to_numpy().astype(float)
        values = df.values.astype(float)
        cols = MultiIndexSearch.from_product([df.columns.to_list(), [unit]])
        super().__init__(data=values, index=time, columns=cols)

        # transform
        if strip:
            self.strip(inplace=True)
        if reset_index:
            self.reset_index(inplace=True)

    def strip(self, inplace: bool = True):
        """
        Remove leading and trailing rows where all values are NaN.

        Parameters
        ----------
        inplace : bool, default=True
            Whether to modify the object in place.
        """
        valid = ~np.isnan(self.values).all(axis=-1)
        if not np.any(valid):
            raise ValueError("All values are NaN; cannot strip.")
        index0, index1 = np.where(valid)[0][[0, -1]]
        times = self.index.to_numpy()[np.arange(index0, index1 + 1)]
        sliced = self.loc[times]
        if inplace:
            self[:] = sliced
        else:
            return sliced

    def reset_index(self, inplace: bool = True):
        """
        Reset the time index to start from zero.

        Parameters
        ----------
        inplace : bool, default=True
            Whether to modify the object in place.
        """
        time = self.index.to_numpy()
        new_time = time - time[0]
        if inplace:
            self.index = pd.Index(new_time)
        else:
            out = self.copy()
            out.index = pd.Index(new_time)
            return out


class Signal1D(TimeSeries):
    """
    A class for representing 1D time series signals.

    Parameters
    ----------
    **data: 1D numeric arrays
    time : array-like, optional
        Time values corresponding to the signal. Required if `data` is not a
        DataFrame.
    name : str, optional
        Name of the signal.
    unit : str, optional
        Unit of the signal (e.g., "V" for volts).
    strip : bool, default=True
        Whether to strip leading/trailing NaNs.
    reset_index : bool, default=False
        Whether to reset the time index to start from zero.

    Attributes
    ----------
    unit : str
        the unit of measurement

    Raises
    ------
    ValueError
        If time and data lengths do not match.
    TypeError
        If data contains non-numeric values.
    """

    @property
    def unit(self):
        """Unit of the signal."""
        return self.columns.to_list()[0][-1]

    def __init__(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame, list],
        time: Optional[Union[np.ndarray, pd.Series, pd.DataFrame, list]] = None,
        unit: str = "",
        strip: bool = True,
        reset_index: bool = False,
    ):

        # check input data
        if isinstance(data, (np.ndarray, list, pd.Series)):
            data = np.asarray(data).astype(float)
        elif isinstance(data, pd.DataFrame):
            time = data.index.to_numpy()
            data = np.asarray(data.values).astype(float)
        else:
            raise TypeError("Unsupported data type for Signal1D.")
        if data.ndim != 1:
            raise ValueError("data must be one-dimensional.")

        # Initialize base class
        super().__init__(
            amplitude=data,
            time=time,
            unit=unit,
            strip=strip,
            reset_index=reset_index,
        )


class Signal3D(TimeSeries):
    """
    A class for representing 3D time series signals (e.g., trajectories, forces).

    Parameters
    ----------
    xarr : array-like
        X-component of the signal.
    yarr : array-like
        Y-component of the signal.
    zarr : array-like
        Z-component of the signal.
    time : array-like, optional
        Time values corresponding to the signal. If None, a default range is used.
    unit : str, optional
        Unit of the signal (e.g., "m" for meters).
    strip : bool, default=True
        Whether to strip leading/trailing NaNs.
    reset_index : bool, default=False
        Whether to reset the time index to start from zero.

    Attributes
    ----------
    unit : str
        the unit of measurement

    Raises
    ------
    ValueError
        If time and data lengths do not match.
    TypeError
        If data contains non-numeric values.
    """

    @property
    def unit(self):
        """Unit of the signal."""
        return self.columns.to_list()[0][-1]

    def __init__(
        self,
        xarr: Union[np.ndarray, list],
        yarr: Union[np.ndarray, list],
        zarr: Union[np.ndarray, list],
        time: Optional[Union[np.ndarray, list]] = None,
        unit: str = "",
        strip: bool = True,
        reset_index: bool = False,
    ):
        # Convert and validate input arrays
        x = np.asarray(xarr).astype(float).flatten()
        y = np.asarray(yarr).astype(float).flatten()
        z = np.asarray(zarr).astype(float).flatten()

        if not (len(x) == len(y) == len(z)):
            raise ValueError("xarr, yarr, and zarr must have the same length.")

        super().__init__(
            time=time,
            unit=unit,
            strip=strip,
            reset_index=reset_index,
            **{"X": x, "Y": y, "Z": z},
        )

    def _validate_reference_frame_axes(
        self,
        axis1: np.ndarray | list[float | int],
        axis2: np.ndarray | list[float | int],
        axis3: np.ndarray | list[float | int],
        tol: float = 1e-6,
    ):
        axis1 = np.asarray(axis1, dtype=float)
        axis2 = np.asarray(axis2, dtype=float)
        axis3 = np.asarray(axis3, dtype=float)

        if axis1.shape != (3,) or axis2.shape != (3,) or axis3.shape != (3,):
            raise ValueError("Each axis must be a 3D vector.")

        # Check orthogonality
        if not (
            np.abs(np.dot(axis1, axis2)) < tol
            and np.abs(np.dot(axis1, axis3)) < tol
            and np.abs(np.dot(axis2, axis3)) < tol
        ):
            raise ValueError("Axes must be mutually orthogonal.")

        # Optional: check unit length
        if not (
            np.isclose(np.linalg.norm(axis1), 1.0, atol=tol)
            and np.isclose(np.linalg.norm(axis2), 1.0, atol=tol)
            and np.isclose(np.linalg.norm(axis3), 1.0, atol=tol)
        ):
            raise ValueError("Each axis must be a unit vector.")

    def change_reference_frame(
        self,
        origin: np.ndarray | list[float | int] = [0, 0, 0],
        axis1: np.ndarray | list[float | int] = [1, 0, 0],
        axis2: np.ndarray | list[float | int] = [0, 1, 0],
        axis3: np.ndarray | list[float | int] = [0, 0, 1],
        inplace: bool = True,
    ):
        """
        Rotate and translate the signal to a new reference frame.

        Parameters
        ----------
        origin : array-like, default=[0, 0, 0]
            Origin of the new reference frame.
        axis1 : array-like, default=[1, 0, 0]
            First axis of the new reference frame.
        axis2 : array-like, default=[0, 1, 0]
            Second axis of the new reference frame.
        axis3 : array-like, default=[0, 0, 1]
            Third axis of the new reference frame.
        inplace : bool, default=True
            Whether to modify the object in place.

        Returns
        -------
        Signal3D or None
            Transformed signal if `inplace=False`, otherwise None.
        """
        if self.shape[0] == 0:
            return self.copy()
        self._validate_reference_frame_axes(axis1, axis2, axis3)
        rotated = signalprocessing.to_reference_frame(
            obj=self.values,
            origin=origin,
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
        )
        rotated = np.asarray(rotated, dtype=float)
        if inplace:
            self[:] = rotated
        else:
            out = self.copy()
            out.iloc[:, :] = rotated
            return out


class EMGSignal(Signal1D):
    """
    A class for representing EMG (electromyography) signals.

    Parameters
    ----------
    data : array-like, pd.Series, pd.DataFrame, list, or dict[str, array-like]
        The EMG signal data. If a single array-like object is provided, it is
        treated as a single-channel signal.
        If a DataFrame is provided, its index is used as the time vector.
    time : array-like, optional
        Time values corresponding to the signal. Required if `data` is not a
        DataFrame.
    name : str, optional
        Name of the muscle (e.g., "left biceps femoris").
    strip : bool, default=True
        Whether to strip leading/trailing NaNs.
    reset_index : bool, default=False
        Whether to reset the time index to start from zero.

    Attributes
    ----------
    side : str
        Side of the body ('left', 'right', or '').
    unit : pint.Quantity
        Unit of the signal (volts).
    """

    @property
    def side(self):
        """The side of the body the muscle is on ('left', 'right', or '')."""
        return self.columns[0][-2]

    def __init__(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame, list],
        time: Optional[Union[np.ndarray, pd.Series, pd.DataFrame, list]] = None,
        name: str = "",
        strip: bool = True,
        reset_index: bool = False,
    ):

        # Build the signal
        super().__init__(
            data=data,
            time=time,
            unit="V",
            strip=strip,
            reset_index=reset_index,
        )

        # Infer side from name
        splits = name.lower().split(" ")
        side_idx = next(
            (i for i, v in enumerate(splits) if v in ["left", "right"]), None
        )
        side = "" if side_idx is None else splits.pop(side_idx)  # type: ignore
        name = "_".join(splits[:2]) if splits else ""

        # Validate side
        if side not in ["left", "right", ""]:
            raise ValueError("Side must be 'left', 'right', or ''.")

        # update the columns
        self.columns = MultiIndexSearch.from_tuples([(name, side, "V")])


class Point3D(Signal3D):
    """
    A class for representing 3D spatial points over time.

    Parameters
    ----------
    xarr : array-like
        X-component of the point.
    yarr : array-like
        Y-component of the point.
    zarr : array-like
        Z-component of the point.
    time : array-like, optional
        Time values corresponding to the point. If None, a default range is used.
    strip : bool, default=True
        Whether to strip leading/trailing NaNs.
    reset_index : bool, default=False
        Whether to reset the time index to start from zero.

    Raises
    ------
    ValueError
        If input arrays have inconsistent lengths.
    TypeError
        If input arrays are not numeric.

    Attributes
    ----------
    unit : str
        Unit of the point (meters).
    """

    def __init__(
        self,
        xarr: Union[np.ndarray, list],
        yarr: Union[np.ndarray, list],
        zarr: Union[np.ndarray, list],
        time: Optional[Union[np.ndarray, list]] = None,
        strip: bool = True,
        reset_index: bool = False,
    ):
        super().__init__(
            xarr=xarr,
            yarr=yarr,
            zarr=zarr,
            time=time,
            unit="m",
            strip=strip,
            reset_index=reset_index,
        )


class ForcePlatform(SmartDataFrame):
    """
    A class representing a force platform dataset, including origin, force,
    and torque signals.

    Parameters
    ----------
    origin : Point3D
        The origin of the force platform.
    force : Signal3D
        The 3D force vector over time.
    torque : Signal3D
        The 3D torque vector over time.
    strip : bool, default=True
        Whether to strip leading/trailing NaNs.
    reset_index : bool, default=True
        Whether to reset the time index to start from zero.

    Raises
    ------
    ValueError
        If inputs are not of the correct type or have inconsistent time
        coordinates or missing units.
    """

    @property
    def origin(self):
        """Return the origin coordinates as a Point3D."""
        xarr, yarr, zarr = self["origin"].values.astype(float).T
        return Point3D(
            xarr=xarr,
            yarr=yarr,
            zarr=zarr,
            time=self.index.to_numpy(),
            strip=False,
            reset_index=False,
        )  # type: ignore

    @property
    def force(self):
        """Return the force vector as a Signal3D."""
        xarr, yarr, zarr = self["force"].values.astype(float).T
        unit = self.unit["force"]
        return Signal3D(
            xarr=xarr,
            yarr=yarr,
            zarr=zarr,
            time=self.index.to_numpy(),
            unit=unit,
            strip=False,
            reset_index=False,
        )  # type: ignore

    @property
    def torque(self):
        """Return the torque vector as a Signal3D."""
        xarr, yarr, zarr = self["torque"].values.astype(float).T
        unit = self.unit["torque"]
        return Signal3D(
            xarr=xarr,
            yarr=yarr,
            zarr=zarr,
            time=self.index.to_numpy(),
            unit=unit,
            strip=False,
            reset_index=False,
        )  # type: ignore

    def __init__(
        self,
        origin,
        force,
        torque,
        strip=True,
        reset_index=True,
    ):
        if not isinstance(origin, Point3D):
            raise ValueError("'origin' must be a Point3D instance.")
        if not isinstance(force, Signal3D):
            raise ValueError("'force' must be a Signal3D instance.")
        if not isinstance(torque, Signal3D):
            raise ValueError("'torque' must be a Signal3D instance.")

        # build the object
        out = SmartDataFrame(pd.concat([origin, force, torque], axis=1))
        origin_cols = MultiIndexSearch.from_product(
            [["origin"], ["X", "Y", "Z"], [origin.unit]]
        )
        force_cols = MultiIndexSearch.from_product(
            [["force"], ["X", "Y", "Z"], [force.unit]]
        )
        torque_cols = MultiIndexSearch.from_product(
            [["torque"], ["X", "Y", "Z"], [torque.unit]]
        )
        cols = origin_cols.tolist() + force_cols.tolist() + torque_cols.tolist()
        out.columns = cols
        super().__init__(out)

        if strip:
            self.strip(inplace=True)
        if reset_index:
            self.reset_index(inplace=True)

    def strip(self, inplace=True):
        valid = ~np.isnan(self["origin"].values.astype(float)).all(axis=-1)
        if not np.any(valid):
            raise ValueError("All origin values are NaN; cannot strip.")
        index0, index1 = np.where(valid)[0][[0, -1]]
        index = np.arange(index0, index1 + 1)
        if inplace:
            self = self.iloc[index]
        else:
            out = self.copy()
            return out.iloc[index]

    def reset_index(self, inplace=True):
        if inplace:
            self.index = self.index - self.index[0]
        else:
            out = self.copy()
            out.index = out.index - self.index[0]
            return out

    def slice(self, from_time, to_time, inplace=True):
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if inplace:
            self.loc[(self.index >= from_time) & (self.index <= to_time)]
        else:
            out = self.copy()
            out.loc[(out.index >= from_time) & (out.index <= to_time)]
        return out

    def change_reference_frame(
        self,
        origin: Union[np.ndarray, list] = np.array([0, 0, 0]),
        axis1: Union[np.ndarray, list] = np.array([1, 0, 0]),
        axis2: Union[np.ndarray, list] = np.array([0, 1, 0]),
        axis3: Union[np.ndarray, list] = np.array([0, 0, 1]),
        inplace=True,
    ):
        if inplace:
            obj = self
        else:
            obj = self.copy()
        for i in ["origin", "force", "torque"]:

            # rotate origin
            cols = [i for i in self.columns if i[0] == "origin"]
            rotated = signalprocessing.to_reference_frame(
                self[cols],  # type: ignore
                origin,
                axis1,
                axis2,
                axis3,
            )
            self.loc[self.index, cols] = rotated

            # rotate force
            cols = [i for i in self.columns if i[0] == "force"]
            rotated = signalprocessing.to_reference_frame(
                self[cols],  # type: ignore
                [0, 0, 0],
                axis1,
                axis2,
                axis3,
            )
            self.loc[self.index, cols] = rotated

            # rotate torque
            cols = [i for i in self.columns if i[0] == "torque"]
            rotated = signalprocessing.to_reference_frame(
                self[cols],  # type: ignore
                [0, 0, 0],
                axis1,
                axis2,
                axis3,
            )
            self.loc[self.index, cols] = rotated
        if not inplace:
            return obj


class StateFrame(SmartDataFrame):

    _ALLOWED_TYPES = (Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform)

    def __init__(
        self,
        strip: bool = True,
        reset_index: bool = False,
        **signals: Union[Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform],
    ):
        super().__init__()
        self.add(**signals, strip=strip, reset_index=reset_index)

    @property
    def signals1d(self):
        out: dict[str, Signal1D] = {}
        sub = self["Signal1D"]
        for lbl in sub.columns:
            out[lbl] = Signal1D(
                data=sub[lbl].values.astype(float).flatten(),
                time=sub[lbl].index.to_numpy(),
                unit=str(sub.columns[0][-1]),
                strip=False,
                reset_index=False,
            )  # type: ignore
        return out

    @property
    def signals3d(self):
        sub = self["Signal3D"]
        out: dict[str, Signal3D] = {}
        for lbl in sub.columns.get_level_values(0).unique():
            dfr = sub[lbl]
            out[lbl] = Signal3D(
                xarr=dfr["X"].values.astype(float).flatten(),
                yarr=dfr["Y"].values.astype(float).flatten(),
                zarr=dfr["Z"].values.astype(float).flatten(),
                time=dfr.index.to_numpy(),
                unit=str(dfr.columns[0][-1]),
                strip=False,
                reset_index=False,
            )  # type: ignore
        return out

    @property
    def emgsignals(self):
        sub = self["EMGSignal"]
        out: dict[str, EMGSignal] = {}
        for col in sub.columns:
            dfr = sub[col]
            lbl = "_".join([col[1], col[0]])
            out[lbl] = EMGSignal(
                data=dfr.values.astype(float).flatten(),
                time=dfr.index.to_numpy(),
                strip=False,
                reset_index=False,
            )  # type: ignore
        return out

    @property
    def points3d(self):
        sub = self["Point3D"]
        out: dict[str, Point3D] = {}
        for lbl in sub.columns.get_level_values(0).unique():
            dfr = sub[lbl]
            out[lbl] = Point3D(
                xarr=dfr["X"].values.astype(float).flatten(),
                yarr=dfr["Y"].values.astype(float).flatten(),
                zarr=dfr["Z"].values.astype(float).flatten(),
                time=dfr.index.to_numpy(),
                strip=False,
                reset_index=False,
            )  # type: ignore
        return out

    @property
    def forceplatforms(self):
        sub = self["ForcePlatform"]
        out: dict[str, ForcePlatform] = {}
        for lbl in sub.columns.get_level_values(0).unique():
            dfr = sub[lbl]
            objs = {}
            for src in ["origin", "force", "torque"]:
                if src == "origin":
                    classfun = lambda x, y, z, t, u: Point3D(x, y, z, t, False, False)  # type: ignore
                else:
                    classfun = lambda x, y, z, t, u: Signal3D(x, y, z, t, u, False, False)  # type: ignore
                objs[src] = classfun(
                    dfr["X"].values.astype(float).flatten(),
                    dfr["Y"].values.astype(float).flatten(),
                    dfr["Z"].values.astype(float).flatten(),
                    dfr.index.to_numpy(),
                    str(dfr.columns[0][-1]),
                    False,  # type: ignore
                    False,
                )
            out[lbl] = ForcePlatform(
                origin=objs["origin"],
                force=objs["force"],
                torque=objs["torque"],
                strip=False,
                reset_index=False,
            )  # type: ignore

        return out

    def add(
        self,
        strip: bool = True,
        reset_index: bool = False,
        **signals: Union[Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform],
    ):
        for key, signal in signals.items():

            # check the input type
            stype = type(signal).__name__
            if not isinstance(signal, self._ALLOWED_TYPES):
                msg = f"Invalid signal type: {SocketType}. "
                types = ", ".join(t.__name__ for t in self._ALLOWED_TYPES)
                msg += f"Allowed types are: {types}"
                raise TypeError(msg)

            # concatenate
            obj = signal.copy()
            obj.columns = MultiIndexSearch.from_tuples(
                [(stype, key, *i) for i in obj.columns]
            )
            self.loc[obj.index, obj.columns] = obj.values

        # transform
        if strip:
            self.strip(inplace=True)
        if reset_index:
            self.reset_index(inplace=True)

    def strip(self, inplace=True):
        valid = ~np.isnan(self.values).all(axis=-1)
        if not np.any(valid):
            raise ValueError("All origin values are NaN; cannot strip.")
        index0, index1 = np.where(valid)[0][[0, -1]]
        index = np.arange(index0, index1 + 1)
        if inplace:
            self = self.iloc[index]
        else:
            out = self.copy()
            return out.iloc[index]

    def reset_index(self, inplace=True):
        if inplace:
            self.index = self.index - self.index[0]
        else:
            out = self.copy()
            out.index = out.index - self.index[0]
            return out

    def slice(self, from_time, to_time, inplace=True):
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if inplace:
            self.loc[(self.index >= from_time) & (self.index <= to_time)]
        else:
            out = self.copy()
            out.loc[(out.index >= from_time) & (out.index <= to_time)]
        return out

    def change_reference_frame(
        self,
        origin: Union[np.ndarray, list] = np.array([0, 0, 0]),
        axis1: Union[np.ndarray, list] = np.array([1, 0, 0]),
        axis2: Union[np.ndarray, list] = np.array([0, 1, 0]),
        axis3: Union[np.ndarray, list] = np.array([0, 0, 1]),
        inplace=True,
    ):
        if inplace:
            obj = self.copy()
        for key, value in self.points3d.items():  # type: ignore
            cols = [("Point3D", key, *i) for i in value.columns]
            if inplace:
                self.loc[value.index, cols] = value.change_reference_frame(
                    origin, axis1, axis2, axis3
                )
            else:
                obj.loc[value.index, cols] = value.change_reference_frame(
                    origin, axis1, axis2, axis3
                )
        for key, value in self.forceplatforms.items():  # type: ignore
            cols = [("ForcePlatform", key, *i) for i in value.columns]
            if inplace:
                self.loc[value.index, cols] = value.change_reference_frame(
                    origin, axis1, axis2, axis3
                )
            else:
                obj.loc[value.index, cols] = value.change_reference_frame(
                    origin, axis1, axis2, axis3
                )
        if inplace:
            return obj

    def save(self, file_path: str):
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + self.__class__.__name__.lower()
        if not file_path.endswith(extension):
            file_path += extension
        overwrite = False
        while exists(file_path) and not overwrite:
            overwrite = messages.askyesnocancel(
                title="File already exists",
                message="the provided file_path already exist. Overwrite?",
            )
            if not overwrite:
                file_path = file_path[: len(extension)] + "_" + extension
        if not exists(file_path) or overwrite:
            with open(file_path, "wb") as buf:
                pickle.dump(self, buf)

    @classmethod
    def load(cls, file_path: str):
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + cls.__name__.lower()
        if not file_path.endswith(extension):
            raise ValueError(f"'file_path' must have {extension}.")
        try:
            with open(file_path, "rb") as buf:
                return pickle.load(buf)
        except Exception:
            raise RuntimeError(f"an error occurred importing {file_path}.")

    @classmethod
    def from_tdf_file(
        cls,
        file: str,
        strip: bool = True,
        reset_index: bool = False,
    ):
        # check of the input file
        if (
            not isinstance(file, str)
            or not file.lower().endswith(".tdf")
            or not exists(file)
        ):
            raise ValueError("file must be and existing .tdf object")

        # read the tdf file
        tdf = io.read_tdf(file)
        obj = cls(reset_index=False, strip=False)  # type: ignore

        # extract raw marker data
        try:
            markers: pd.DataFrame = tdf["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
            points = {}
            for lbl in markers.columns.get_level_values(0).unique():  # type: ignore
                pnt = Point3D(
                    xarr=markers[lbl]["X"].values.astype(float).flatten(),
                    yarr=markers[lbl]["Y"].values.astype(float).flatten(),
                    zarr=markers[lbl]["Z"].values.astype(float).flatten(),
                    time=markers.index.to_numpy(),
                    strip=False,
                    reset_index=False,
                )  # type: ignore
                points[lbl] = pnt
                obj.add(**points, strip=False, reset_index=False)
        except Exception as exc:
            warn(
                f"the {file} file does not contain Point3D data.",
                RuntimeWarning,
            )

        # extract raw forceplatform data
        try:
            forceplatforms: pd.DataFrame = tdf["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]  # type: ignore
            fps = {}
            for lbl in forceplatforms.columns.get_level_values(0).unique():  # type: ignore
                origin = Point3D(
                    xarr=forceplatforms[lbl]["ORIGIN"]["X"]
                    .values.astype(float)
                    .flatten(),
                    yarr=forceplatforms[lbl]["ORIGIN"]["Y"]
                    .values.astype(float)
                    .flatten(),
                    zarr=forceplatforms[lbl]["ORIGIN"]["Z"]
                    .values.astype(float)
                    .flatten(),
                    time=forceplatforms.index.to_numpy(),
                    strip=False,
                    reset_index=False,
                )  # type: ignore
                force = Signal3D(
                    xarr=forceplatforms[lbl]["FORCE"]["X"]
                    .values.astype(float)
                    .flatten(),
                    yarr=forceplatforms[lbl]["FORCE"]["Y"]
                    .values.astype(float)
                    .flatten(),
                    zarr=forceplatforms[lbl]["FORCE"]["Z"]
                    .values.astype(float)
                    .flatten(),
                    time=forceplatforms.index.to_numpy(),
                    unit="N",
                    strip=False,
                    reset_index=False,
                )  # type: ignore
                torque = Signal3D(
                    xarr=forceplatforms[lbl]["TORQUE"]["X"]
                    .values.astype(float)
                    .flatten(),
                    yarr=forceplatforms[lbl]["TORQUE"]["Y"]
                    .values.astype(float)
                    .flatten(),
                    zarr=forceplatforms[lbl]["TORQUE"]["Z"]
                    .values.astype(float)
                    .flatten(),
                    time=forceplatforms.index.to_numpy(),
                    unit="Nm",
                    strip=False,
                    reset_index=False,
                )  # type: ignore
                plt = ForcePlatform(
                    origin,
                    force,  # type: ignore
                    torque,  # type: ignore
                    False,
                    False,
                )  # type: ignore
                fps[lbl] = plt
            obj.add(**fps, strip=False, reset_index=False)
        except Exception as exc:
            warn(
                message=f"the {file} file does not contain ForcePlatform data.",
                category=RuntimeWarning,
            )

        # extract raw EMG data
        try:
            emgs: pd.DataFrame = tdf["EMG"]["TRACKS"]  # type: ignore
            for muscle in emgs.columns.get_level_values(0).unique():  # type: ignore
                muscle_data = emgs[muscle]
                emgs_dict = {}
                for side in muscle_data.columns.get_level_values(0).unique():
                    name = "_".join([side, muscle])
                    emg = EMGSignal(
                        data=muscle_data[side].values.astype(float),
                        time=muscle_data.index.to_numpy(),
                        name=name,
                        strip=False,
                        reset_index=False,
                    )  # type: ignore
                    emgs_dict[name] = emg
                    obj.add(**emgs_dict, strip=False, reset_index=False)
        except Exception as exc:
            warn(
                f"the {file} file does not contain EMGSignal data.",
                RuntimeWarning,
            )

        # apply the reset_index and strip options
        if not isinstance(strip, bool):
            raise ValueError("strip must be a boolean.")
        if not isinstance(reset_index, bool):
            raise ValueError("reset_index must be a boolean.")
        if strip:
            obj.strip(inplace=True)
        if reset_index:
            obj.reset_index(inplace=True)

        return obj
