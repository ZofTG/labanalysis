"""timeseries record module"""

# -*- coding: utf-8 -*-


#! IMPORTS


from collections import UserDict
import numpy as np
import pandas as pd
import pint

from ...io.read.btsbioengineering import read_tdf
from ..processingpipeline import *
from ..timeseries import *
from .forceplatform import *

ureg = pint.UnitRegistry()


__all__ = ["TimeseriesRecord"]


class TimeseriesRecord(UserDict):
    """
    A dictionary-like container for Timeseries, TimeseriesRecord, and ForcePlatform objects,
    supporting type filtering and DataFrame conversion.

    Parameters
    ----------
    vertical_axis : str, optional
        The label for the vertical axis (default "Y").
    anteroposterior_axis : str, optional
        The label for the anteroposterior axis (default "Z").
    strip : bool, optional
        If True, remove leading/trailing rows or columns that are all NaN from all contained objects (default True).
    reset_time : bool, optional
        If True, reset the time index to start at zero for all contained objects (default True).
    **signals : dict
        Key-value pairs of Timeseries subclasses, TimeseriesRecord, or ForcePlatform to include in the record.

    Attributes
    ----------
    _vertical_axis : str
        The vertical axis label.
    _antpos_axis : str
        The anteroposterior axis label.

    Methods
    -------
    copy()
        Return a deep copy of the TimeseriesRecord.
    strip(axis=0, inplace=False)
        Remove leading/trailing rows or columns that are all NaN from all contained objects.
    reset_time(inplace=False)
        Reset the time index to start at zero for all contained objects.
    apply(func, axis=0, inplace=False, *args, **kwargs)
        Apply a function or ProcessingPipeline to all contained objects.
    fillna(value=None, n_regressors=None, inplace=False)
        Fill NaNs for all contained objects.
    to_dataframe()
        Convert the record to a pandas DataFrame with MultiIndex columns.
    from_tdf(filename)
        Create a TimeseriesRecord from a TDF file.
    """

    _vertical_axis: str
    _antpos_axis: str

    @property
    def vertical_axis(self):
        """
        Returns the vertical axis label used in force data.

        Returns
        -------
        str
            The vertical axis label.
        """
        return self._vertical_axis

    @property
    def anteroposterior_axis(self):
        """
        Returns the anteroposterior axis label used in force data.

        Returns
        -------
        str
            The anteroposterior axis label.
        """
        return self._antpos_axis

    @property
    def lateral_axis(self):
        """
        Returns the lateral axis label used in force data.

        Returns
        -------
        str
            The lateral axis label.
        """
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        if left_foot is not None:
            axes = left_foot["force"].columns
        elif right_foot is not None:
            axes = right_foot["force"].columns
        else:
            raise ValueError("not valid data have been found")
        ml = [
            i for i in axes if i not in [self.vertical_axis, self.anteroposterior_axis]
        ]
        if len(ml) != 1:
            raise ValueError("number of axes is not coherent with data structure.")
        return str(ml[0])

    def __init__(
        self,
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
        strip: bool = True,
        reset_time: bool = True,
        **signals,
    ):
        """
        Initialize a TimeseriesRecord.

        Parameters
        ----------
        **signals : dict
            Key-value pairs of Timeseries subclasses, TimeseriesRecord, or ForcePlatform.
        """
        super().__init__()
        self._data = {}
        for key, value in signals.items():
            if not issubclass(type(value), TimeseriesRecord) and not issubclass(
                type(value), Timeseries
            ):
                raise ValueError(
                    f"{key} must be a Timeseries or TimeseriesRecord "
                    + "subclass instance."
                )
            self[key] = value

        # check axes consistency
        axes = []
        for signal in self.signals3d.values():
            axes += signal.columns
        for signal in self.points3d.values():
            axes += signal.columns
        for signal in self.Forceplatforms.values():
            for axis in signal.values():
                axes += axis.columns
        axes = np.unique(axes)
        if len(axes) != 3:
            raise ValueError("axes must be the same across all 3D elements.")

        # check vertical axis
        if not isinstance(vertical_axis, str):
            raise ValueError("vertical_axis must be a string")
        if vertical_axis not in axes:
            raise ValueError(f"vertical_axis must be any of {axes}")
        self._vertical_axis = vertical_axis

        # check anteroposterior axis
        if not isinstance(anteroposterior_axis, str):
            raise ValueError("anteroposterior_axis must be a string")
        if anteroposterior_axis not in axes:
            raise ValueError(f"anteroposterior_axis must be any of {axes}")
        self._antpos_axis = anteroposterior_axis

        # evaluate strip
        if not isinstance(strip, bool):
            raise ValueError("'strip' must be True or False")
        if strip:
            self.strip(inplace=True)

        # evaluate reset time
        if not isinstance(reset_time, bool):
            raise ValueError("'reset_time' must be True or False")
        if reset_time:
            self.reset_time(inplace=True)

    def _validate(self, key, value):
        """
        Internal: Validate items that are required to be added
        to the record.
        """
        if not isinstance(value, (Timeseries, TimeseriesRecord)) or not issubclass(
            type(value), (Timeseries, TimeseriesRecord)
        ):
            raise TypeError(
                f"Value for key '{key}' must be a Timeseries, TimeseriesRecord, or ForcePlatform"
            )

    def __getitem__(self, key):
        """
        Get an item by key.

        Parameters
        ----------
        key : str
            Key name.

        Returns
        -------
        Timeseries, TimeseriesRecord or their subclasses
            The value associated with the key.
        """
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Set an item by key.

        Parameters
        ----------
        key : str
            Key name.
        value : Timeseries, TimeseriesRecord, or ForcePlatform
            Value to set.

        Raises
        ------
        TypeError
            If value is not a valid type.
        """
        self._validate(key, value)
        self._data[key] = value

    def __getattr__(self, name):
        if name in list(self._data.keys()):
            return self[name]
        raise ValueError("name is not a valid attribute of this TimeseriesRecord")

    def __setattr__(self, key, value):
        self._validate(key, value)
        self[key] = value

    def _filter_by_type(self, cls):
        """
        Internal: Filter contained items by type.

        Parameters
        ----------
        cls : type

        Returns
        -------
        TimeseriesRecord
            A view (not a copy) of the filtered items. Changes to elements affect the original TimeseriesRecord.
        """
        filtered = TimeseriesRecord()
        filtered._data = {k: v for k, v in self._data.items() if isinstance(v, cls)}
        return filtered

    @property
    def points3d(self):
        """
        Get all Point3D objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(Point3D)

    @property
    def signals3d(self):
        """
        Get all Signal3D objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(Signal3D)

    @property
    def signals1d(self):
        """
        Get all Signal1D objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(Signal1D)

    @property
    def emgsignals(self):
        """
        Get all EMGSignal objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(EMGSignal)

    @property
    def forceplatforms(self):
        """
        Get all ForcePlatform objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(ForcePlatform)

    @property
    def index(self):
        """
        Get the index shared across all elements in the record.

        Returns
        -------
        1D numpy array of floats
            A sorted, unique array of all time indices.
        """
        idxs = np.concatenate([i.index for i in self._data.values()])
        idxs = idxs.astype(float).flatten()
        return np.unique(idxs)

    @property
    def columns(self):
        """
        Get the columns of all elements in the record.

        Returns
        -------
        dict
            A dictionary where keys are the names of the contained objects and values are the columns of each object.
        """
        return {k: v.columns for k, v in self._data.items()}

    @property
    def units(self):
        """
        Get the units of all elements in the record.

        Returns
        -------
        dict
            A dictionary where keys are the names of the contained objects and values are the units of each object.
        """
        return {
            k: v.unit if not isinstance(v, ForcePlatform) else v.units
            for k, v in self._data.items()
        }

    @property
    def data(self):
        """
        Get the data valued contained on all elements in the record.

        Returns
        -------
        2D numpy array of floats
            A concatenated array of all data values.
        """
        all_data = []
        for key, ts in self._data.items():
            data = ts.data if isinstance(ts, ForcePlatform) else ts._data
            all_data.append(data)
        return np.concatenate(all_data, axis=1).astype(float)

    def to_dataframe(self):
        """
        Convert the record to a pandas DataFrame with MultiIndex columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all the data from the TimeseriesRecord.
        """
        if not self._data:
            return pd.DataFrame()
        frames = []
        for key, val in self._data.items():
            new = val.to_dataframe()
            new.columns = pd.MultiIndex.from_tuples([(key, *i) for i in new.columns])
            frames += [new]
        return pd.concat(frames, axis=1).sort_index(axis=1)

    @property
    def vertical_force(self):
        fps = self.forceplatorms
        if len(fps) == 0:
            raise ValueError("This record does not contain ForcePlatform elements.")
        time = self.index.tolist()
        vgrf = np.zeros_like(time)
        units = []
        for fp in fps.values():
            vrgf += np.asarray(fp["force"][self.vertical_axis], float)
            units += [fp["force"].unit]
        units = np.unique(units)
        if len(units) != 1:
            raise ValueError("distinct units were found on forceplatforms elements.")
        return Signal1D(
            vgrf,
            time,
            str(units[0]),
            self.vertical_axis,
        )

    @property
    def centre_of_pressure(self):
        fps = self.forceplatforms
        if len(fps) == 0:
            raise ValueError("This record does not contain ForcePlatform elements.")

        f_total = np.zeros((self.shape[0], 3))
        m_total = np.zeros((self.shape[0], 3))
        axes = []
        for fp in fps.values():
            r = np.asarray(fp["origin"].data, float)
            F = np.asarray(fp["force"].data, float)
            M = np.asarray(fp["torque"].data, float)
            f_total += F
            m_total += M + np.cross(r, F)
            if len(axes) == 0:
                axes = fp["origin"].columns
        cop = np.cross(f_total, m_total) / np.dot(f_total, f_total)

        return Point3D(
            cop,
            self.index.tolist(),
            self["origin"].unit,
            axes,
        )

    @classmethod
    def from_tdf(cls, filename: str):
        """
        Create a TimeseriesRecord from a TDF file.

        Parameters
        ----------
        filename : str
            Path to the TDF file.

        Returns
        -------
        TimeseriesRecord
            A TimeseriesRecord populated with the data from the TDF file.
        """
        data = read_tdf(filename)
        record = cls()

        # Handle 3D points from CAMERA TRACKED
        if data.get("CAMERA") and data["CAMERA"].get("TRACKED"):  # type: ignore
            df = data["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
            for label in df.columns.get_level_values(0).unique():
                sub_df = df[label]
                record[label] = Point3D(
                    data=sub_df.values,
                    index=sub_df.index.tolist(),
                    columns=sub_df.columns.get_level_values(0),
                    unit=sub_df.columns[0][-1],
                )

        # Handle EMG signals
        if data.get("EMG") and data["EMG"].get("TRACKS") is not None:  # type: ignore
            df = data["EMG"]["TRACKS"]  # type: ignore
            for col in df.columns:
                signal = df[col]
                muscle_name, side, unit = col
                record[f"{side}_{muscle_name}".lower()] = EMGSignal(
                    data=signal.values.reshape(-1, 1),
                    index=df.index.tolist(),
                    muscle_name=muscle_name.lower(),
                    side=side.lower(),
                    unit=unit,
                )

        # Handle Force Platforms
        if data.get("FORCE_PLATFORM") and data["FORCE_PLATFORM"].get("TRACKED"):  # type: ignore
            df = data["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]  # type: ignore
            for label in df.columns.get_level_values("LABEL").unique():
                origin = df[label]["ORIGIN"]
                force = df[label]["FORCE"]
                torque = df[label]["TORQUE"]
                record[label] = ForcePlatform(
                    origin=Point3D(
                        data=origin.values,
                        index=origin.index.tolist(),
                        columns=origin.columns.get_level_values(0),
                        unit=origin.columns[0][-1],
                    ),
                    force=Signal3D(
                        data=force.values,
                        index=force.index.tolist(),
                        columns=force.columns.get_level_values(0),
                        unit=force.columns[0][-1],
                    ),
                    torque=Signal3D(
                        data=torque.values,
                        index=torque.index.tolist(),
                        columns=torque.columns.get_level_values(0),
                        unit=torque.columns[0][-1],
                    ),
                )

        return record

    def copy(self):
        """
        Return a deep copy of the TimeseriesRecord.

        Returns
        -------
        TimeseriesRecord
            A new TimeseriesRecord object with the same data.
        """
        return TimeseriesRecord(
            **{i: v for i, v in self.signals1d},
            **{i: v for i, v in self.signals3d},
            **{i: v for i, v in self.emgsignals},
            **{i: v for i, v in self.points3d},
            **{i: v for i, v in self.forceplatforms},
        )

    def strip(self, axis=0, inplace=False):
        """
        Remove leading/trailing rows or columns that are all NaN from all
        contained Timeseries-like objects.

        Parameters
        ----------
        axis : int, optional
            0 for rows, 1 for columns (default: 0).
        inplace : bool, optional
            If True, modifies in place. If False, returns a new TimeseriesRecord.

        Returns
        -------
        TimeseriesRecord or None
            Stripped TimeseriesRecord if inplace is False, otherwise None.
        """
        if inplace:
            for v in self._data.values():
                if hasattr(v, "strip"):
                    v.strip(axis=axis)
        else:
            out = self.copy()
            out.strip(axis=axis, inplace=True)
            return out

    def reset_time(self, inplace=False):
        """
        Reset the time index to start at zero for all contained Timeseries-like
        objects.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify in place. If False, return a new TimeseriesRecord.

        Returns
        -------
        TimeseriesRecord or None
            A TimeseriesRecord with reset time if inplace is False, otherwise None.
        """
        if inplace:
            for v in self._data.values():
                if hasattr(v, "reset_time"):
                    v.reset_time(inplace=True)
        else:
            out = self.copy()
            out.reset_time(inplace=True)
            return out

    def apply(self, func, axis=0, inplace=False, *args, **kwargs):
        """
        Apply a function or ProcessingPipeline to all contained objects.

        Parameters
        ----------
        func : callable or ProcessingPipeline
            Function, class, or method to apply to the data, or a ProcessingPipeline.
        axis : int, optional
            0 to apply by row, 1 to apply by column (default: 0).
        inplace : bool, optional
            If True, modifies self. If False, returns a new object.
        *args, **kwargs : additional arguments to pass to func.

        Returns
        -------
        TimeseriesRecord or None
            If inplace is False, returns a new TimeseriesRecord with the function applied.
            If inplace is True, returns None.
        """
        if isinstance(func, ProcessingPipeline):
            if inplace:
                func.apply(*list(self.values()), inplace=True, *args, **kwargs)
            else:
                out = self.copy()
                for k, v in out.items():
                    out[k] = func.apply(v, inplace=False, *args, **kwargs)
                return out
        else:
            out = self if inplace else self.copy()
            for k, v in self._data.items():
                v.apply(func, axis=axis, inplace=True, *args, **kwargs)
            if inplace:
                return out

    def fillna(self, value=None, n_regressors=None, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using advanced imputation for all contained objects.

        Parameters
        ----------
        value : float or int or None, optional
            Value to use for NaNs. If None, use interpolation or regression.
        n_regressors : int or None, optional
            Number of regressors to use for regression-based imputation. If None, use cubic spline interpolation.
        inplace : bool, optional
            If True, fill in place. If False, return a new object.

        Returns
        -------
        ForcePlatform
            Filled record.
        """
        if inplace:
            for k, v in self._data.items():
                self[k].fillna(value, n_regressors)
        else:
            out = self.copy()
        for k, v in out._data.items():
            v.fillna(value=value, n_regressors=n_regressors, inplace=True)
        return out
