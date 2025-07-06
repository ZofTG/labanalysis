"""
frames module containing useful classes for kinetic, kinematic and EMG data
analysis
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import pickle
from copy import deepcopy
from os.path import exists
from types import FunctionType, LambdaType, MethodType
from typing import Any, Iterable, Literal
from warnings import warn

import numpy as np
import pandas as pd

from . import messages, signalprocessing
from .io.read import read_tdf

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


class TimeSeries:
    """
    Base class defining a set of data sampled over time.

    Parameters
    ----------
    data: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the data sampled over time

    time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the time samples of each object. The time samples must be in seconds.

    unit: str (default = "")
        the unit of measurement.

    name: str (default = "")
        the name of the timeseries.

    strip: bool (default=True)
        if true, remove missing data at the beginning and at the end of the
        data.

    reset_index: bool (default=True)
        ensure the time array starts from zero.

    Attributes
    ----------
    processing_options
        the parameters to set the filtering of the provided signals

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    copy
        return a copy of the object.

    slice
        return a subset of the StateFrame.

    process_data
        process internal data to remove/replace missing values and smooth the
        signals.

    is_processed
        returns True if the actual object already run the process data method

    save
        save the object.

    load (classmethod)
        load an instance of the object from local file.
    """

    # *class variables

    data: pd.DataFrame
    _processing_options: dict[str, Any]
    _name: str
    _unit: str

    # *attributes

    @property
    def name(self):
        """return the name of the object"""
        return self._name

    @property
    def unit(self):
        """return the unit of measurement"""
        return self._unit

    @property
    def processing_options(self):
        """the processing options"""
        return self._processing_options

    # *methods

    def __str__(self):
        return self.to_dataframe().__str__()

    def __repr__(self):
        return self.to_dataframe().__repr__()

    def strip(self, inplace: bool = True):
        """
        remove missing values at the beginning or at the end of the data

        Parameters
        ----------
        inplace:bool = True
            if True, the operations are made directly in the current object.

        Returns
        -------
        if inplace=False, return the current object with the missing values
        removed at the beginning and at the end.
        """
        valid = self.data.notna().all(axis=1).values.astype(bool)
        time = self.data.index.to_numpy()
        index0, index1 = np.where(valid)[0][[0, -1]]
        time0, time1 = time[[index0, index1]]
        self.slice(time0, time1, inplace=True)
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if not inplace:
            return self.copy()

    def reset_index(self, inplace: bool = True):
        """
        ensure the time array starts from zero

        Parameters
        ----------
        inplace : bool (default=True)
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations.

        Returns
        -------
        if inplace=False, return the current object.
        """
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        resetter = lambda x: x[0] * np.ones_like(x)
        self.data.index -= resetter(self.data.index.to_numpy())
        if not inplace:
            return self.copy()

    def process(
        self,
        pipeline: (
            FunctionType
            | MethodType
            | LambdaType
            | list[FunctionType | MethodType | LambdaType]
        ),
        inplace: bool = True,
    ):
        """
        process the available data according to the given options

        Parameters
        ----------
        pipeline: FunctionType | MethodType | list[FunctionType | MethodType]
            the processing pipeline to be applied to the data.

        strip: bool = True
            remove nans at the beginning and end of the data

        reset_index: bool = False
            if True the reduced data are reindexed such as they start from zero

        inplace: bool = True
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations
        """
        # apply the pipeline
        if not isinstance(pipeline, list):
            pipeline = [pipeline]
        for func in pipeline:
            if not isinstance(func, (FunctionType, MethodType, LambdaType)):
                raise ValueError("pipeline must be a function or a list of functions")
            self.data = self.data.apply(func, raw=True, axis=0)

        # check for the 'inplace' input
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")

        # track the processing steps
        self._processing_options["inplace"] = inplace
        if "pipeline" not in list(self._processing_options.keys()):
            self._processing_options["pipeline"] = []
        self._processing_options["pipeline"] += pipeline

        # return a copy of this object is inplace is False
        if not inplace:
            return self.copy()

    def to_dataframe(self):
        """return a pandas DataFrame object containing all the available data"""
        if self.data.shape[1] == 1:
            cols = [(self.name, self.unit)]
        else:
            cols = self.data.columns.to_numpy()
            cols = [(self.name, *i, self.unit) for i in cols]
        out = self.data.copy()
        out.columns = pd.MultiIndex.from_tuples(cols)
        return out

    def copy(self):
        """create a copy of this object"""
        return deepcopy(self)

    def is_processed(self):
        """
        returns True if the actual object already run the process data method
        """
        return len(self._processing_options) > 0

    def slice(
        self,
        from_time: int | float | np.number,
        to_time: int | float | np.number,
        inplace: bool = True,
    ):
        """
        return a subset of the object with time index within the given range

        Parameters
        ----------
        from_time: int | float | np.number
            the returned slice starts from the provided time.

        to_time: int | float | np.number
            the returned slice ends at the provided time.

        inplace: bool = True
            if False a copy of the sliced object is returned.

        Returns
        -------
        if inplace=False, return a subset of the original object with time
            index within the given from_time and to_time.
        """
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean.")
        out = self if inplace else self.copy()
        mask = out.data.index >= from_time
        mask &= out.data.index <= to_time
        out.data = out.data.loc[mask]
        if not inplace:
            return out

    def _validate_inputs(self, *objs: object):
        """
        private method used to check if obj is a numeric 1D array

        Parameters
        ----------
        *obj: objects
            the object to be checked.
        """
        return objs[0]

    def save(self, file_path: str):
        """
        save the object to the input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the object name. If this is not the case, the appropriate extension
            is appended.
        """
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

    # *constructors

    def __init__(
        self,
        data: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        unit: str = "",
        name: str = "",
        strip: bool = True,
        reset_index: bool = False,
    ):
        # check the inputs
        xval = self._validate_inputs(data)
        if not isinstance(strip, bool):
            raise ValueError("'strip' must be True or False")
        if not isinstance(reset_index, bool):
            raise ValueError("'reset_index' must be True or False")
        if not isinstance(name, str):
            raise ValueError("'name' must be a str object.")

        # generate the data object
        self.data = pd.DataFrame(
            data=xval,  # type: ignore
            index=pd.Series(time, name="Time (s)"),
        )

        # strip
        if strip:
            self.strip(inplace=True)

        # reset index
        if reset_index:
            self.reset_index(inplace=True)

        # initialize the processing options data
        self._processing_options = {}
        self._name = name
        self._unit = unit

    @classmethod
    def load(cls, file_path: str):
        """
        load the object from an input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the object name. If this is not the case, the appropriate extension
            is appended.

        Returns
        -------
        obj: Self
            the object
        """
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


class Signal1D(TimeSeries):
    """
    Base class defining a 1D signal sampled over time.

    Parameters
    ----------
    amplitude: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the signal amplitude

    time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the time samples of each object. The time samples must be in seconds.

    unit: str (default = "")
        the unit of measurement.

    name: str (default = "")
        the name of the timeseries.

    strip: bool (default=True)
        if true, remove missing data at the beginning and at the end of the
        data.

    reset_index: bool (default=True)
        ensure the time array starts from zero.

    Attributes
    ----------
    processing_options
        the parameters to set the filtering of the provided signals

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    copy
        return a copy of the object.

    slice
        return a subset of the StateFrame.

    process_data
        process internal data to remove/replace missing values and smooth the
        signals.

    is_processed
        returns True if the actual object already run the process data method

    save
        save the object.

    load (classmethod)
        load an instance of the object from local file.
    """

    # *methods

    def _validate_inputs(self, *objs: object):
        """
        private method used to check if obj is a numeric 1D array

        Parameters
        ----------
        *objs: objects
            the object to be checked.
        """
        if not isinstance(objs, Iterable):
            raise ValueError("only one object is accepted")
        if len(objs) > 1:
            raise ValueError("only one object is accepted")
        msg = "The object must be castable to a numpy 1D array"
        obj = objs[0]  # type: ignore
        try:
            new = np.array([obj]).astype(float).flatten()
        except Exception as exc:
            raise ValueError(msg) from exc
        return new

    # *constructors

    def __init__(
        self,
        amplitude: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        name: str = "",
        unit: str = "",
        strip: bool = True,
        reset_index: bool = False,
    ):
        super().__init__(
            data=amplitude,
            time=time,
            name=name,
            unit=unit,
            strip=strip,
            reset_index=reset_index,
        )


class Signal3D(TimeSeries):
    """
    Base class defining 3D objects sampled over time.

    Parameters
    ----------
    xarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the x coordinates of the object

    yarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the y coordinates of the object

    zarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the z coordinates of the object

    time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the time samples of each object. The time samples must be in seconds.

    name: str (default = "")
        the name of the timeseries.

    unit: str
        the unit of measurement of the coordinates.

    strip: bool (default=True)
        if true, remove missing data at the beginning and at the end of the
        data.

    reset_index: bool (default=True)
        ensure the time array starts from zero.

    Attributes
    ----------
    processing_options
        the parameters to set the filtering of the provided signals

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    change_reference_frame
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

    copy
        return a copy of the object.

    slice
        return a subset of the StateFrame.

    process_data
        process internal data to remove/replace missing values and smooth the
        signals.

    is_processed
        returns True if the actual object already run the process data method

    save
        save the object.

    load (classmethod)
        load an instance of the object from local file.
    """

    # *methods

    def change_reference_frame(
        self,
        origin: np.ndarray | list[float | int] = [0, 0, 0],
        axis1: np.ndarray | list[float | int] = [1, 0, 0],
        axis2: np.ndarray | list[float | int] = [0, 1, 0],
        axis3: np.ndarray | list[float | int] = [0, 0, 1],
        inplace: bool = True,
    ):
        """
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

        Parameters
        ----------
        origin: Iterable[int | float] = [0, 0, 0]
            The coordinates of the origin of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis1: Iterable[int | float] | pd.DataFrame | None = [1, 0, 0]
            The coordinates of the first axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis2: Iterable[int | float] | pd.DataFrame | None = [0, 1, 0]
            The coordinates of the second axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis3: Iterable[int | float] | pd.DataFrame | None = [0, 1, 0]
            The coordinates of the third axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        inplace: bool = True
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations.

        Returns
        -------
        Nothing if 'inplace = True', otherwise a copy of the actual object
        with the rotated axes.

        Note
        ----
        This method uses internally the Gram-Schmidt ortogonalization agorithm
        to derive the orthogonal unit versors from the provided axes.
        Then it align all the available markers and forceplatforms to the
        input origin and finally it rotates them according to the input axes.
        """
        if self.data.shape[0] == 0:
            return self.copy()

        data = self.data.copy()
        idx = data.index
        for lbl in np.unique(data.columns.get_level_values(0)):
            coords = [i for i in data.columns if i[0] == lbl]
            data.loc[idx, coords] = signalprocessing.to_reference_frame(
                obj=data.loc[idx, coords],
                origin=origin,
                axis1=axis1,
                axis2=axis2,
                axis3=axis3,
            )
        self.data = data

        # handle the inplace input
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if not inplace:
            return self.copy()

    # *constructors

    def __init__(
        self,
        xarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        yarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        zarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        unit: str,
        name: str,
        strip: bool = True,
        reset_index: bool = False,
    ):

        # aggregate the inputs
        outs = []
        for obj in [xarr, yarr, zarr]:
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                val = obj.values
            else:
                val = obj
            if not isinstance(val, (np.ndarray, list)):
                raise ValueError("The object must be castable to a numpy array")
            outs += [val]
        outs = np.vstack(np.atleast_2d(*outs)).T

        super().__init__(
            data=outs,
            time=time,
            unit=unit,
            name=name,
            strip=strip,
            reset_index=reset_index,
        )
        self.data.columns = pd.Index(["X", "Y", "Z"])


class EMGSignal(Signal1D):
    """
    Base class defining a 1D EMG signal sampled over time.

    Parameters
    ----------
    amplitude: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the signal amplitude

    time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the time samples of each object. The time samples must be in seconds.

    muscle: str
        the name of the muscle

    side: Literal['left', 'right']

    strip: bool (default=True)
        if true, remove missing data at the beginning and at the end of the
        data.

    reset_index: bool (default=True)
        ensure the time array starts from zero.

    Attributes
    ----------
    processing_options
        the parameters to set the filtering of the provided signals

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    copy
        return a copy of the object.

    slice
        return a subset of the StateFrame.

    process_data
        process internal data to remove/replace missing values and smooth the
        signals.

    is_processed
        returns True if the actual object already run the process data method

    save
        save the object.

    load (classmethod)
        load an instance of the object from local file.
    """

    _side: Literal["right", "left", "undefined"]

    # *attributes

    @property
    def side(self):
        """the muscle side"""
        return self._side

    # *methods

    def to_dataframe(self):
        """return a pandas DataFrame object containing all the available data"""
        cols = pd.MultiIndex.from_tuples([(self.name, self.side, self.unit)])
        out = self.data.copy()
        out.columns = cols
        return out

    # *constructors

    def __init__(
        self,
        amplitude: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        name: str,
        strip: bool = True,
        reset_index: bool = False,
    ):
        super().__init__(
            amplitude=amplitude,
            time=time,
            name="",
            unit="V",
            strip=strip,
            reset_index=reset_index,
        )

        # split the raw muscle name in words
        splits = name.lower().split(" ")

        # get the index of the word denoting the side
        side_idx = [i for i, v in enumerate(splits) if v in ["left", "right"]]
        side_idx = None if len(side_idx) == 0 else side_idx[0]

        # adjust the muscle name
        self._side = "undefined" if side_idx is None else splits.pop(side_idx)  # type: ignore
        self._name = "_".join(splits[:2])


class Point3D(Signal3D):
    """
    Point in a 3D space sampled over time

    Parameters
    ----------
    xarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the x coordinates of the object

    yarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the y coordinates of the object

    zarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the z coordinates of the object

    time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame
        the time samples of each object. The time samples must be in seconds.

    name: str
        the name of the object

    strip: bool (default=True)
        if true, remove missing data at the beginning and at the end of the
        data.

    reset_index: bool (default=True)
        ensure the time array starts from zero.

    Attributes
    ----------
    processing_options
        the parameters to set the filtering of the provided signals

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    change_reference_frame
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

    copy
        return a copy of the object.

    slice
        return a subset of the StateFrame.

    process_data
        process internal data to remove/replace missing values and smooth the
        signals.

    is_processed
        returns True if the actual object already run the process data method

    save
        save the object.

    load (classmethod)
        load an instance of the object from local file.
    """

    def __init__(
        self,
        xarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        yarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        zarr: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        time: np.ndarray | list[float | int] | pd.Series | pd.DataFrame,
        name: str,
        strip: bool = True,
        reset_index: bool = False,
    ):
        super().__init__(
            xarr=xarr,
            yarr=yarr,
            zarr=zarr,
            time=time,
            name=name,
            unit="m",
            strip=strip,
            reset_index=reset_index,
        )


class ForcePlatform:
    """
    Base class defining a force platform object sampled over time

    Parameters
    ----------
    origin: Signal3D | Point3D
        a Signal3D object defining the origin of the force platform

    force: Signal3D | Point3D
        a Signal3D object defining the force vector of the force platform

    torque: Signal3D | Point3D
        a Signal3D object defining the torque vector of the force platform

    name: str
        the name of the object

    strip: bool (default=True)
        if true, remove missing data at the beginning and at the end of the
        data.

    reset_index: bool (default=True)
        ensure the time array starts from zero.

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    change_reference_frame
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

    copy
        return a copy of the object.

    slice
        return a subset of the StateFrame.

    save
        save the object.

    load (classmethod)
        load an instance of the object from local file.
    """

    # *class variables

    _origin: Point3D
    _force: Signal3D
    _torque: Signal3D
    _name: str

    # *attributes

    @property
    def origin(self):
        """the orgin coordinates"""
        return self._origin

    @property
    def force(self):
        """the force vector amplitude"""
        return self._force

    @property
    def torque(self):
        """the torque coordinates"""
        return self._torque

    @property
    def data(self):
        """return the available data as dataframe"""
        pairs = [self.origin.to_dataframe()]
        pairs += [self.force.to_dataframe()]
        pairs += [self.torque.to_dataframe()]
        return pd.concat(pairs, axis=1)

    @property
    def name(self):
        """the name of the forceplatform object"""
        return self._name

    # *methods

    def __str__(self):
        return self.to_dataframe().__str__()

    def __repr__(self):
        return self.to_dataframe().__repr__()

    def slice(
        self,
        from_time: int | float | np.number,
        to_time: int | float | np.number,
        inplace: bool = True,
    ):
        """
        return a subset of the object with time index within the given range

        Parameters
        ----------
        from_time: int | float | np.number
            the returned slice starts from the provided time.

        to_time: int | float | np.number
            the returned slice ends at the provided time.

        inplace: bool = True
            if False a copy of the sliced object is returned.

        Returns
        -------
        a subset of the original object with time index within the given
            from_time and to_time.
        """
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        out = self if inplace else self.copy()
        out._origin.slice(from_time, to_time, inplace=True)
        out._force.slice(from_time, to_time, inplace=True)
        out._torque.slice(from_time, to_time, inplace=True)
        if not inplace:
            return out

    def strip(self, inplace: bool = True):
        """
        remove missing values at the beginning or at the end of the data

        Parameters
        ----------
        inplace:bool = True
            if True, the operations are made directly in the current object.

        Returns
        -------
        if inplace=False, return the current object with the missing values
        removed at the beginning and at the end.
        """
        data = [self.origin, self.force, self.torque]
        time0 = np.min([i.data.dropna().index[0] for i in data])
        time1 = np.max([i.data.dropna().index[-1] for i in data])
        self.slice(time0, time1, inplace=True)
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if not inplace:
            return self.copy()

    def reset_index(self, inplace: bool = True):
        """
        ensure the time array starts from zero

        Parameters
        ----------
        inplace : bool (default=True)
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations.

        Returns
        -------
        if inplace=False, return the current object.
        """
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")

        resetter = lambda x: x[0] * np.ones_like(x)
        out = self if inplace else self.copy()
        out._origin.data.index -= resetter(out._origin.data.index.to_numpy())
        out._force.data.index -= resetter(out._force.data.index.to_numpy())
        out._torque.data.index -= resetter(out._torque.data.index.to_numpy())
        if not inplace:
            return out

    def to_dataframe(self):
        """return a pandas DataFrame object containing all the available data"""
        data = self.data.copy()
        cols = data.columns.to_numpy()
        data.columns = pd.MultiIndex.from_tuples([(self.name, *i) for i in cols])
        return data

    def change_reference_frame(
        self,
        origin: np.ndarray | list[float | int] = [0, 0, 0],
        axis1: np.ndarray | list[float | int] = [1, 0, 0],
        axis2: np.ndarray | list[float | int] = [0, 1, 0],
        axis3: np.ndarray | list[float | int] = [0, 0, 1],
        inplace: bool = True,
    ):
        """
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

        Parameters
        ----------
        origin: Iterable[int | float] = [0, 0, 0]
            The coordinates of the origin of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis1: Iterable[int | float] | pd.DataFrame | None = [1, 0, 0]
            The coordinates of the first axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis2: Iterable[int | float] | pd.DataFrame | None = [0, 1, 0]
            The coordinates of the second axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis3: Iterable[int | float] | pd.DataFrame | None = [0, 1, 0]
            The coordinates of the third axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        inplace: bool = True
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations.

        Returns
        -------
        Nothing if 'inplace = True', otherwise a copy of the actual object
        with the rotated axes.

        Note
        ----
        This method uses internally the Gram-Schmidt ortogonalization agorithm
        to derive the orthogonal unit versors from the provided axes.
        Then it align all the available markers and forceplatforms to the
        input origin and finally it rotates them according to the input axes.
        """
        data = [self.origin, self.force, self.torque]
        if any([i.data.shape[0] == 0 for i in data]):
            return self.copy()
        self._origin.data.loc[self._origin.data.index, self._origin.data.columns] = (
            signalprocessing.to_reference_frame(
                obj=self._origin.data,
                origin=origin,
                axis1=axis1,
                axis2=axis2,
                axis3=axis3,
            )
        )
        self._force.data.loc[self._force.data.index, self._force.data.columns] = (
            signalprocessing.to_reference_frame(
                obj=self._force.data,
                origin=[0, 0, 0],
                axis1=axis1,
                axis2=axis2,
                axis3=axis3,
            )
        )
        self._torque.data.loc[self._torque.data.index, self._torque.data.columns] = (
            signalprocessing.to_reference_frame(
                obj=self._torque.data,
                origin=[0, 0, 0],
                axis1=axis1,
                axis2=axis2,
                axis3=axis3,
            )
        )

        # handle the inplace input
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if not inplace:
            return self.copy()

    def copy(self):
        """create a copy of this object"""
        return deepcopy(self)

    def save(self, file_path: str):
        """
        save the object to the input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the object name. If this is not the case, the appropriate extension
            is appended.
        """
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

    # *constructors

    def __init__(
        self,
        origin: Signal3D | Point3D,
        force: Signal3D | Point3D,
        torque: Signal3D | Point3D,
        name: str,
        strip: bool = True,
        reset_index=True,
    ):

        if not isinstance(origin, (Signal3D, Point3D)):
            raise ValueError("'origin' must be a Signal3D or Point3D instance.")
        if not isinstance(force, (Signal3D, Point3D)):
            raise ValueError("'force' must be a Signal3D or Point3D instance.")
        if not isinstance(torque, (Signal3D, Point3D)):
            raise ValueError("'torque' must be a Signal3D or Point3D instance.")
        if not isinstance(strip, bool):
            raise ValueError("'strip' must be a boolean.")
        if not isinstance(reset_index, bool):
            raise ValueError("'reset_index' must be a boolean.")
        self._origin = Point3D(
            xarr=origin.data["X"].values.astype(float),
            yarr=origin.data["Y"].values.astype(float),
            zarr=origin.data["Z"].values.astype(float),
            time=origin.data.index.to_numpy(),
            name="origin",
            strip=True,
            reset_index=False,
        )
        self._force = Signal3D(
            xarr=force.data["X"].values.astype(float),
            yarr=force.data["Y"].values.astype(float),
            zarr=force.data["Z"].values.astype(float),
            time=force.data.index.to_numpy(),
            name="force",
            unit="N",
            strip=True,
            reset_index=False,
        )
        self._torque = Signal3D(
            xarr=torque.data["X"].values.astype(float),
            yarr=torque.data["Y"].values.astype(float),
            zarr=torque.data["Z"].values.astype(float),
            time=torque.data.index.to_numpy(),
            name="torque",
            unit="Nm",
            strip=True,
            reset_index=False,
        )
        if strip:
            self.strip(inplace=True)
        if reset_index:
            self.reset_index(inplace=True)
        self._name = name

    @classmethod
    def load(cls, file_path: str):
        """
        load the object from an input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the object name. If this is not the case, the appropriate extension
            is appended.

        Returns
        -------
        obj: Self
            the object
        """
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


class StateFrame:
    """
    base class used for reading and perform basic processing of
    kinetic, kinematic and emg data.

    Parameters
    ----------
    *signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        the signals to be processed. The signals can be provided as a list of
        Signal1D, Signal3D, EMGSignal, Point3D or ForcePlatform objects.
        The signals must have the same time index.

    strip: bool (default=True)
        remove missing data from the start and the end of the StateFrame
        as it is created.

    reset_index: bool (default=True)
        remove missing data from the start and the end of the StateFrame
        as it is created.

    Attributes
    ----------
    signals: list[Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform]
        the list of available signals

    signals1d: list[Signal1D]
        the list of available Signal1D instances

    signals3d: list[Signal3D]
        the list of available Signal3D instances

    emgsignals: list[EMGSignal]
        the list of available EMGSignal instances

    points3d: list[Point3D]
        the list of available 3D points

    forceplatforms: list[ForcePlatform]
        the list of available ForcePlatform instances

    Methods
    -------
    add
        add novel signals to the current object.

    remove
        remove the first occurrence of the specified value.

    pop
        remove the signal at the specified index.

    to_dataframe
        return the available data as single pandas DataFrame.

    change_reference_frame
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

    copy
        return a copy of the object.

    slice
        return a subset of the StateFrame.

    strip
        remove missing data at the beginning and at the end of the file

    reset_index
        ensure that all data starts from time = 0
    """

    # *class variables

    _signals: list[Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform]

    # *attributes

    @property
    def signals(self):
        """return the list of signals"""
        return self._signals

    @property
    def signals1d(self):
        """return the list of 1D signals"""
        return [i for i in self.signals if type(i) == Signal1D]

    @property
    def signals3d(self):
        """return the list of 3D signals"""
        return [i for i in self.signals if type(i) == Signal3D]

    @property
    def emgsignals(self):
        """return the list of EMG signals"""
        return [i for i in self.signals if type(i) == EMGSignal]

    @property
    def points3d(self):
        """return the list of 3D points"""
        return [i for i in self.signals if type(i) == Point3D]

    @property
    def forceplatforms(self):
        """return the list of force platforms"""
        return [i for i in self.signals if type(i) == ForcePlatform]

    # *methods

    def __str__(self):
        return self.to_dataframe().__str__()

    def __repr__(self):
        return self.to_dataframe().__repr__()

    def add(
        self,
        *signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """add new signals to the current frame"""
        for signal in signals:
            if not isinstance(
                signal, (EMGSignal, Signal1D, Signal3D, Point3D, ForcePlatform)
            ):
                msg = "only EMGSignal, Signal1D, Signal3D, Point3D or "
                msg += "ForcePlatform objects are accepted."
                raise ValueError(msg)
            self._signals.append(signal)

    def remove(self, name: str):
        """remove the first occurrence of the object with the given name"""
        if not isinstance(name, str):
            raise ValueError("'name' must be a str")
        to_remove = [i for i in self.signals if i.name == name]
        if len(to_remove) > 0:
            to_remove = to_remove[0]
            self._signals.remove(to_remove)

    def pop(self, index: int):
        """remove the signal at the provided index"""
        if len(self._signals) > 0:
            self._signals.pop(index)

    def strip(self, inplace: bool = True):
        """
        remove missing values at the beginning or at the end of the data

        Parameters
        ----------
        inplace:bool = True
            if True, the operations are made directly in the current object.

        Returns
        -------
        if inplace=False, return the current object with the missing values
        removed at the beginning and at the end.
        """
        signals = [i for i in self.signals if type(i) != EMGSignal]
        if len(signals) == 0:
            signals = self.emgsignals
        time0 = np.min([i.data.dropna().index[0] for i in signals])
        time1 = np.max([i.data.dropna().index[-1] for i in signals])
        self.slice(time0, time1, inplace=True)
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if not inplace:
            return self.copy()

    def reset_index(self, inplace: bool = True):
        """
        ensure the time array starts from zero

        Parameters
        ----------
        inplace : bool (default=True)
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations.

        Returns
        -------
        if inplace=False, return the current object.
        """
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        for signal in self.signals:
            signal.reset_index(inplace=True)
        if not inplace:
            return self.copy()

    def to_dataframe(self):
        """return a pandas DataFrame object containing all the available data"""
        out = []
        for signal in self.signals:
            new = signal.to_dataframe()
            name = type(signal).__name__
            cols = new.columns.to_numpy()
            cols = pd.MultiIndex.from_tuples([(name, *i) for i in new.columns])
            new.columns = cols
            new.columns
            out += [new]

        return pd.concat(out, axis=1)

    def copy(self):
        """create a copy of this object"""
        return deepcopy(self)

    def slice(
        self,
        from_time: int | float | np.number,
        to_time: int | float | np.number,
        inplace: bool = True,
    ):
        """
        return a subset of the StateFrame.

        Parameters
        ----------
        from_time: int | float | np.number
            the returned slice starts from the provided time.

        to_time: int | float | np.number
            the returned slice ends at the provided time.

        inplace: bool = True
            if False a copy of the sliced object is returned.

        Returns
        -------
        slice: StateFrame
            a subset of the original StateFrame with time index within the
            given from_time and to_time.
        """
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean.")
        out = self if inplace else self.copy()
        for signal in out.signals:
            signal.slice(from_time, to_time, inplace=True)
        if not inplace:
            return out

    def change_reference_frame(
        self,
        origin: np.ndarray | list[float | int] = [0, 0, 0],
        axis1: np.ndarray | list[float | int] = [1, 0, 0],
        axis2: np.ndarray | list[float | int] = [0, 1, 0],
        axis3: np.ndarray | list[float | int] = [0, 0, 1],
        inplace: bool = True,
    ):
        """
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

        Parameters
        ----------
        origin: Iterable[int | float] = [0, 0, 0]
            The coordinates of the origin of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis1: Iterable[int | float] | pd.DataFrame | None = [1, 0, 0]
            The coordinates of the first axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis2: Iterable[int | float] | pd.DataFrame | None = [0, 1, 0]
            The coordinates of the second axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        axis3: Iterable[int | float] | pd.DataFrame | None = [0, 1, 0]
            The coordinates of the third axis of the new reference frame.
            It has to be provided as an iterable castable to a 1D numpy array
            with len = 3.

        inplace: bool = True
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations.

        Returns
        -------
        Nothing if 'inplace = True', otherwise a copy of the actual object
        with the rotated axes.

        Note
        ----
        This method uses internally the Gram-Schmidt ortogonalization agorithm
        to derive the orthogonal unit versors from the provided axes.
        Then it align all the available markers and forceplatforms to the
        input origin and finally it rotates them according to the input axes.
        """
        for signal in self._signals:
            if hasattr(signal, "change_reference_frame"):
                signal.change_reference_frame(  # type: ignore
                    origin=origin,
                    axis1=axis1,
                    axis2=axis2,
                    axis3=axis3,
                    inplace=True,
                )
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if not inplace:
            return self.copy()

    def save(self, file_path: str):
        """
        save the test to the input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the test name. If this is not the case, the appropriate extension
            is appended.
        """
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

    # *constructors

    def __init__(
        self,
        *signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
        strip: bool = True,
        reset_index: bool = False,
    ):
        if not isinstance(strip, bool):
            raise ValueError("'strip' must be True or False")
        if not isinstance(reset_index, bool):
            raise ValueError("'reset_index' must be True or False")
        self._signals = []
        for signal in signals:
            self.add(signal)
        if strip:
            self.strip(inplace=True)
        if reset_index:
            self.reset_index(inplace=True)

    @classmethod
    def load(cls, file_path: str):
        """
        load the test data from an input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the test name. If this is not the case, the appropriate extension
            is appended.

        Returns
        -------
        obj: Self
            the test object
        """
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
    def from_tdf_file(cls, file: str, strip: bool = True, reset_index: bool = False):
        """
        generate a StateFrame from a .tdf file

        Parameters
        ----------
        file : str
            a valid .tdf file containing (tracked) markers, force platforms and
            (optionally) EMG data

        strip:bool (default=True)
            strip the data to omit nan values at the beginning and at the end
            of the file

        reset_index:bool (default=True)
            reset the time index such as it will start from zero

        Returns
        -------
        frame: StateFrame
            a state frame instance of the data contained in the .tdf file.
        """
        # check of the input file
        if (
            not isinstance(file, str)
            or not file.lower().endswith(".tdf")
            or not exists(file)
        ):
            raise ValueError("file must be and existing .tdf object")

        # read the tdf file
        tdf = read_tdf(file)
        obj = cls(reset_index=False, strip=False)

        # extract raw marker data
        try:
            markers: pd.DataFrame = tdf["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
            for lbl in markers.columns.get_level_values(0).unique():  # type: ignore
                pnt = Point3D(
                    xarr=markers[lbl]["X"].values.astype(float).flatten(),
                    yarr=markers[lbl]["Y"].values.astype(float).flatten(),
                    zarr=markers[lbl]["Z"].values.astype(float).flatten(),
                    time=markers.index.to_numpy(),
                    name=lbl,
                    strip=False,
                    reset_index=False,
                )
                obj.add(pnt)
        except Exception as exc:
            warn(f"the {file} file does not contain Point3D data.", RuntimeWarning)

        # extract raw forceplatform data
        try:
            forceplatforms: pd.DataFrame = tdf["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]  # type: ignore
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
                    name="origin",
                    strip=False,
                    reset_index=False,
                )
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
                    name="force",
                    unit="N",
                    strip=False,
                    reset_index=False,
                )
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
                    name="torque",
                    unit="Nm",
                    strip=False,
                    reset_index=False,
                )
                obj.add(ForcePlatform(origin, force, torque, lbl, False, False))
        except Exception as exc:
            warn(
                f"the {file} file does not contain ForcePlatform data.", RuntimeWarning
            )

        # extract raw EMG data
        try:
            emgs: pd.DataFrame = tdf["EMG"]["TRACKS"]  # type: ignore
            for muscle in emgs.columns.get_level_values(0).unique():  # type: ignore
                muscle_data = emgs[muscle]
                for side in muscle_data.columns.get_level_values(0).unique():
                    emg = EMGSignal(
                        amplitude=muscle_data[side].values.astype(float),
                        time=muscle_data.index.to_numpy(),
                        name=" ".join([side, muscle]),
                        strip=False,
                        reset_index=False,
                    )
                    obj.add(emg)
        except Exception as exc:
            warn(f"the {file} file does not contain EMGSignal data.", RuntimeWarning)

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
