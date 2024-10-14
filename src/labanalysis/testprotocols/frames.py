"""
frames module containing useful classes for kinetic, kinematic and EMG data
analysis
"""

#! IMPORTS


from os.path import exists
from typing import Any, Iterable
from warnings import warn

import numpy as np
import pandas as pd
from labio import read_tdf
from scipy.spatial.transform import Rotation

from ..signalprocessing import butterworth_filt, fillna, gram_schmidt, rms_filt

__all__ = ["StateFrame"]

#! CLASSES


class StateFrame:
    """
    base class used for reading and perform basic processing of
    kinetic, kinematic and emg data.

    Parameters
    ----------
    markers: pd.DataFrame | None
        a DataFrame being composed by:
            * one or more triplets of columns like:
                | <NAME> | <NAME> | <NAME> |
                |    X   |   Y    |    Z   |
                |    m   |   m    |    m   |
            * the time instant of each sample in seconds as index.

    forceplatforms: pd.DataFrame | None
        a DataFrame being composed by:
            * one or more packs of columns like:
                | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> |
                | ORIGIN | ORIGIN | ORIGIN |  FORCE | FORCE  | FORCE  | TORQUE | TORQUE | TORQUE |
                |    X   |   Y    |    Z   |    X   |   Y    |    Z   |    X   |   Y    |    Z   |
                |    m   |   m    |    m   |    N   |   N    |    N   |    Nm  |   Nm   |   Nm   |
            * the time instant of each sample in seconds as index.

    emgs: pd.DataFrame | None
        a DataFrame being composed by:
            * one or more packs of columns like:
                | <NAME> |
                |    V   |
            * the time instant of each sample in seconds as index.

    Attributes
    ----------
    markers
        the processed kinematic data

    forceplatforms
        the processed force data

    emgs
        the processed EMG data

    emg_processing_options
        the parameters to set the filtering of the EMG signal

    forceplatform_processing_options
        the parameters to set the filtering of the force signal

    marker_processing_options
        the parameters to set the filtering of the kinematic signals

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    to_reference_frame
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
    """

    # *class variables

    _processed: bool
    _markers: pd.DataFrame
    _forceplatforms: pd.DataFrame
    _emgs: pd.DataFrame
    _emg_processing_options: dict[str, Any] | None
    _forceplatform_processing_options: dict[str, Any] | None
    _marker_processing_options: dict[str, Any] | None

    # *attributes

    @property
    def emg_processing_options(self):
        """
        the EMG processing options
        """
        return self._emg_processing_options

    @property
    def forceplatform_processing_options(self):
        """
        the force processing options
        """
        return self._forceplatform_processing_options

    @property
    def marker_processing_options(self):
        """
        the marker processing options
        """
        return self._marker_processing_options

    @property
    def markers(self):
        """
        the processed markers coordinates
        """
        return self._markers

    @property
    def forceplatforms(self):
        """
        the processed force platform data
        """
        return self._forceplatforms

    @property
    def emgs(self):
        """
        the processed EMGs signals
        """
        return self._emgs

    # *methods

    def process_data(
        self,
        ignore_index: bool = True,
        inplace: bool = True,
        markers_fcut: int | float | None = 6,
        forces_fcut: int | float | None = 100,
        emgs_fband: tuple[int | float, int | float] | None = (30, 400),
        emgs_rms_win: int | float | None = 0.2,
    ):
        """
        process the available data according to the given options

        Parameters
        ----------
        ignore_index: bool = True
            if True the reduced data are reindexed such as they start from zero

        inplace: bool = True
            if True, the operations are made directly in the current object.
            Otherwise a copy is created and returned at the end of the
            operations

        markers_fcut:  int | float | None = 6
            cut frequency of the lowpass, 4th order, phase-corrected,
            Butterworth filter (in Hz) used to smooth the provided coordinates.

        forces_fcut: int | float | None = 100
            cut frequency of the lowpass, 4th order, phase-corrected,
            Butterworth filter (in Hz) used to smooth the provided force and
            torque data.

        emgs_fband: tuple[int | float, int | float] | None = (30, 400)
            frequency limits of the bandpass, 4th order, phase-corrected,
            Butterworth filter (in Hz) used to smooth the provided EMG data.

        emgs_rms_win: int | float | None = 0.2
            the Root Mean Square window (in seconds) used to create the EMG
            envelopes.

        Processing procedure
        --------------------

        Markers
            1. missing values at the beginning and end of the data are removed
            2. missing values in the middle of the trial are replaced by cubic
            spline interpolation
            3. the data are low-pass filtered by means of a lowpass, Butterworth
            filter with the entered marker options

        Force Platforms
            1. the data in between the start and end of the marker data are retained.
            2. missing values in the middle of the data are replaced by zeros
            3. Force and Torque data are low-pass filtered by means of a lowpass,
            Butterworth filter with the entered force options.
            4. Force vector origin's coordinates are low-pass filtered by means of
            a lowpass, Butterworth filter with the entered marker options.

        EMGs (optional)
            1. the data in between the start and end of the markers are retained.
            2. the signals are bandpass filtered with the provided emg options
            3. the root-mean square filter with the given time window is applied to
            get the envelope of the signals.

        All
            1. if 'ignore_index=True' then the time indices of all components is
            adjusted to begin with zero.
        """
        # check for the start and stop of the data
        if self._markers.shape[0] > 0:
            valid = self._markers.notna().all(axis=1).values.astype(bool)
            time = self._markers.index.to_numpy()

        elif self._forceplatforms.shape[0] > 0:
            valid = self._forceplatforms.notna().all(axis=1).values.astype(bool)
            time = self._forceplatforms.index.to_numpy()

        elif self._emgs.shape[0] > 0:
            valid = self._emgs.notna().all(axis=1).values.astype(bool)
            time = self._emgs.index.to_numpy()

        else:
            valid = np.array([])
            time = np.array([])

        # resize all data
        index0, index1 = np.where(valid)[0][[0, -1]]
        time0, time1 = time[[index0, index1]]

        if self._markers.shape[0] > 0:
            time = self._markers.index.to_numpy()
            index = np.where((time >= time0) & (time <= time1))[0]
            markers = self._markers.iloc[index]
        else:
            markers = pd.DataFrame()

        if self._forceplatforms.shape[0] > 0:
            time = self._forceplatforms.index.to_numpy()
            index = np.where((time >= time0) & (time <= time1))[0]
            fps = self._forceplatforms.iloc[index]
        else:
            fps = pd.DataFrame()

        if self._emgs.shape[0] > 0:
            time = self._emgs.index.to_numpy()
            index = np.where((time >= time0) & (time <= time1))[0]
            emgs = self._emgs.iloc[index]
        else:
            emgs = pd.DataFrame()

        # fill nans
        if markers.shape[0] > 0:
            markers = pd.DataFrame(fillna(markers))  # cubic spline interpolation

        if fps.shape[0] > 0:
            fcols = [i for i, v in enumerate(fps.columns) if v[1] != "ORIGIN"]
            fps.iloc[:, fcols] = fillna(fps.iloc[:, fcols], 0)  # zeros  # type: ignore
            pcols = [i for i, v in enumerate(fps.columns) if v[1] == "ORIGIN"]
            fps.iloc[:, pcols] = fillna(
                fps.iloc[:, pcols]
            )  # cubic spline interpolation    # type: ignore
            fps = pd.DataFrame(fps)

        if emgs.shape[0] > 0:
            emgs = pd.DataFrame(fillna(emgs))  # cubic spline interpolation

        # check index
        if not isinstance(ignore_index, bool):
            raise ValueError("ignore_index must be a boolean")

        if ignore_index:
            if markers.shape[0] > 0:
                markers.index -= time0
            if fps.shape[0] > 0:
                fps.index -= time0
            if emgs.shape[0] > 0:
                emgs.index -= time0

        # check markers filtering
        if markers_fcut is None or markers.shape[0] == 0:
            self._marker_processing_options = None
        else:
            mfsamp = float(1 / np.mean(np.diff(self._markers.index.to_numpy())))
            try:
                self._marker_processing_options = {
                    "fcut": float(markers_fcut),
                    "order": 4,
                    "phase_corrected": True,
                    "fsamp": mfsamp,
                    "ftype": "lowpass",
                }
            except Exception as exc:
                msg = "markers_fcut must be castable to float"
                raise ValueError(msg) from exc

            # apply the filter
            if markers.shape[0] > 0:
                markers = markers.apply(
                    butterworth_filt,
                    raw=True,
                    **self._marker_processing_options,
                )

        # check force platforms filtering
        if forces_fcut is None or fps.shape[0] == 0:
            self._forceplatform_processing_options = None
        else:
            ffsamp = np.mean(np.diff(self._forceplatforms.index.to_numpy()))
            ffsamp = float(1 / ffsamp)
            try:
                self._forceplatform_processing_options = {
                    "fcut": float(forces_fcut),
                    "order": 4,
                    "phase_corrected": True,
                    "fsamp": ffsamp,
                    "ftype": "lowpass",
                }
            except Exception as exc:
                msg = "forces_fcut must be castable to float"
                raise ValueError(msg) from exc

            # apply the filter
            if fps.shape[0] > 0:
                fps = fps.apply(
                    butterworth_filt,
                    raw=True,
                    **self._forceplatform_processing_options,
                )

        # check force EMG filtering
        if emgs_fband is None or emgs_rms_win is None or emgs.shape[0] == 0:
            self._emg_processing_options = None
        else:

            # check input bandpass filter cutoffs
            msg = "emgs_fband must be a len=2 iterable of float-like objects"
            if not isinstance(emgs_fband, Iterable) or len(emgs_fband) != 2:
                raise ValueError(msg) from exc
            try:
                band = (float(emgs_fband[0]), float(emgs_fband[1]))
            except Exception as exc:
                raise ValueError(msg) from exc

            # check for rms window
            emg_fsamp = float(1 / np.mean(np.diff(self._emgs.index.to_numpy())))
            try:
                rms_win = float(emgs_rms_win)
            except Exception as exc:
                msg = "emgs_rms_win must be castable to float"
                raise ValueError(msg) from exc

            self._emg_processing_options = {
                "fcut": band,
                "order": 4,
                "phase_corrected": True,
                "fsamp": emg_fsamp,
                "ftype": "bandpass",
                "rms_window": rms_win,
            }

            # apply the bandpass filter
            bp_options = self.emg_processing_options.copy()  # type: ignore
            bp_options.pop("rms_window")
            emgs = emgs.apply(
                butterworth_filt,
                raw=True,
                **bp_options,
            )

            # apply the rms envelope
            rms_dt = int(round(rms_win * emg_fsamp))
            emgs = emgs.apply(rms_filt, raw=True, order=rms_dt)

        # check for the 'inplace' input
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")

        if inplace:
            self._processed = True
            self._markers = pd.DataFrame(markers)
            self._forceplatforms = pd.DataFrame(fps)
            self._emgs = pd.DataFrame(emgs)
        else:
            out = self.copy()
            out._processed = True
            out._markers = pd.DataFrame(markers)
            out._forceplatforms = pd.DataFrame(fps)
            out._emgs = pd.DataFrame(emgs)

            return out

    def to_dataframe(self):
        """return a pandas DataFrame object containing all the available data"""

        # prepare the markers
        markers = self.markers.copy()
        markers_cols = markers.columns.to_frame()
        markers_cols.insert(1, "SOURCE", "COORDINATE")
        markers_cols.insert(0, "TYPE", "MARKER")
        markers.columns = pd.MultiIndex.from_frame(markers_cols)

        # prepare the force platforms
        fps = self.forceplatforms.copy()
        fps_cols = fps.columns.to_frame()
        fps_cols.insert(0, "TYPE", "FORCE_PLATFORM")
        fps.columns = pd.MultiIndex.from_frame(fps_cols)

        # prepare the emg data
        emgs = self.emgs.copy()
        emgs_cols = emgs.columns.to_frame()
        emgs_cols.insert(1, "SOURCE", "CHANNEL")
        emgs_cols.insert(0, "TYPE", "EMG")
        emgs.columns = pd.MultiIndex.from_frame(emgs_cols)

        # return
        return pd.concat([markers, fps, emgs], axis=1)

    def copy(self):
        """create a copy of this object"""
        obj = StateFrame(
            markers_raw=self.markers,
            forceplatforms_raw=self.forceplatforms,
            emgs_raw=self.emgs,
        )
        obj._processed = self.is_processed()
        obj._marker_processing_options = self.marker_processing_options
        obj._forceplatform_processing_options = self.forceplatform_processing_options
        obj._emg_processing_options = self.emg_processing_options
        return obj

    def is_processed(self):
        """
        returns True if the actual object already run the process data method
        """
        return self._processed

    def slice(
        self,
        from_time: int | float | np.number,
        to_time: int | float | np.number,
    ):
        """
        return a subset of the StateFrame.

        Parameters
        ----------
        from_time: int | float | np.number
            the returned slice starts from the provided time.

        to_time: int | float | np.number
            the returned slice ends at the provided time.

        Returns
        -------
        slice: StateFrame
            a subset of the original StateFrame with time index within the
            given from_time and to_time.
        """
        out = self.copy()

        # slice the markers
        markers_mask = out._markers.index >= from_time
        markers_mask &= out._markers.index <= to_time
        out._markers = out._markers.loc[markers_mask]

        # slice the force platform data
        fps_mask = out._forceplatforms.index >= from_time
        fps_mask &= out._forceplatforms.index <= to_time
        out._forceplatforms = out._forceplatforms.loc[fps_mask]

        # slice the emgs
        emg_mask = out._emgs.index >= from_time
        emg_mask &= out._emgs.index <= to_time
        out._emgs = out._emgs.loc[emg_mask]

        return out

    def to_reference_frame(
        self,
        origin: Iterable[int | float] | pd.DataFrame = [0, 0, 0],
        axis1: Iterable[int | float] | pd.DataFrame | None = [1, 0, 0],
        axis2: Iterable[int | float] | pd.DataFrame | None = [0, 1, 0],
        axis3: Iterable[int | float] | pd.DataFrame | None = [0, 0, 1],
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
        # check inputs
        ori = self._validate_array(origin)
        ax1 = self._validate_array(axis1)
        ax2 = self._validate_array(axis2)
        ax3 = self._validate_array(axis3)

        # create the rotation matrix
        rmat = Rotation.from_matrix(gram_schmidt(ax1, ax2, ax3))

        # apply to markers
        if self._markers.shape[0] > 0:
            markers = self.markers.copy()
            ones = np.ones((markers.shape[0], 3))
            idx = markers.index
            for lbl in np.unique(markers.columns.get_level_values(0)):
                coords = [i for i in markers.columns if i[0] == lbl]
                markers.loc[idx, coords] -= ones * ori
                vals = markers.loc[idx, coords].values
                markers.loc[idx, coords] = rmat.apply(vals)
        else:
            markers = self._markers

        # apply to forceplatforms
        if self._forceplatforms.shape[0] > 0:
            fps = self._forceplatforms.copy()
            ones = np.ones((fps.shape[0], 3))
            idx = fps.index
            for lbls in np.unique(fps.columns.get_level_values(0)):
                dfr = fps[[i for i in fps.columns if i[0] == lbls]]

                # marker coordinates
                coords = [i for i in dfr.columns if i[1] == "ORIGIN"]
                fps.loc[idx, coords] -= ones * ori
                fps.loc[idx, coords] = rmat.apply(fps.loc[idx, coords].values)

                # forces
                forces = [i for i in dfr.columns if i[1] == "FORCE"]
                fps.loc[idx, forces] = rmat.apply(fps.loc[idx, forces].values)

                # torque
                torques = [i for i in dfr.columns if i[1] == "TORQUE"]
                fps.loc[idx, torques] = rmat.apply(fps.loc[idx, torques].values)
        else:
            fps = self._forceplatforms

        # handle the inplace input
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' must be a boolean")
        if inplace:
            self._markers = markers
            self._forceplatforms = fps
        else:
            obj = self.copy()
            obj._markers = markers
            obj._forceplatforms = fps
            return obj

    def _validate_array(self, obj: object):
        """
        private method used to check if obj is a numeric 1D array with len = 3

        Parameters
        ----------
        obj: object
            the object to be checked.
        """
        msg = "The object must be castable to a numpy 1D array with 3 elements"
        try:
            new = np.array([obj]).astype(float).flatten()
        except Exception as exc:
            raise ValueError(msg) from exc
        if len(new) != 3:
            raise ValueError(msg)
        return new

    def _validate_frame(self, obj: object):
        """
        private method used to check if obj is a DataFrame with MultiIndex
        columns

        Parameters
        ----------
        obj: object
            the object to be checked.
        """
        # check if obj is a dataframe
        if not isinstance(obj, pd.DataFrame):
            raise ValueError("obj must be a DataFrame instance")

        # check if obj has multiindex columns
        if obj.shape[0] > 0:
            cols_msg = "obj.columns must be a MultiIndex instance with 3 levels"
            if not isinstance(obj.columns, pd.MultiIndex):
                raise ValueError(cols_msg)

    def _validate_triplet(self, arr: list[tuple], unit: str):
        """
        private method used to validate input array

        Parameters
        ----------
        arr: list[tuple[str, str, str]]
            a list of 3 tuples where each of them should be of len = 3.

        unit: str
            the expected unit of measurement

        Note
        ----
        this method checks if the second element of each tuple in list is any of
        'X', 'Y' or 'Z'. Then it checks that only one unit of measurement is
        declated as third element of each tuple and that this unit corresponds
        to the expected one.
        """
        msg = "obj.columns must be a MultiIndex instance with 3 levels"
        if len(arr) != 3 or len(arr[0]) != 3:
            raise ValueError(msg)

        # check the unit of measurement
        units = np.unique([i[2] for i in arr])
        if len(units) != 1 or units[0] != unit:
            raise ValueError(f"{units} were found instead of '{unit}'")

        # check that the X, Y and Z axes exist for each marker
        for lbl in np.unique([i[0] for i in arr]):
            axes = sorted([i[1] for i in arr if i[0] == lbl])
            if len(axes) != 3 or axes[0] != "X" or axes[1] != "Y" or axes[2] != "Z":
                raise ValueError(f"{lbl} does not contain X, Y, Z axes.")

    def _validate_markers(self, obj: object):
        """
        check if the provided object is suitable to be considered a
        markers dataframe. Raise an error otherwise

        Parameters
        ----------
        obj: object
            the object to be checked

        Note
        ----
        A valid markers dataframe has a layout like:
            * one or more triplets of columns like:
                | <NAME> | <NAME> | <NAME> |
                |    X   |   Y    |    Z   |
                |    m   |   m    |    m   |
            * the time instant of each sample in seconds as index.
        """
        # check if obj is a dataframe with multiindex columns
        self._validate_frame(obj)

        # check if the columns layout is appropriate
        if obj.shape[0] > 0:  # type: ignore
            self._validate_triplet(obj.columns.to_list(), "m")  # type: ignore

    def _validate_forceplatforms(self, obj: object):
        """
        check if the provided object is suitable to be considered a
        forceplatforms dataframe. Raise an error otherwise

        Parameters
        ----------
        obj: object
            the object to be checked

        Note
        ----
        A valid forceplatforms dataframe has a layout like:
            * one or more packs of columns like:
                | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> |
                | ORIGIN | ORIGIN | ORIGIN |  FORCE | FORCE  | FORCE  | TORQUE | TORQUE | TORQUE |
                |    X   |   Y    |    Z   |    X   |   Y    |    Z   |    X   |   Y    |    Z   |
                |    m   |   m    |    m   |    N   |   N    |    N   |    Nm  |   Nm   |   Nm   |
            * the time instant of each sample in seconds as index.
        """
        # check if obj is a dataframe with multiindex columns
        self._validate_frame(obj)

        # check if the columns layout is appropriate
        if obj.shape[0] > 0:  # type: ignore
            arr = obj.columns.to_list()  # type: ignore
            lbls = np.unique([i[0] for i in arr])
            for lbl in lbls:
                for dom, unt in zip(["ORIGIN", "FORCE", "TORQUE"], ["m", "N", "Nm"]):
                    arr = obj[lbl][[dom]].columns.tolist()  # type: ignore
                    self._validate_triplet(arr, unt)

    def _validate_emgs(self, obj: object):
        """
        check if the provided object is suitable to be considered an
        EMG dataframe. Raise an error otherwise

        Parameters
        ----------
        obj: object
            the object to be checked

        Note
        ----
        A valid markers dataframe has a layout like:
            * one or more triplets of columns like:
                | <NAME> |
                |    V   |
            * the time instant of each sample in seconds as index.
        """
        # check if obj is a dataframe with multiindex columns
        self._validate_frame(obj)

        # check if the columns have 2 or more levels
        if obj.shape[0] > 0:  # type: ignore
            cols = obj.columns.to_list()  # type: ignore
            if len(cols[0]) < 2:
                raise ValueError("obj must have 2 or more levels as columns")

            # check the unit of measurement
            unit = np.unique([i[-1] for i in cols])
            if len(unit) != 1 and unit[0][-1] != "V":
                raise ValueError("obj must be measured in multiple of Volts")

    def _check_processed(self):
        """private method used to check if processed output are available"""
        if not self.is_processed():
            msg = "The current object has not been processed."
            msg += "Please call the '<object>.process_data(*args, **kwargs)'."
            warn(msg, category=UserWarning)

    def _get_muscle_name(self, raw_muscle_name: str):
        """
        private method used understand the muscle side according to its name

        Parameters
        ----------
        raw_muscle_name: str
            the raw muscle name

        Returns
        -------
        name: tuple[str, str | None]
            the muscle name divided as (<NAME>, <SIDE>). If proper side is not
            found (e.g. for Rectus Abdominis), the <SIDE> term is None.
        """
        # split the raw muscle name in words
        splits = raw_muscle_name.lower().split(" ")

        # get the index of the word denoting the side
        side_idx = [i for i, v in enumerate(splits) if v in ["left", "right"]]
        side_idx = None if len(side_idx) == 0 else side_idx[0]

        # adjust the muscle name
        side = None if side_idx is None else splits.pop(side_idx)
        muscle = "_".join(splits[:2])

        # return the tuple
        return (muscle, side)

    # *constructors

    def __init__(
        self,
        markers_raw: pd.DataFrame | None = None,
        forceplatforms_raw: pd.DataFrame | None = None,
        emgs_raw: pd.DataFrame | None = None,
    ):
        """
        generate a new StateFrame object

        Parameters
        ----------
        markers_raw: pd.DataFrame | None = None
            a dataframe containing raw markers data. If None, an empty dataframe
            is used

        forceplatforms_raw: pd.DataFrame | None = None
            a raw dataframe containing raw forceplatforms data.
            If None, an empty dataframe is used

        emgs_raw: pd.DataFrame | None = None
            a raw dataframe containing raw emg data.
            If None, an empty dataframe is used
        """
        # check and initialize markers data
        if markers_raw is None:
            markers_raw = pd.DataFrame()
        self._validate_markers(markers_raw)
        self._markers = markers_raw

        # check and initialize force platforms data
        if forceplatforms_raw is None:
            forceplatforms_raw = pd.DataFrame()
        self._validate_forceplatforms(forceplatforms_raw)
        self._forceplatforms = forceplatforms_raw

        # check and initialize EMG data
        if emgs_raw is None:
            emgs_raw = pd.DataFrame()
        self._validate_emgs(emgs_raw)
        self._emgs = emgs_raw

        # separate the EMG data by side (where possible)
        if len(self._emgs.columns[0]) < 3:
            raw_names = self._emgs.columns.get_level_values(0)
            muscles = [self._get_muscle_name(i) + ("V",) for i in raw_names]
            self._emgs.columns = pd.MultiIndex.from_tuples(muscles)

        # set options to None
        self._marker_processing_options = None
        self._forceplatform_processing_options = None
        self._emg_processing_options = None
        self._processed = False

    @classmethod
    def from_tdf_file(cls, file: str):
        """
        generate a StateFrame from a .tdf file

        Parameters
        ----------
        file : str
            a valid .tdf file containing (tracked) markers, force platforms and
            (optionally) EMG data

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

        # extract raw marker data
        try:
            markers_raw: pd.DataFrame = tdf["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
        except Exception as exc:
            markers_raw = pd.DataFrame()
            msg = "the provided .tdf file does not contain marker data."
            warn(msg, category=UserWarning)

        # extract raw forceplatform data
        try:
            forceplatforms_raw = tdf["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]  # type: ignore
        except Exception as exc:
            forceplatforms_raw = pd.DataFrame()
            msg = "the provided .tdf file does not contain force platform data."
            warn(msg, category=UserWarning)

        # extract raw EMG data
        # controllo sia disponibile il segnale emg
        try:
            emgs_raw: pd.DataFrame = tdf["EMG"]["TRACKS"]  # type: ignore
        except Exception as exc:
            emgs_raw = pd.DataFrame()
            msg = "the provided .tdf file does not contain EMG data."
            warn(msg, category=UserWarning)

        # generate a new object
        return cls(markers_raw, forceplatforms_raw, emgs_raw)
