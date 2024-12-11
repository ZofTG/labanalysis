"""kinematics module"""

#! IMPORTS


from os.path import exists
from typing import Literal

import numpy as np
import pandas as pd
from labio import read_tdf
from plotly.subplots import make_subplots

from .. import signalprocessing as labsp

#! CONSTANTS


__all__ = [
    "GaitTest",
    "RunningStep",
    "WalkingStride",
]


#! CLASSESS


class GaitCycle:
    """basic gait class to be properly implemented"""

    def as_dict(self):
        """return the object as dict"""
        keys = [i for i in dir(self) if i[0] != "_" and i[:2] != "as"]
        out = {i: getattr(self, i) for i in keys}
        return {i: float(v) if i != "side" else str(v) for i, v in out.items()}

    def __repr__(self):
        return pd.Series(self.as_dict()).__repr__()

    def __str__(self):
        return pd.Series(self.as_dict()).__str__()


class RunningStep(GaitCycle):
    """
    basic running step class.

    Parameters
    ----------
    footstrike_s: int | float
        the foot-strike time in seconds

    midstance_s: int | float
        the mid-stance time in seconds

    toeoff_s: int | float
        the toe-off time in seconds

    ending_s: int | float
        the next foot-strike time in seconds

    side: Literal['RIGHT', 'LEFT']
        the side of the test

    Attributes
    ----------
    step_time_s: float
        the step time in seconds

    contact_time_s: float
        the contact time in seconds

    loading_response_time_s: float
        the loading response time in seconds

    propulsion_time_s: float
        the propulsive time in seconds

    flight_time_s: float
        the flight time in seconds

    cadence_spm: float
        the pace of the step in steps per minute

    Note
    ----
    the step starts from the toeoff and ends at the next toeoff of the
    contralateral foot.
    """

    _fs: float
    _ms: float
    _to: float
    _ed: float
    _side: Literal["RIGHT", "LEFT"] | None

    # * properties

    @property
    def footstrike_s(self):
        """return the foot-strike time in seconds"""
        return self._fs

    @property
    def midstance_s(self):
        """return the mid-stance time in seconds"""
        return self._ms

    @property
    def toeoff_s(self):
        """return the toe-off time in seconds"""
        return self._to

    @property
    def ending_s(self):
        """return the landing time in seconds"""
        return self._ed

    @property
    def side(self):
        """return the side of the stride"""
        return self._side

    @property
    def step_time_s(self):
        """return the stride time in seconds"""
        return self.ending_s - self.toeoff_s

    @property
    def contact_time_s(self):
        """return the contact time in seconds"""
        return self.ending_s - self.footstrike_s

    @property
    def flight_time_s(self):
        """return the flight time in seconds"""
        return self.footstrike_s - self.toeoff_s

    @property
    def loading_response_time_s(self):
        """return the loading response time in seconds"""
        return self.midstance_s - self.footstrike_s

    @property
    def propulsion_time_s(self):
        """return the propulsion time in seconds"""
        return self.ending_s - self.midstance_s

    @property
    def cadence_spm(self):
        """return the cadence of the stride in strides per minute"""
        return 60 / self.step_time_s

    # * constructor

    def __init__(
        self,
        footstrike_s: int | float,
        midstance_s: int | float,
        toeoff_s: int | float,
        ending_s: int | float,
        side: Literal["RIGHT", "LEFT"] | None = None,
    ):
        super().__init__()
        self._fs = float(footstrike_s)
        self._ms = float(midstance_s)
        self._to = float(toeoff_s)
        self._ed = float(ending_s)
        self._side = side


class WalkingStride(GaitCycle):
    """
    basic walking stride class.

    Parameters
    ----------
    ipsilateral_toeoff_s: int | float
        the toe-off time in seconds of the contralateral foot defining the start
        of the step

    ipsilateral_footstrike_s: int | float
        the foot-strike time in seconds of the foot setting the start of the
        stance phase

    contralateral_toeoff_s: int | float
        the toeoff of the contralateral foot. This sets the end of the
        double-support phase

    ending_s: int | float
        the toeoff time in seconds of the ipsilateral foot setting the end
        of the step

    side: Literal['RIGHT', 'LEFT']
        the side of the test

    Attributes
    ----------
    stride_time_s: float
        the stride time in seconds

    stance_time_s: float
        the stance time in seconds

    swing_time_s: float
        the swing time in seconds

    single_support_time_s: float
        the single support time in seconds

    double_support_time_s: float
        the double support time in seconds

    step_time_s: float
        the time lapse between one toe-off and the one of the contralateral
        step in seconds.

    Note
    ----
    the stride starts from the toe-off of the ipsilateral foot.
    """

    _ipsilateral_footstrike_s: float
    _ipsilateral_toeoff_s: float
    _contralateral_footstrike_s: float
    _contralateral_toeoff_s: float
    _ending_s: float
    _side: Literal["RIGHT", "LEFT"] | None

    # * properties

    @property
    def ipsilateral_footstrike_s(self):
        """
        return the foot-strike time of the foot performing the actual stride
        in seconds
        """
        return self._ipsilateral_footstrike_s

    @property
    def contralateral_footstrike_s(self):
        """
        return the foot-strike time of the foot not performing the actual stride
        in seconds
        """
        return self._contralateral_footstrike_s

    @property
    def ipsilateral_toeoff_s(self):
        """
        return the toe-off time of the foot performing the actual stride
        in seconds
        """
        return self._ipsilateral_toeoff_s

    @property
    def contralateral_toeoff_s(self):
        """
        return the toe-off time of the foot not performing the actual stride
        in seconds
        """
        return self._contralateral_toeoff_s

    @property
    def ending_s(self):
        """
        return the ending toe-off (i.e. the second ipsilateral toeoff time)
        in seconds
        """
        return self._ending_s

    @property
    def side(self):
        """return the side of the stride"""
        return self._side

    @property
    def stride_time_s(self):
        """return the stride time in seconds"""
        return self.ending_s - self.ipsilateral_toeoff_s

    @property
    def step_time_s(self):
        """return the step time in seconds"""
        return self.contralateral_toeoff_s - self.ipsilateral_toeoff_s

    @property
    def stance_time_s(self):
        """return the stance time in seconds"""
        return self.ending_s - self.ipsilateral_footstrike_s

    @property
    def swing_time_s(self):
        """return the swing time in seconds"""
        return self.ipsilateral_footstrike_s - self.ipsilateral_toeoff_s

    @property
    def single_support_time_s(self):
        """return the single support time in seconds"""
        return self.contralateral_toeoff_s - self.contralateral_footstrike_s

    @property
    def double_support_time_s(self):
        """return the double support time in seconds"""
        return self.stride_time_s - self.single_support_time_s

    # * constructor

    def __init__(
        self,
        ipsilateral_toeoff_s: int | float,
        ipsilateral_footstrike_s: int | float,
        contralateral_toeoff_s: int | float,
        contralateral_footstrike_s: int | float,
        ending_s: int | float,
        side: Literal["RIGHT", "LEFT"] | None = None,
    ):
        super().__init__()
        self._ipsilateral_footstrike_s = ipsilateral_footstrike_s
        self._ipsilateral_toeoff_s = ipsilateral_toeoff_s
        self._contralateral_footstrike_s = contralateral_footstrike_s
        self._contralateral_toeoff_s = contralateral_toeoff_s
        self._ending_s = ending_s
        self._side = side


class GaitTest:
    """
    detect steps and strides from kinematic data and extract biofeedback
    info

    Parameters
    ----------
    grf: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, N), (Y, N), (Z, N)]) denoting the
        ground rection forces applied to the centre of pressure where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    cop: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of the centre of pressure where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    lheel: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of the left heel where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    ltoe: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of the left toe where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    rheel: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of the right heel where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    rtoe: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of the right where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    lmidfoot: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of the fifth metatarsal head of the left foot,
        where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    rmidfoot: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of fifth metatarsal head of the right foot, where:
            X = latero-lateral axis (right direction is positive);
            Y = anterior-posterio axis (forward direction is positive);
            Z = vertical axis (upward direction is positive);

    preprocess: bool = True,
        if True, the provided data is assumed to be "raw" and the following
        processing is performed:
            - gap filling via linear regression of the known coordinates
            - lowpass filtering via a Butterworth, 6th order, phase-corrected
                filter with cut-off frequency of 12 Hz.

    height_thresh: float | int = 0.02,
        the minimum threshold in meters to be considered for assuming that
        one foot is in contact to the ground.

    force_thresh: float | int = 30,
        the minimum force (in N) to be considered for assuming that one foot
        is in contact to the ground.
    """

    _raw_coordinates: pd.DataFrame | None
    _raw_grf: pd.DataFrame | None
    _raw_cop: pd.DataFrame | None
    _preprocessed: bool
    _coordinates: pd.DataFrame | None
    _cop: pd.DataFrame | None
    _grf: pd.DataFrame | None
    _height_threshold: int | float
    _force_threshold: int | float
    _source_file: str | None
    _strides: list[RunningStep | WalkingStride]
    _vertical_axis: Literal["X", "Y", "Z"]

    @property
    def vertical_axis(self):
        """return the vertical axis name"""
        return self._vertical_axis

    @property
    def raw_coordinates(self):
        """the input marker coordinates"""
        return self._raw_coordinates

    @property
    def coordinates(self):
        """the marker coordinates"""
        return self._coordinates

    @property
    def raw_cop(self):
        """the input cop coordinates"""
        return self._raw_cop

    @property
    def cop(self):
        """the cop coordinates"""
        return self._cop

    @property
    def raw_grf(self):
        """the input ground reaction forces"""
        return self._raw_grf

    @property
    def grf(self):
        """the ground reaction forces"""
        return self._grf

    @property
    def preprocessed(self):
        """
        has the raw data being preprocessed before fininding
        steps and strides?
        """
        return self._preprocessed

    @property
    def height_threshold(self):
        """the minimum threshold in meters to be considered for assuming that
        one foot is in contact to the ground."""
        return self._height_threshold

    @property
    def force_threshold(self):
        """the minimum threshold in N to be considered for assuming that
        one foot is in contact to the ground."""
        return self._force_threshold

    @property
    def source_file(self):
        "return the tdf file containing the data of the test"
        return self._source_file

    @property
    def cycles(self):
        """the detected gait cycles"""
        return self._cycles

    # * methods

    def summary(self):
        """return a summary of the collected steps"""

        # get the cycles data
        out = [pd.DataFrame(pd.Series(i.as_dict())).T for i in self.cycles]
        out = pd.concat(out, ignore_index=True).reset_index(drop=True)

        return out

    def _valid_dataframe(
        self,
        obj: object,
    ):
        """
        private method used to check the input data

        Parameters
        ----------
        obj: object
            the object to be checked

        Returns
        -------
        valid: bool
            True if the object is a DataFrame with 3 columns with XYZ labels.
            False otherwise.
        """
        if obj is None:
            return False
        if not isinstance(obj, pd.DataFrame):
            return False
        if obj.shape[1] != 3:
            return False
        labels = [i[0] for i in obj.columns.tolist()]
        if not all(i in labels for i in ["X", "Y", "Z"]):
            return False
        return True

    def _apply_preprocessing(
        self,
        frame: pd.DataFrame,
        filling_value: float | int | None = None,
        cutoff_freq: float | int = 12,
    ):
        """
        private method used to preprocess data

        Parameters
        ----------
        frame: pd.DataFrame
            the dataframe to be processed

        filling_value: float | int | None (optional, default = None)
            if provided, it replaces the missing values before the application
            of lowpass filtering. Otherwise, linear regression is adopted
            to estimate the missing values.

        cutoff_freq: float | int (optional, default = 12)
            the cutoff frequency to be used for smoothing the data.

        Returns
        -------
        processed: pd.DataFrame
            the data frame after the removal of missing values and the
            application of a lowpass filter.
        """

        # remove missing values at the beginning and at the end of the frame
        all_nans_mask = frame.isna().all(axis=1).values.astype(bool).flatten()
        batches = labsp.continuous_batches(all_nans_mask)
        if len(batches) > 0 and batches[0][0] == 0:
            idx_start = batches[0][-1] + 1
        else:
            idx_start = 0
        if len(batches) > 0 and batches[-1][-1] == len(all_nans_mask) - 1:
            idx_stop = batches[-1][0]
        else:
            idx_stop = len(all_nans_mask)
        idx = np.arange(idx_start, idx_stop)

        # fill missing values
        out = pd.DataFrame(labsp.fillna(frame.iloc[idx, :], filling_value, 6))

        # smooth all marker coordinates
        return out.apply(
            labsp.butterworth_filt,  # type: ignore
            fcut=cutoff_freq,
            fsamp=1 / np.mean(np.diff(frame.index.to_numpy())),
            order=6,
            ftype="lowpass",
            phase_corrected=True,
            raw=True,
        )

    def _find_cycles_from_grf(self):
        """find steps via grf coordinates"""
        if self.grf is not None and self.cop is not None:
            # TODO
            pass

    def _find_cycles_from_markers(self):
        """find steps via markers coordinates"""
        if self.coordinates is not None:

            # get the vertical coordinates of all relevant markers
            vcoords = self.coordinates.columns
            vcoords = [i for i in vcoords if i[1] == self.vertical_axis]
            vcoords = self.coordinates[vcoords].copy()
            vcoords.columns = pd.Index([i[0] for i in vcoords.columns])
            vlc = vcoords[[i for i in vcoords.columns if i[0] == "l"]]
            vlc.columns = pd.Index([i[1:] for i in vlc.columns])
            vrc = vcoords[[i for i in vcoords.columns if i[0] == "r"]]
            vrc.columns = pd.Index([i[1:] for i in vrc.columns])

            # get the instants where heels and toes are on ground
            vlc -= vlc.min(axis=0)
            vrc -= vrc.min(axis=0)

            # create a function to extract the contact phases on a given
            # marker
            def get_contact_phases(series: pd.Series):
                arr = series.values.astype(float).flatten()
                time = series.index.to_numpy()
                fsamp = float(1 / np.mean(np.diff(time)))
                frq, pwr = labsp.psd(arr, fsamp)
                ffrq = frq[np.argmax(pwr)]
                cycle_time = 1 / ffrq
                dsamples = int(cycle_time * fsamp * 0.1)
                condition = arr < self.height_threshold
                batches = labsp.continuous_batches(condition)
                i = 1
                while i < len(batches):
                    if batches[i][0] - batches[i - 1][-1] <= dsamples:
                        batches[i - 1] = batches[i - 1] + batches[i]
                        batches.pop(i)
                    else:
                        i += 1
                return [i for i in batches if len(i) > dsamples]

            # get the left foot-strikes
            flc = vlc[[i for i in ["Heel", "Mid"] if i in vlc.columns.tolist()]]
            flc = flc.min(axis=1)
            fsl = [i[0] for i in get_contact_phases(flc) if i[0] > 0]
            fsl = pd.DataFrame(fsl, columns=["Value"])
            fsl.insert(0, "Side", np.tile("LEFT", fsl.shape[0]))
            fsl.insert(0, "Source", np.tile("FS", fsl.shape[0]))

            # get the right foot-strikes
            frc = vrc[[i for i in ["Heel", "Mid"] if i in vrc.columns.tolist()]]
            frc = frc.min(axis=1)
            fsr = [i[0] for i in get_contact_phases(frc) if i[0] > 0]
            fsr = pd.DataFrame(fsr, columns=["Value"])
            fsr.insert(0, "Side", np.tile("RIGHT", fsr.shape[0]))
            fsr.insert(0, "Source", np.tile("FS", fsr.shape[0]))

            # get the left toe-offs
            tol = [i[-1] for i in get_contact_phases(vlc["Toe"])]
            tol = pd.DataFrame(tol, columns=["Value"])
            tol.insert(0, "Side", np.tile("LEFT", tol.shape[0]))
            tol.insert(0, "Source", np.tile("TO", tol.shape[0]))

            # get the right toe-offs
            tor = [i[-1] for i in get_contact_phases(vrc["Toe"])]
            tor = pd.DataFrame(tor, columns=["Value"])
            tor.insert(0, "Side", np.tile("RIGHT", tor.shape[0]))
            tor.insert(0, "Source", np.tile("TO", tor.shape[0]))

            # aggregate the detected indices and sort them
            events = pd.concat([fsl, fsr, tor, tol], ignore_index=True)
            events.sort_values("Value", inplace=True)
            events.reset_index(inplace=True, drop=True)

            # get the mean values between all markers of the same foot
            # (they are used for mid-stance detection in case of a running step)
            mlc = vlc.mean(axis=1).values.astype(float).flatten()
            mrc = vrc.mean(axis=1).values.astype(float).flatten()

            # extrapolate the appropriate gait cycles
            self._cycles = []
            idx0 = 0
            time = vcoords.index.to_numpy()
            while idx0 < events.shape[0]:
                if idx0 + 3 > events.shape[0]:
                    break

                # try to extract a running step
                indices = np.arange(3) + idx0
                sides = events.Side.values[indices]
                sources = events.Source.values[indices]
                samples = events.Value.values[indices]
                times = time[samples]
                if (
                    sources[0] == "TO"
                    and sources[1] == "FS"
                    and sources[2] == "TO"
                    and sides[0] != sides[1]
                    and sides[0] != sides[2]
                ):
                    if sides[1] == "RIGHT":
                        msi = np.argmin(mrc[samples[1] : samples[2]])
                    else:
                        msi = np.argmin(mlc[samples[1] : samples[2]])
                    mst = float(time[samples[1] + msi])
                    cycle = RunningStep(
                        toeoff_s=times[0],
                        footstrike_s=times[1],
                        midstance_s=mst,
                        ending_s=times[2],
                        side=sides[1],
                    )
                    self._cycles += [cycle]
                    idx0 += 2
                    continue

                # try to extract a walking stride
                if idx0 + 5 > events.shape[0]:
                    break
                indices = np.arange(5) + idx0
                sides = events.Side.values[indices]
                sources = events.Source.values[indices]
                samples = events.Value.values[indices]
                times = time[samples]
                if (
                    sources[0] == "TO"  # ipsilateral
                    and sources[1] == "FS"  # ipsilateral
                    and sources[2] == "TO"  # contralateral
                    and sources[3] == "FS"  # contralateral
                    and sources[4] == "TO"  # ipsilateral
                    and sides[0] == sides[1]
                    and sides[0] != sides[2]
                    and sides[0] != sides[3]
                    and sides[0] == sides[4]
                ):
                    cycle = WalkingStride(
                        ipsilateral_toeoff_s=times[0],
                        ipsilateral_footstrike_s=times[1],
                        contralateral_footstrike_s=times[2],
                        contralateral_toeoff_s=times[3],
                        ending_s=times[4],
                        side=sides[0],
                    )
                    self._cycles += [cycle]
                    idx0 += 2
                    continue

                # if no events have been found increase the counter
                idx0 += 1

    # * constructors

    def __init__(
        self,
        grf: pd.DataFrame | None = None,
        cop: pd.DataFrame | None = None,
        lheel: pd.DataFrame | None = None,
        ltoe: pd.DataFrame | None = None,
        rheel: pd.DataFrame | None = None,
        rtoe: pd.DataFrame | None = None,
        lmidfoot: pd.DataFrame | None = None,
        rmidfoot: pd.DataFrame | None = None,
        preprocess: bool = True,
        height_thresh: float | int = 0.02,
        force_thresh: float | int = 30,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
    ):

        # check preprocess
        if not isinstance(preprocess, bool):
            raise TypeError("preprocess must be True or False")
        self._preprocessed = preprocess

        # check height_thresh
        if not isinstance(height_thresh, (int, float)):
            raise ValueError("height_thresh must be an int or float.")
        self._height_threshold = height_thresh

        # check force_thresh
        if not isinstance(force_thresh, (int, float)):
            raise ValueError("force_thresh must be an int or float.")
        self._force_threshold = force_thresh

        # check the cop
        if self._valid_dataframe(cop):
            cols = cop.columns.get_level_values(0)  # type: ignore
            cols = pd.MultiIndex.from_product(
                iterables=[["COP"], ["X", "Y", "Z"], ["m"]],
                names=["NAME", "AXIS", "UNIT"],
            )
            obj = cop.copy()  # type: ignore
            obj.columns = cols
            self._raw_cop = obj
            if preprocess:
                self._cop = self._apply_preprocessing(obj, 0, 12)
            else:
                self._cop = self._raw_cop.copy()  # type: ignore
        else:
            self._raw_cop = None
            self._cop = None

        # check the grf
        if self._valid_dataframe(grf):
            cols = grf.columns.get_level_values(0)  # type: ignore
            cols = pd.MultiIndex.from_product(
                iterables=[["GRF"], ["X", "Y", "Z"], ["N"]],
                names=["NAME", "AXIS", "UNIT"],
            )
            obj = grf.copy()  # type: ignore
            obj.columns = cols
            self._raw_grf = obj
            if preprocess:
                self._grf = self._apply_preprocessing(obj, 0, 12)
            else:
                self._grf = self._raw_grf.copy()  # type: ignore
        else:
            self._raw_grf = None
            self._grf = None

        # check the marker entries
        _raw_markers_in = {
            "lHeel": lheel,
            "lToe": ltoe,
            "rHeel": rheel,
            "rToe": rtoe,
            "lMid": lmidfoot,
            "rMid": rmidfoot,
        }
        _raw_marker_coords = []
        for key, val in _raw_markers_in.items():
            if self._valid_dataframe(val):
                cols = val.columns.get_level_values(0)  # type: ignore
                cols = pd.MultiIndex.from_product(
                    iterables=[[key], ["X", "Y", "Z"], ["m"]],
                    names=["NAME", "AXIS", "UNIT"],
                )
                obj = val.copy()  # type: ignore
                obj.columns = cols
                _raw_marker_coords += [obj]
        if len(_raw_marker_coords) > 0:
            _raw_markers = pd.concat(_raw_marker_coords, axis=1)
            if self._grf is None or self._cop is None:
                lbls = np.unique(_raw_markers.columns.get_level_values(0))
                lbls = lbls.tolist()
                for i in ["Heel", "Toe"]:
                    for j in ["r", "l"]:
                        lbl = j + i
                        if lbl not in lbls:
                            msg = "'lHeel', 'rHeel', 'lToe' and 'rToe' must"
                            msg += " be provided if 'grf' and 'cop' are None."
                            raise ValueError(msg)
            self._raw_coordinates = _raw_markers
            if preprocess:
                self._coordinates = self._apply_preprocessing(
                    frame=self._raw_coordinates,
                    filling_value=None,
                    cutoff_freq=12,
                )
            else:
                self._coordinates = self._raw_coordinates.copy()
        else:
            self._raw_coordinates = None
            self._coordinates = None

        # check the axis
        if not isinstance(vertical_axis, (str, Literal)):
            raise ValueError('vertical_axies must be any between "X", "Y", "Z".')
        self._vertical_axis = vertical_axis

        # find steps and strides
        self._cycles: list[RunningStep | WalkingStride] = []
        self._find_cycles_from_markers()
        if len(self._cycles) == 0:
            self._find_cycles_from_grf()

    @classmethod
    def from_tdf_file(
        cls,
        file: str,
        grf_label: str | None = None,
        rheel_label: str | None = None,
        lheel_label: str | None = None,
        rtoe_label: str | None = None,
        ltoe_label: str | None = None,
        lmid_label: str | None = None,
        rmid_label: str | None = None,
        height_thresh: float | int = 0.02,
        force_thresh: float | int = 30,
    ):
        """
        Generate a GaitTest object directly from a .tdf file.

        Parameters
        ----------
        file: str
            the path to a ".tdf" file.

        grf_label: str | None (optional, default = None),
            the label of the resultant force vector in the tdf file.

        lheel_label: str | None (optional, default = None),
            the label of the marker defining the left heel in the tdf file.

        rheel_label: str | None (optional, default = None),
            the label of the marker defining the right heel in the tdf file.

        ltoe_label: str | None (optional, default = None),
            the label of the marker defining the left toe in the tdf file.

        rtoe_label: str | None (optional, default = None),
            the label of the marker defining the right toe in the tdf file.

        lmid_label: str | None (optional, default = None),
            the label of the marker defining the right toe in the tdf file.

        rmid_label: str | None (optional, default = None),
            the label of the marker defining the right toe in the tdf file.

        height_thresh: float | int (optional, default = 0.02),
            the minimum height threshold in meters to be considered for
            assuming that one foot is in contact to the ground.

        force_thresh: float | int (optional, default = 30),
            the minimum force threshold in N to be considered for assuming that
            one foot is in contact to the ground.

        Returns
        -------
        test: GaitTest
            the GaitTest class object.
        """

        # check the input file
        if not (exists(file) and isinstance(file, str)):
            raise ValueError("file must be the path to an existing .tdf file.")
        tdf = read_tdf(file)

        # check the markers
        markers = {
            "lheel": lheel_label,
            "rheel": rheel_label,
            "ltoe": ltoe_label,
            "rtoe": rtoe_label,
        }
        out = {}
        if np.any([i is not None for i in markers.values()]):
            try:
                mks = tdf["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
            except Exception:
                msg = "the Input file does not contain all the required "
                msg += "tracked data."
                raise ValueError(msg)
            for key, value in markers.items():
                if value is not None:
                    out[key] = mks[value]
        opt_markers = {
            "lmidfoot": lmid_label,
            "rmidfoot": rmid_label,
        }
        if np.any([i is not None for i in opt_markers.values()]):
            for key, value in opt_markers.items():
                if value in mks.columns.tolist():
                    out[key] = mks[value]

        # check the grf
        if grf_label is not None:
            try:
                frz = tdf["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]  # type: ignore
            except Exception:
                msg = "the Input file does not contain all the required "
                msg += "tracked data."
                raise ValueError(msg)
            forces = np.unique(frz.columns.get_level_values(0)).tolist()
            if grf_label not in forces:
                msg = "the Input file does not contain all the required "
                msg += "tracked data."
                raise ValueError(msg)
            out["grf"] = frz[grf_label].FORCE
            out["cop"] = frz[grf_label].ORIGIN
        else:
            out["grf"] = None
            out["cop"] = None

        # finalize and create the object
        obj = cls(
            preprocess=True,
            height_thresh=height_thresh,
            force_thresh=force_thresh,
            **out,
        )
        obj._source_file = file
        return obj
