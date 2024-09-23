"""kinematics module"""

#! IMPORTS


from os.path import exists
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from scipy.signal import detrend

from ..signalprocessing import continuous_batches, fillna, butterworth_filt
from labio import read_tdf


#! CONSTANTS


__all__ = [
    "GaitTest",
    "RunningStep",
    "RunningStride",
    "WalkingStep",
    "WalkingStride",
]


#! CLASSESS


class _Step:
    """
    Step class used internally for generation of walking and running step
    objects.

    Parameters
    ----------
    foot_strike_s: int | float
        the foot-strike time in seconds.

    mid_stance_s: int | float
        the mid-stance time in seconds.

    toe_off_s: int | float
        the toe_off time in seconds.

    landing_s: int | float
        the landing time in seconds.

    side: Literal['RIGHT', 'LEFT'] | None (default = None)
        the step's side.

    grf: Iterable[int | float] | None (default = None)
        the ground reaction force data over the step

    cop_ml: Iterable[int | float] | None (default = None)
        the medial-lateral coordinates of the centre of pressure over the step

    cop_ap: Iterable[int | float] | None (default = None)
        the anterior-posterior coordinates of the centre of pressure over the step
    """

    _fs: int | float
    _ms: int | float
    _to: int | float
    _ld: int | float
    _side: Literal["RIGHT", "LEFT"] | None
    _grf: Iterable[int | float] | None
    _cop_ml: Iterable[int | float] | None
    _cop_ap: Iterable[int | float] | None

    @property
    def grf_kgf(self):
        """return the ground reaction force data in kgf"""
        if self._grf is None:
            return None
        out = np.array(self._grf, dtype=float).flatten()
        time = np.linspace(self.foot_strike_s, self.landing_s, len(out))
        return out, time.flatten()

    @property
    def cop_ml_m(self):
        """return the medial-lateral coordinates of the centre of pressure in m"""
        if self._cop_ml is None:
            return None
        out = np.array(self._cop_ml, dtype=float).flatten()
        time = np.linspace(self.foot_strike_s, self.landing_s, len(out))
        return out, time.flatten()

    @property
    def cop_ap_m(self):
        """return the anterior-posterior coordinates of the centre of pressure in m"""
        if self._cop_ap is None:
            return None
        out = np.array(self._cop_ap, dtype=float).flatten()
        time = np.linspace(self.foot_strike_s, self.landing_s, len(out))
        return out, time.flatten()

    @property
    def foot_strike_s(self):
        """return the foot-strike time"""
        return self._fs

    @property
    def mid_stance_s(self):
        """return the mid_stance time"""
        return self._ms

    @property
    def toe_off_s(self):
        """return the toe-off time"""
        return self._to

    @property
    def landing_s(self):
        """return the landing time"""
        return self._ld

    @property
    def step_time_s(self):
        """return the step time"""
        return self.landing_s - self.foot_strike_s

    @property
    def step_cadence_spm(self):
        """return the foot-strike time"""
        return 60.0 / self.step_time_s

    @property
    def grf_max_kgf(self):
        """return the peak ground reaction force of the step"""
        if self.grf_kgf is None:
            return None
        return float(np.max(self.grf_kgf))

    @property
    def lat_dis_m(self):
        """return the lateral displacement during the step"""
        if self.cop_ml_m is None:
            return None
        return float(abs(np.max(self.cop_ml_m) - np.min(self.cop_ml_m)))

    @property
    def side(self):
        """return the step side"""
        return self._side

    def as_dict(self, user_weight_kg: float | int | None = None):
        """
        return the current step as dict

        Parameters
        ----------
        user_weight_kg: float | int | None (optional, default = None)

        Results
        -------
        out: dict[(str, str), None | int | float]
            the output values
        """
        keys = [i for i in dir(self) if i[0] != "_" and i[:2] != "as"]
        out = {}
        for i in keys:
            val = getattr(self, i)
            if isinstance(val, tuple):
                out["time"] = val[1]
                val = val[0]
            if isinstance(val, Callable):
                if user_weight_kg is not None:
                    out[i] = val(user_weight_kg)
                    out["user_weight_kg"] = user_weight_kg
                else:
                    out[i] = None
            else:
                out[i] = val
        return out

    def __repr__(self):
        return pd.Series(self.as_dict()).__repr__()

    def __str__(self):
        return pd.Series(self.as_dict()).__str__()

    def __init__(
        self,
        foot_strike_s: int | float,
        mid_stance_s: int | float,
        toe_off_s: int | float,
        landing_s: int | float,
        side: Literal["RIGHT", "LEFT"] | None = None,
        grf_kgf: Iterable[int | float] | None = None,
        cop_ml: Iterable[int | float] | None = None,
        cop_ap: Iterable[int | float] | None = None,
    ):
        self._fs = foot_strike_s
        self._ms = mid_stance_s
        self._to = toe_off_s
        self._ld = landing_s
        self._side = side
        self._grf = grf_kgf
        self._cop_ap = cop_ap
        self._cop_ml = cop_ml
        series = [grf_kgf, cop_ap, cop_ml]
        lens = [len(i) for i in series if i is not None]  # type: ignore
        if len(lens) > 0:
            if not all(i == lens[0] for i in lens[1:]):
                msg = "grf_kgf, cop_ml and cop_ap must have the same length."
                raise ValueError(msg)


class RunningStep(_Step):
    """
    Running Step object.

    Parameters
    ----------
    foot_strike_s: int | float
        the foot-strike time in seconds.

    mid_stance_s: int | float
        the mid-stance time in seconds.

    toe_off_s: int | float
        the toe_off time in seconds.

    landing_s: int | float
        the landing time in seconds.

    side: Literal['RIGHT', 'LEFT'] | None (default = None)
        the step's side.
    """

    @property
    def contact_time_s(self):
        """return the step contact time in seconds."""
        return self.toe_off_s - self.foot_strike_s

    @property
    def propulsion_time_s(self):
        """return the step propulsion time in seconds."""
        return self.toe_off_s - self.mid_stance_s

    @property
    def loading_response_s(self):
        """return the step loading response time in seconds."""
        return self.mid_stance_s - self.foot_strike_s

    @property
    def flight_time_s(self):
        """return the step flight time in seconds."""
        return self.landing_s - self.toe_off_s

    def vrt_dis_m(self, user_weight_kg: float | int = 70):
        """
        return the vertical displacement during the step

        Parameters
        ----------
        user_weight_kg: float | int (default = 70)
            the user weight in kg

        Returns
        -------
        return the vertical displacement during the step in meters.

        Description
        -----------
        the vertical displacement of the CoM is calculated assuming that,
        during the flying phase, the motion of the CoM is parabolic:

                                Hf = g * K ** 2 / 8
        where:
            K: is the flying time (in s)
            g: is the acceleration of gravity (9.80665 m/s**2)

        The resulting vertical displacement is then added to the vertical
        displacement calculated from the measured force and according to
        the Morin et al. (2005) model.

                    Hc = C ** 2 * ( g / 8 - F / ( M * pi ** 2 ) )

        where:
            C: is twice the loading response time (in s)
            F: is the peak force measured during the step (in N)
            M: is the USER WEIGHT (in kg)

        Thus, the total vertical displacement becomes:

                            Ht = Hc + Hf

        References
        ----------
        Morin, Jean Benoît; Dalleau, Georges; Kyröläinen, Heikki; Jeannin
        Thibault; Belli, Alain;
            A simple method for measuring stiffness during running.
            2005 Journal of applied biomechanics 21(2):167-180.
        """
        # check if the necessary data are available
        if self.grf_max_kgf is None:
            return None

        # do the calculations
        Hf = (self.flight_time_s**2) / 8
        Hc = (self.loading_response_s * 2) ** 2
        Hc *= 1 / 8 - self.grf_max_kgf / user_weight_kg / np.pi**2
        return 9.80665 * (Hf + Hc)

    def __init__(
        self,
        foot_strike_s: int | float,
        mid_stance_s: int | float,
        toe_off_s: int | float,
        landing_s: int | float,
        side: Literal["RIGHT", "LEFT"] | None = None,
    ):
        super().__init__(
            foot_strike_s=foot_strike_s,
            mid_stance_s=mid_stance_s,
            toe_off_s=toe_off_s,
            landing_s=landing_s,
            side=side,
        )


class WalkingStep(_Step):
    """
    Walking Step object.

    Parameters
    ----------
    foot_strike_s: int | float
        the foot-strike time in seconds.

    mid_stance_s: int | float
        the mid-stance time in seconds.

    toe_off_s: int | float
        the toe_off time in seconds.

    landing_s: int | float
        the landing time in seconds.

    side: Literal['RIGHT', 'LEFT'] | None (default = None)
        the step's side.
    """

    @property
    def single_support_time_s(self):
        """return the single support time in seconds."""
        return self.landing_s - self.toe_off_s

    @property
    def double_support_time_s(self):
        """return the double support time in seconds."""
        return self.toe_off_s - self.foot_strike_s

    def vrt_dis_m(self, user_weight_kg: float | int = 70):
        """
        return the vertical displacement during the step calculated via double
        integration of the mean acceleration.

        Parameters
        ----------
        user_weight_kg: float | int (default = 70)
            the user weight in kg

        Returns
        -------
        return the vertical displacement during the step in meters.

        References
        ----------
        Saini, M., Kerrigan, D. C., Thirunarayan, M. A., & Duff-Raffaele, M.
        (1998). The vertical displacement of the center of mass during walking:
        a comparison of four measurement methods.
        Journal of Biomechanical Engineering, 120(1), 133–139.
        https://doi.org/10.1115/1.2834293
        """
        # check if the necessary data are available
        if self.grf_kgf is None:
            return None

        # extract the (detrended) vertical displacement
        F, T = self.grf_kgf
        G = 9.80665
        A = (F * G - user_weight_kg * G) / user_weight_kg
        D = detrend(np.trapezoid(np.trapezoid(A, T), T).astype(float))

        # get the range
        return np.max(D) - np.min(D)

    def __init__(
        self,
        foot_strike_s: int | float,
        mid_stance_s: int | float,
        toe_off_s: int | float,
        landing_s: int | float,
        side: Literal["RIGHT", "LEFT"] | None = None,
    ):
        super().__init__(
            foot_strike_s=foot_strike_s,
            mid_stance_s=mid_stance_s,
            toe_off_s=toe_off_s,
            landing_s=landing_s,
            side=side,
        )


class _Stride:
    """
    basic stride class.

    Parameters
    ----------
    step1: WalkingStep
        the first step of the stride.

    step2: WalkingStep
        the second step of the stride.
    """

    _step1: WalkingStep | RunningStep
    _step2: WalkingStep | RunningStep

    @property
    def step_1(self):
        "return the first step"
        return self._step1

    @property
    def step_2(self):
        "return the second step"
        return self._step2

    @property
    def stride_time_s(self):
        """return the stride time in seconds"""
        return self.step_1.step_time_s + self.step_2.step_time_s

    @property
    def stride_cadence_spm(self):
        """return the stride cadence in spm"""
        return 60.0 / self.stride_time_s

    @property
    def grf_kgf(self):
        """return the ground reaction force data in kgf"""
        if self.step_1.grf_kgf is None or self.step_2.grf_kgf is None:
            return None
        f1, t1 = self.step_1.grf_kgf
        f2, t2 = self.step_2.grf_kgf
        return np.concatenate([f1, f2[1:]]), np.concatenate([t1, t2[1:]])

    @property
    def cop_ml_m(self):
        """return the medial-lateral coordinates of the centre of pressure in m"""
        if self.step_1.cop_ml_m is None or self.step_2.cop_ml_m is None:
            return None
        f1, t1 = self.step_1.cop_ml_m
        f2, t2 = self.step_2.cop_ml_m
        return np.concatenate([f1, f2[1:]]), np.concatenate([t1, t2[1:]])

    @property
    def cop_ap_m(self):
        """return the anterior-posterior coordinates of the centre of pressure in m"""
        if self.step_1.cop_ap_m is None or self.step_2.cop_ap_m is None:
            return None
        f1, t1 = self.step_1.cop_ap_m
        f2, t2 = self.step_2.cop_ap_m
        return np.concatenate([f1, f2[1:]]), np.concatenate([t1, t2[1:]])

    @property
    def side(self):
        """return the stride side"""
        return self.step_1.side

    def as_dict(self):
        """return the current step as dict"""
        st1 = self.step_1.as_dict()
        st1 = {(i, self.step_1.side): v for i, v in st1.items() if i != "side"}
        st2 = self.step_2.as_dict()
        st2 = {(i, self.step_2.side): v for i, v in st2.items() if i != "side"}
        return {**st1, **st2}

    def __repr__(self):
        return pd.Series(self.as_dict()).__repr__()

    def __str__(self):
        return pd.Series(self.as_dict()).__str__()

    def __init__(
        self,
        step1: WalkingStep | RunningStep,
        step2: WalkingStep | RunningStep,
    ):
        self._step1 = step1
        self._step2 = step2


class WalkingStride(_Stride):
    """
    Walking Stride class.

    Parameters
    ----------
    step1: WalkingStep
        the first step of the stride.

    step2: WalkingStep
        the second step of the stride.
    """

    _step1: WalkingStep
    _step2: WalkingStep

    @property
    def single_support_time_s(self):
        """return the stride single support time in seconds."""
        return self._step1.single_support_time_s + self._step2.single_support_time_s

    @property
    def double_support_time_s(self):
        """return the stride double support time in seconds."""
        return self._step1.double_support_time_s + self._step2.double_support_time_s

    @property
    def stance_time_s(self):
        """return the stance time in seconds."""
        return self.step_2.toe_off_s - self.step_1.foot_strike_s

    @property
    def swing_time_s(self):
        """return the swing time in seconds."""
        return self.step_2.landing_s - self.step_1.toe_off_s

    def __init__(self, step1: WalkingStep, step2: WalkingStep):
        super().__init__(step1, step2)


class RunningStride(_Stride):
    """
    Running Stride class.

    Parameters
    ----------
    step1: RunningStep
        the first step of the stride.

    step2: RunningStep
        the second step of the stride.
    """

    _step1: RunningStep
    _step2: RunningStep

    def __init__(self, step1: RunningStep, step2: RunningStep):
        self._step1 = step1
        self._step2 = step2


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
    _steps: list[RunningStep | WalkingStep]
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
        return self._cop

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
    def steps(self):
        """the detected steps"""
        return self._steps

    @property
    def strides(self):
        """the detected strides"""
        strides: list[RunningStride | WalkingStride] = []
        for st1, st2 in zip(self._steps[:-1], self._steps[1:]):
            valid_stride = (
                st1.landing_s == st2.foot_strike_s
                and st1.side is not None
                and st2.side is not None
                and st1.side != st2.side
            )
            if (
                valid_stride
                and isinstance(st1, RunningStep)
                and isinstance(st2, RunningStep)
            ):
                strides += [RunningStride(st1, st2)]
            elif (
                valid_stride
                and isinstance(st1, WalkingStep)
                and isinstance(st2, WalkingStep)
            ):
                strides += [WalkingStride(st1, st2)]

        return strides

    # * methods

    def strides_summary(self):
        """return a summary of the collected strides"""
        out = []
        for i, stride in enumerate(self.strides):
            line = pd.DataFrame(pd.Series(stride.as_dict())).T
            line.insert(0, ("Stride", "#"), [i + 1])
            out += [line]
        out = pd.concat(out, ignore_index=True)
        out.sort_index(axis=1, inplace=True)
        return out

    def steps_summary(self):
        """return a summary of the collected steps"""
        out = []
        for i, step in enumerate(self.steps):
            line = pd.DataFrame(pd.Series(step.as_dict())).T
            line.insert(0, "step_#", [i + 1])
            out += [line]
        out = pd.concat(out, ignore_index=True)
        steps = out["step_#"].values.astype(int).flatten()
        out.drop("step_#", axis=1, inplace=True)
        out.sort_index(axis=1, inplace=True)
        out.insert(0, "step_#", steps)
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
        batches = continuous_batches(all_nans_mask)
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
        out = pd.DataFrame(fillna(frame.iloc[idx, :], filling_value, 6))

        # smooth all marker coordinates
        return out.apply(
            butterworth_filt,  # type: ignore
            fcut=cutoff_freq,
            fsamp=1 / np.mean(np.diff(frame.index.to_numpy())),
            order=6,
            ftype="lowpass",
            phase_corrected=True,
            raw=True,
        )

    def _find_steps_from_grf(self):
        """find steps via grf coordinates"""
        if self.grf is not None and self.cop is not None:
            # TODO
            pass

    def _find_steps_from_markers(self):
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

            # get the mean values (they are used for mid-stance detection)
            mlc = vlc.mean(axis=1)
            mrc = vrc.mean(axis=1)

            # get the batches of time with part of the feet on ground
            glc = vlc < self._height_threshold
            grc = vrc < self._height_threshold
            blc = continuous_batches(glc.any(axis=1).values.astype(bool))
            brc = continuous_batches(grc.any(axis=1).values.astype(bool))

            # exclude those batches that start at the beginning of the data
            # acquisition or are continuning at the end of the data acquisition
            if len(blc) > 0:
                if blc[0][0] == 0:
                    blc = blc[1:]
                if blc[-1][-1] >= len(vlc) - 1:
                    blc = blc[:-1]
            if len(brc) > 0:
                if brc[0][0] == 0:
                    brc = brc[1:]
                if brc[-1][-1] >= len(vrc) - 1:
                    brc = brc[:-1]

            # get the events
            time = self.coordinates.index.to_numpy()
            evts_map = {
                "FS LEFT": [time[i[0]] for i in blc],
                "FS RIGHT": [time[i[0]] for i in brc],
                "MS LEFT": [time[np.argmin(mlc.iloc[i]) + i[0]] for i in blc],
                "MS RIGHT": [time[np.argmin(mrc.iloc[i]) + i[0]] for i in brc],
                "TO LEFT": [time[i[-1]] for i in blc],
                "TO RIGHT": [time[i[-1]] for i in brc],
            }
            evts_map = {i: np.array(j) for i, j in evts_map.items()}
            self._extract_steps(evts_map)

    def _extract_steps(
        self,
        evts: dict[str, np.ndarray],
    ):
        """extract steps from events map"""
        evts_val = np.concatenate(list(evts.values()))
        evts_lbl = [np.tile(i, len(v)) for i, v in evts.items()]
        evts_lbl = np.concatenate(evts_lbl)
        evts_idx = np.argsort(evts_val)
        evts_val = evts_val[evts_idx]
        evts_side = np.array([i.split(" ")[1] for i in evts_lbl[evts_idx]])
        evts_lbl = np.array([i.split(" ")[0] for i in evts_lbl[evts_idx]])

        # get the steps
        run_seq = ["FS", "MS", "TO", "LD"]
        walk_seq = ["FS", "TO", "MS", "LD"]
        for n in np.arange(0, len(evts_lbl) - 4, 3):
            idx = np.arange(4) + n
            seq = evts_lbl[idx].copy()
            seq[-1] = "LD"
            sides = evts_side[idx].copy()
            vals = evts_val[idx].copy()
            s0 = sides[0]
            if (
                all([i == v for i, v in zip(seq, run_seq)])
                & all(i == s0 for i in sides[:-1])
                & (sides[-1] != s0)
            ):  # running
                # TODO add COP and GRF parameters
                self._steps += [RunningStep(*vals, side=s0.upper())]
            elif (
                all([i == v for i, v in zip(seq, walk_seq)])
                & all(i == s0 for i in sides[2:-1])
                & (sides[1] != s0)
                & (sides[-1] != s0)
            ):  # walking
                # TODO add COP and GRF parameters
                self._steps += [WalkingStep(*vals, side=s0.upper())]

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
                iterables=[["GRF"], ["X", "Y", "Z"], ["m"]],
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
        self._steps: list[RunningStep | WalkingStep] = []
        self._find_steps_from_markers()
        if len(self._steps) == 0:
            self._find_steps_from_grf()

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
            "lmidfoot": lmid_label,
            "rmidfoot": rmid_label,
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
