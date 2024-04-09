"""gait events module containing Step and Stride class objects"""

#! IMPORTS


from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.signal import detrend


#! CONSTANTS


__all__ = ["RunningStep", "WalkingStep", "RunningStride", "WalkingStride"]


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

    def as_dict(self):
        """return the current step as dict"""
        keys = [i for i in dir(self) if i[0] != "_" and i[:2] != "as"]
        out = {}
        for i in keys:
            val = getattr(self, i)
            if isinstance(val, tuple):
                out["time"] = val[1]
                val = val[0]
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
        D = detrend(np.trapz(np.trapz(A, T), T))

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
