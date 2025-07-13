"""basic gait cycle"""

#! IMPORTS


from typing import Literal

import numpy as np
import pandas as pd

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from .gaitobject import GaitObject

#! CONSTANTS


__all__ = ["GaitCycle"]


#! CLASSESS


class GaitCycle(GaitObject):
    """
    Basic gait cycle class.

    Parameters
    ----------
    side : {'left', 'right'}
        The side of the cycle.
    algorithm : {'kinematics', 'kinetics'}
        The cycle detection algorithm.
    left_heel : Point3D or None
        Marker data for the left heel.
    right_heel : Point3D or None
        Marker data for the right heel.
    left_toe : Point3D or None
        Marker data for the left toe.
    right_toe : Point3D or None
        Marker data for the right toe.
    left_metatarsal_head : Point3D or None
        Marker data for the left metatarsal head.
    right_metatarsal_head : Point3D or None
        Marker data for the right metatarsal head.
    ground_reaction_force : ForcePlatform or None
        Ground reaction force data.
    ground_reaction_force_threshold : float or int, optional
        Minimum ground reaction force for contact detection.
    height_threshold : float or int, optional
        Maximum vertical height for contact detection.
    vertical_axis : {'X', 'Y', 'Z'}, optional
        The vertical axis.
    antpos_axis : {'X', 'Y', 'Z'}, optional
        The anterior-posterior axis.
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals to include.

    Notes
    -----
    The cycle starts from the toeoff and ends at the next toeoff of the same foot.
    """

    # * class variables

    _side: Literal["left", "right"]
    _footstrike_s: float
    _midstance_s: float
    _absolute_time_events: list[str] = [
        "footstrike_s",
        "midstance_s",
        "init_s",
        "end_s",
    ]

    # * constructor

    def __init__(
        self,
        side: Literal["left", "right"],
        algorithm: Literal["kinematics", "kinetics"],
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_metatarsal_head: Point3D | None = None,
        right_metatarsal_head: Point3D | None = None,
        ground_reaction_force: ForcePlatform | None = None,
        ground_reaction_force_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
        strip: bool = True,
        reset_time: bool = True,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a GaitCycle.

        Parameters
        ----------
        side : {'left', 'right'}
            The side of the cycle.
        algorithm : {'kinematics', 'kinetics'}
            The cycle detection algorithm.
        left_heel : Point3D or None
            Marker data for the left heel.
        right_heel : Point3D or None
            Marker data for the right heel.
        left_toe : Point3D or None
            Marker data for the left toe.
        right_toe : Point3D or None
            Marker data for the right toe.
        left_metatarsal_head : Point3D or None
            Marker data for the left metatarsal head.
        right_metatarsal_head : Point3D or None
            Marker data for the right metatarsal head.
        ground_reaction_force : ForcePlatform or None
            Ground reaction force data.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force for contact detection.
        height_threshold : float or int, optional
            Maximum vertical height for contact detection.
        vertical_axis : {'X', 'Y', 'Z'}, optional
            The vertical axis.
        antpos_axis : {'X', 'Y', 'Z'}, optional
            The anterior-posterior axis.
        **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
            Additional signals to include.
        """
        super().__init__(
            algorithm=algorithm,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_metatarsal_head=left_metatarsal_head,
            right_metatarsal_head=right_metatarsal_head,
            ground_reaction_force=ground_reaction_force,
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,
            strip=strip,
            reset_time=reset_time,
            **extra_signals,
        )
        self.set_side(side)

    # * attributes

    @property
    def side(self):
        """
        Return the side of the cycle.

        Returns
        -------
        str
        """
        return self._side

    @property
    def init_s(self):
        """
        Return the first toeoff time in seconds.

        Returns
        -------
        float
        """
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.index[0])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.index[0])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

    @property
    def end_s(self):
        """
        Return the toeoff time corresponding to the end of the cycle in seconds.

        Returns
        -------
        float
        """
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.index[-1])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.index[-1])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

    @property
    def cycle_time_s(self):
        """
        Return the cycle time in seconds.

        Returns
        -------
        float
        """
        return self.end_s - self.init_s

    @property
    def footstrike_s(self):
        """
        Return the foot-strike time in seconds.

        Returns
        -------
        float
        """
        return self._footstrike_s

    @property
    def midstance_s(self):
        """
        Return the mid-stance time in seconds.

        Returns
        -------
        float
        """
        return self._midstance_s

    @property
    def time_events(self):
        """
        Return all the time events defining the cycle.

        Returns
        -------
        pd.DataFrame
        """
        evts: dict[str, float] = {}
        for lbl in dir(self):
            if lbl.endswith("_s") and not lbl.startswith("_"):
                name = lbl.rsplit("_", 1)[0].strip().split(" ")[0].lower()
                time = getattr(self, lbl)
                perc = time
                if lbl in self._absolute_time_events:
                    perc -= self.init_s
                perc = perc / self.cycle_time_s * 100
                evts[f"{name.lower().replace("_time", "")}_s"] = float(time)
                evts[f"{name.lower().replace("_time", "")}_%"] = float(perc)
        return evts

    @property
    def lateral_displacement(self):
        cop = np.asarray(self.centre_of_pressure[self.lateral_axis], float).flatten()
        return float(np.max(cop) - np.min(cop))

    @property
    def peak_force(self):
        return float(np.max(np.asarray(self.vertical_force.data)))

    @property
    def time_to_peak(self):
        vgrf = np.asarray(self.vertical_force.data, float)
        time = self.index
        return float(time[np.argmax(vgrf)] - time[0])

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """

        # get spatio-temporal parameters
        new = {
            "type": self.__class__.__name__,
            "side": self.side,
            **self.time_events,
        }

        # add kinetic parameters
        if self.get("ground_reaction_force") is not None:
            new.update(
                **{
                    f"lateral_displacement_{self.centre_of_pressure.unit}": self.lateral_displacement,
                    f"peak_vertical_force_{self.vertical_force.unit}": self.peak_force,
                    "time_to_peak_force_s": self.time_to_peak,
                }
            )

        # add emg mean activation
        for muscle, emgsignal in self.emgsignals.items():
            if emgsignal.side == self.side:
                avg = float(np.asarray(emgsignal.data, float).mean())
                new[emgsignal.muscle_name] = avg

        return pd.DataFrame(pd.Series(new)).T

    def _footstrike_kinetics(self) -> float:
        """
        Return the foot-strike time in seconds using the kinetics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _footstrike_kinematics(self) -> float:
        """
        Return the foot-strike time in seconds using the kinematics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _midstance_kinetics(self) -> float:
        """
        Return the mid-stance time in seconds using the kinetics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _midstance_kinematics(self) -> float:
        """
        Return the mid-stance time in seconds using the kinematics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _update_events(self):
        """
        Update gait events.
        """
        if self.algorithm == "kinetics":
            try:
                self._midstance_s = self._midstance_kinetics()
            except Exception:
                self._midstance_s = np.nan
            try:
                self._footstrike_s = self._footstrike_kinetics()
            except Exception:
                self._footstrike_s = np.nan
        elif self.algorithm == "kinematics":
            try:
                self._midstance_s = self._midstance_kinematics()
            except Exception:
                self._midstance_s = np.nan
            try:
                self._footstrike_s = self._footstrike_kinematics()
            except Exception:
                self._footstrike_s = np.nan

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """
        Set the gait cycle detection algorithm.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            Algorithm label.
        """
        super().set_algorithm(algorithm)
        self._update_events()

    def set_side(self, side: Literal["right", "left"]):
        """
        Set the cycle side.

        Parameters
        ----------
        side : {'left', 'right'}
        """
        if not isinstance(side, (Literal, str)):
            raise ValueError("'side' must be 'left' or 'right'.")
        if side not in ["left", "right"]:
            raise ValueError("'side' must be 'left' or 'right'.")
        self._side = side
