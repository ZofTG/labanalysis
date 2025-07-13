"""basic gait module

This module provides classes and utilities for gait analysis, including
GaitObject, GaitCycle, and GaitTest, which support kinematic and kinetic
cycle detection, event extraction, and biofeedback summary generation.
"""

#! IMPORTS


import warnings
from itertools import product
from typing import Literal

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from ..timeseriesrecords.timeseriesrecord import TimeseriesRecord

#! CONSTANTS


__all__ = ["GaitObject"]


#! CLASSESS


class GaitObject(TimeseriesRecord):
    """
    Base class for gait objects, holding marker and force platform data.

    Parameters
    ----------
    algorithm : Literal['kinematics', 'kinetics']
        The cycle detection algorithm to use.
    left_heel : Point3D or None, optional
        Left heel marker data.
    right_heel : Point3D or None, optional
        Right heel marker data.
    left_toe : Point3D or None, optional
        Left toe marker data.
    right_toe : Point3D or None, optional
        Right toe marker data.
    left_metatarsal_head : Point3D or None, optional
        Left metatarsal head marker data.
    right_metatarsal_head : Point3D or None, optional
        Right metatarsal head marker data.
    ground_reaction_force : ForcePlatform or None, optional
        Ground reaction force data.
    ground_reaction_force_threshold : float or int, optional
        Minimum ground reaction force for contact detection.
    height_threshold : float or int, optional
        Maximum vertical height for contact detection.
    vertical_axis : Literal['X', 'Y', 'Z'], optional
        The vertical axis.
    antpos_axis : Literal['X', 'Y', 'Z'], optional
        The anterior-posterior axis.
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals to include.
    """

    # * class variables

    _algorithm: Literal["kinetics", "kinematics"]
    _grf_threshold: float
    _height_threshold: float

    # * constructor

    def __init__(
        self,
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
        Initialize a GaitObject.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            The cycle detection algorithm to use.
        left_heel : Point3D or None, optional
            Left heel marker data.
        right_heel : Point3D or None, optional
            Right heel marker data.
        left_toe : Point3D or None, optional
            Left toe marker data.
        right_toe : Point3D or None, optional
            Right toe marker data.
        left_metatarsal_head : Point3D or None, optional
            Left metatarsal head marker data.
        right_metatarsal_head : Point3D or None, optional
            Right metatarsal head marker data.
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
        # Prepare the signals dict for TimeseriesRecord
        signals = {}

        # Only allow Point3D or None for marker inputs
        def ensure_point3d(val):
            if val is None or isinstance(val, Point3D):
                return val
            raise TypeError("Expected Point3D or None for marker input.")

        # Only allow ForcePlatform or None for grf
        def ensure_forceplatform(val):
            if val is None or isinstance(val, ForcePlatform):
                return val
            raise TypeError("Expected ForcePlatform or None for grf.")

        left_heel = ensure_point3d(left_heel)
        right_heel = ensure_point3d(right_heel)
        left_toe = ensure_point3d(left_toe)
        right_toe = ensure_point3d(right_toe)
        left_metatarsal_head = ensure_point3d(left_metatarsal_head)
        right_metatarsal_head = ensure_point3d(right_metatarsal_head)
        ground_reaction_force = ensure_forceplatform(ground_reaction_force)

        if left_heel is not None:
            signals["left_heel"] = left_heel
        if right_heel is not None:
            signals["right_heel"] = right_heel
        if left_toe is not None:
            signals["left_toe"] = left_toe
        if right_toe is not None:
            signals["right_toe"] = right_toe
        if left_metatarsal_head is not None:
            signals["left_metatarsal_head"] = left_metatarsal_head
        if right_metatarsal_head is not None:
            signals["right_metatarsal_head"] = right_metatarsal_head
        if ground_reaction_force is not None:
            signals["ground_reaction_force"] = ground_reaction_force
        signals.update(extra_signals)
        super().__init__(
            vertical_axis=vertical_axis,
            anteroposterior_axis=antpos_axis,
            strip=strip,
            reset_time=reset_time,
            **signals,
        )

        # set the algorithm
        self.set_algorithm(algorithm)

        # set the thresholds
        self.set_height_threshold(height_threshold)
        self.set_grf_threshold(ground_reaction_force_threshold)

    @property
    def algorithm(self):
        """
        Get the selected cycle detection algorithm.

        Returns
        -------
        str
            The algorithm label.
        """
        return self._algorithm

    @property
    def ground_reaction_force(self) -> None | ForcePlatform:
        """
        Get the ground reaction force object.

        Returns
        -------
        ForcePlatform or None
            The ground reaction force object or None if not available.
        """
        return self.get("ground_reaction_force")

    @property
    def left_heel(self) -> None | Point3D:
        """
        Get the left heel marker.

        Returns
        -------
        Point3D or None
        """
        return self.get("left_heel")

    @property
    def right_heel(self) -> None | Point3D:
        """
        Get the right heel marker.

        Returns
        -------
        Point3D or None
        """
        return self.get("right_heel")

    @property
    def left_toe(self) -> None | Point3D:
        """
        Get the left toe marker.

        Returns
        -------
        Point3D or None
        """
        return self.get("left_toe")

    @property
    def right_toe(self) -> None | Point3D:
        """
        Get the right toe marker.

        Returns
        -------
        Point3D or None
        """
        return self.get("right_toe")

    @property
    def left_metatarsal_head(self) -> None | Point3D:
        """
        Get the left metatarsal head marker.

        Returns
        -------
        Point3D or None
        """
        return self.get("left_metatarstal_head")

    @property
    def right_metatarsal_head(self) -> None | Point3D:
        """
        Get the right metatarsal head marker.

        Returns
        -------
        Point3D or None
        """
        return self.get("right_metatarsal_head")

    @property
    def extra_signals(self):
        """
        Get extra signals not directly used by the analysis.

        Returns
        -------
        dict
            Dictionary of extra signals.
        """
        refs = product(["left", "right"], ["toe", "heel", "metatarsal_head"])
        refs = ["_".join(i) for i in refs] + ["ground_reaction_force"]
        keys = [i for i in self.keys() if i not in refs]
        out: dict[str, Signal1D | Signal3D | Point3D | EMGSignal | ForcePlatform] = {i: self.get(i) for i in keys}  # type: ignore
        return out

    @property
    def ground_reaction_force_threshold(self):
        """
        Get the ground reaction force threshold.

        Returns
        -------
        float
        """
        return self._grf_threshold

    @property
    def height_threshold(self):
        """
        Get the height threshold.

        Returns
        -------
        float
        """
        return self._height_threshold

    def set_grf_threshold(self, threshold: float | int):
        """
        Set the ground reaction force threshold.

        Parameters
        ----------
        threshold : float or int
            Threshold value.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._grf_threshold = float(threshold)

    def set_height_threshold(self, threshold: float | int):
        """
        Set the height threshold.

        Parameters
        ----------
        threshold : float or int
            Threshold value.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._height_threshold = float(threshold)

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """
        Set the gait cycle detection algorithm.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            Algorithm label.
        """
        algorithms = ["kinematics", "kinetics"]
        if not isinstance(algorithm, str) or algorithm not in algorithms:
            msg = "'algorithm' must be any between 'kinematics' or 'kinetics'."
            raise ValueError(msg)
        algo = algorithm
        if (
            algo == "kinetics"
            and self.resultant_force is None
            and all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"'forceplatforms data' not found. The 'algorithm' option"
            msg += " has been set to 'kinematics'."
            warnings.warn(msg)
            algo = "kinematics"
        elif (
            algo == "kinematics"
            and self.resultant_force is not None
            and not all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"Not all left_heel, right_heel, left_toe and right_toe"
            msg += " markers have been found to run the 'kinematics' algorithm."
            msg += " The 'kinetics' algorithm has therefore been selected."
            warnings.warn(msg)
            algo = "kinetics"
        elif self.resultant_force is None and any(
            [
                self.left_heel is None,
                self.left_toe is None,
                self.right_heel is None,
                self.right_toe is None,
            ]
        ):
            msg = "Neither ground reaction force nor left_heel, right_heel, "
            msg += "left_toe and right_toe markers have been found."
            msg += " Therefore none of the available algorithms can be used."
            raise ValueError(msg)

        self._algorithm = algo
