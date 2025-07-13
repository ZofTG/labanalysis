"""
running step
"""

#! IMPORTS


from typing import Literal

import numpy as np

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.timeseriesrecord import TimeseriesRecord
from ..timeseriesrecords.forceplatform import ForcePlatform
from .gaitcycle import GaitCycle

#! CONSTANTS


__all__ = ["RunningStep"]


#! CLASSESS


class RunningStep(GaitCycle):
    """
    Represents a single running step.

    Parameters
    ----------
    side : Literal['left', 'right']
        The side of the cycle.
    frame : StateFrame
        A stateframe object containing all the available kinematic, kinetic and EMG data related to the cycle.
    algorithm : Literal['kinematics', 'kinetics'], optional
        Algorithm used for gait cycle detection. 'kinematics' uses marker data, 'kinetics' uses force platform data.
    left_heel : Point3D or None, optional
        The left heel marker data.
    right_heel : Point3D or None, optional
        The right heel marker data.
    left_toe : Point3D or None, optional
        The left toe marker data.
    right_toe : Point3D or None, optional
        The right toe marker data.
    left_metatarsal_head : Point3D or None, optional
        The left metatarsal head marker data.
    right_metatarsal_head : Point3D or None, optional
        The right metatarsal head marker data.
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

    Note
    ----
    The cycle starts from the toe-off and ends at the next toe-off of the same foot.
    """

    @property
    def flight_phase(self):
        """
        Get the TimeseriesRecord corresponding to the flight phase.

        Returns
        -------
        TimeseriesRecord
        """
        sliced = self[self.init_s : self.footstrike_s]
        out = TimeseriesRecord()
        for key, value in sliced.items():
            out[key] = value
        return out

    @property
    def contact_phase(self):
        """
        Get the TimeseriesRecord corresponding to the contact phase.

        Returns
        -------
        TimeseriesRecord
        """
        sliced = self[self.footstrike_s : self.end_s]
        out = TimeseriesRecord()
        for key, value in sliced.items():
            out[key] = value
        return out

    @property
    def loading_response_phase(self):
        """
        Get the TimeseriesRecord corresponding to the loading response phase.

        Returns
        -------
        TimeseriesRecord
        """
        sliced = self[self.footstrike_s : self.midstance_s]
        out = TimeseriesRecord()
        for key, value in sliced.items():
            out[key] = value
        return out

    @property
    def propulsion_phase(self):
        """
        Get the TimeseriesRecord corresponding to the propulsive phase.

        Returns
        -------
        TimeseriesRecord
        """
        sliced = self[self.midstance_s : self.end_s]
        out = TimeseriesRecord()
        for key, value in sliced.items():
            out[key] = value
        return out

    @property
    def flight_time_s(self):
        """
        Get the flight time in seconds.

        Returns
        -------
        float
            The flight time in seconds.
        """
        return self.footstrike_s - self.init_s

    @property
    def loadingresponse_time_s(self):
        """
        Get the loading response time in seconds.

        Returns
        -------
        float
            The loading response time in seconds.
        """
        return self.midstance_s - self.footstrike_s

    @property
    def propulsion_time_s(self):
        """
        Get the propulsion time in seconds.

        Returns
        -------
        float
            The propulsion time in seconds.
        """
        return self.end_s - self.midstance_s

    @property
    def contact_time_s(self):
        """
        Get the contact time in seconds.

        Returns
        -------
        float
            The contact time in seconds.
        """
        return self.end_s - self.footstrike_s

    def _footstrike_kinetics(self):
        """
        Find the footstrike time using the kinetics algorithm.

        Returns
        -------
        float
            The footstrike time in seconds.

        Raises
        ------
        ValueError
            If no ground reaction force data is available or no footstrike is found.
        """

        # get the contact phase samples
        grf = self.resultant_force
        if grf is None:
            raise ValueError("no ground reaction force data available.")
        vgrf = np.asarray(grf[self.vertical_axis], float).flatten()
        time = grf.index
        grfn = vgrf / np.max(vgrf)
        mask = np.where((grfn < self.height_threshold)[: np.argmax(grfn)])[0]

        # extract the first contact time
        if len(mask) == 0:
            raise ValueError("no footstrike has been found.")

        return float(time[mask[-1]])

    def _footstrike_kinematics(self):
        """
        Find the footstrike time using the kinematics algorithm.

        Returns
        -------
        float
            The footstrike time in seconds.

        Raises
        ------
        ValueError
            If no footstrike has been found.
        """

        # get the relevant vertical coordinates
        vcoords = {}
        contact_foot = self.side.lower()
        for marker in ["heel", "metatarsal_head"]:
            lbl = f"{contact_foot}_{marker}"
            val = self[f"{contact_foot}_{marker}"]
            if val is None:
                continue
            vcoords[lbl] = np.asarray(val[self.vertical_axis], float).flatten()

        # filter the signals and extract the first contact time
        time = self.index
        fs_time = []
        for val in vcoords.values():
            val = val / np.max(val)
            fsi = np.where(val < self.height_threshold)[0]
            if len(fsi) == 0 or fsi[0] == 0:
                raise ValueError("not footstrike has been found.")
            fs_time += [time[fsi[0]]]

        # get output time
        if len(fs_time) > 0:
            return float(np.min(fs_time))
        raise ValueError("no footstrike has been found.")

    def _midstance_kinetics(self):
        """
        Find the midstance time using the kinetics algorithm.

        Returns
        -------
        float
            The midstance time in seconds.

        Raises
        ------
        ValueError
            If no ground reaction force data is available.
        """

        grf = self.resultant_force
        if grf is None:
            raise ValueError("no ground reaction force data available.")
        vgrf = np.asarray(grf[self.vertical_axis], float).flatten()
        time = grf.index
        return float(time[np.argmax(vgrf)])

    def _midstance_kinematics(self):
        """
        Find the midstance time using the kinematics algorithm.

        Returns
        -------
        float
            The midstance time in seconds.
        """

        # get the available markers
        lbls = [f"{self.side.lower()}_{i}" for i in ["heel", "toe"]]
        lbls += [f"{self.side.lower()}_metatarsal_head"]

        # get the mean vertical signal
        time = self.index
        ref = np.zeros_like(time)
        for lbl in lbls:
            val = self[lbl]
            if val is None:
                continue
            ref += np.asarray(val[self.vertical_axis], float).flatten()
        ref /= len(lbls)

        # return the time corresponding to the minimum value
        return float(time[np.argmin(val)])

    # * constructor

    def __init__(
        self,
        side: Literal["right", "left"],
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
        Initialize a RunningStep.

        Parameters
        ----------
        side : {'left', 'right'}
            The side of the cycle.
        algorithm : {'kinematics', 'kinetics'}
            The cycle detection algorithm.
        left_heel, right_heel, left_toe, right_toe : Point3D or None
            Marker data for the respective anatomical points.
        left_metatarsal_head, right_metatarsal_head : Point3D or None
            Metatarsal head marker data for the respective sides.
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
            side=side,
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
