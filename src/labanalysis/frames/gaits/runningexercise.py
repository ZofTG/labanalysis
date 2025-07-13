"""
Module for running test analysis.

This module defines classes for performing running test analysis,
including step detection and summary plots.
"""

#! IMPORTS


import warnings
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...signalprocessing import find_peaks
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from .gaitexercise import GaitExercise

__all__ = ["RunningExercise"]


#! CLASSESS


class RunningExercise(GaitExercise):
    """
    Represents a running test.

    Parameters
    ----------
    frame : StateFrame
        A stateframe object containing all the available kinematic, kinetic and EMG data related to the test.
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
    """

    # * methods

    def _find_cycles_kinematics(self):
        """
        Find the gait cycles using the kinematics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required marker is missing or no toe-offs are found.
        Warns
        -----
        UserWarning
            If left-right steps alternation is not guaranteed.
        """

        # get toe-off times
        tos = []
        time = self.index
        fsamp = float(1 / np.mean(np.diff(time)))
        for lbl in ["left_toe", "right_toe"]:

            # get the vertical coordinates of the toe markers
            obj = self.get(lbl)
            if obj is None:
                raise ValueError(f"{lbl} is missing.")
            arr = np.asarray(obj[self.vertical_axis], float).flatten()

            # filter and rescale
            arr = arr / np.max(arr)

            # get the minimum reasonable contact time for each step
            dsamples = int(round(fsamp / 2))

            # get the peaks at each cycle
            pks = find_peaks(arr, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            side = lbl.split("_")[0]
            for pk in pks:
                idx = np.where(arr[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    line = pd.Series({"Time": time[idx[-1]], "Side": side})
                    tos += [pd.DataFrame(line).T]

        # wrap the events
        if len(tos) == 0:
            raise ValueError("no toe-offs have been found.")
        tos = pd.concat(tos, ignore_index=True)
        tos = tos.drop_duplicates()
        tos = tos.sort_values("Time")
        tos = tos.reset_index(drop=True)

        # check the alternation of the steps
        sides = tos.Side.values
        if not all(s0 != s1 for s0, s1 in zip(sides[:-1], sides[1:])):
            warnings.warn("Left-Right steps alternation not guaranteed.")
        for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
            t0 = float(tos.Time.values[i0])
            t1 = float(tos.Time.values[i1])
            step = self[t0:t1]
            args = {
                "side": tos.Side.values[i1],
                "ground_reaction_force_threshold": self.ground_reaction_force_threshold,
                "height_threshold": self.height_threshold,
                "algorithm": self.algorithm,
                "vertical_axis": self.vertical_axis,
                "antpos_axis": self.anteroposterior_axis,
            }
            elements = [
                "_".join(i)
                for i in product(["left", "right"], ["heel", "toe", "metatarsal_head"])
            ]
            elements += ["ground_reaction_force"]
            args.update(**{i: step.get(i) for i in elements})
            self._cycles += [RunningStep(**step.extra_signals, **args)]  # type: ignore

    def _find_cycles_kinetics(self):
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no ground reaction force data is available or no flight phases are found.
        """

        if self.ground_reaction_force is None:
            raise ValueError("no ground reaction force data available.")

        # get the grf and the latero-lateral COP
        time = self.ground_reaction_force.index
        cop = self.centre_of_pressure
        grf = self.vertical_force
        if cop is None or grf is None:
            raise RuntimeError("ground_reaction_force data not found.")
        vgrf = np.asarray(grf.data, float).flatten()
        cop_ml = np.asarray(cop[self.lateral_axis], float).flatten()

        # check if there are flying phases
        flights = vgrf <= self.ground_reaction_force_threshold
        if not any(flights):
            raise ValueError("No flight phases have been found on data.")

        # get the minimum reasonable contact time for each step
        fsamp = float(1 / np.mean(np.diff(time)))
        dsamples = int(round(fsamp / 4))

        # get the peaks in the normalized grf, then return toe-offs and foot
        # strikes
        grfn = vgrf / np.max(vgrf)
        toi = []
        fsi = []
        pks = find_peaks(grfn, 0.5, dsamples)
        for pk in pks:
            to = np.where(grfn[pk:] < self.height_threshold)[0]
            fs = np.where(grfn[:pk] < self.height_threshold)[0]
            if len(fs) > 0 and len(to) > 0:
                toi += [to[0] + pk]
                if len(toi) > 1:
                    fsi += [fs[-1]]
        toi = np.unique(toi)
        fsi = np.unique(fsi)

        # get the mean latero-lateral position of each contact
        contacts = [np.arange(i, j + 1) for i, j in zip(fsi, toi[1:])]
        pos = [np.nanmean(cop_ml[i]) for i in contacts]

        # get the mean value of alternated contacts and set the step sides
        # accordingly
        evens = np.mean(pos[0:-1:2])
        odds = np.mean(pos[1:-1:2])
        sides = []
        for i in np.arange(len(pos)):
            if evens < odds:
                sides += ["left" if i % 2 == 0 else "right"]
            else:
                sides += ["left" if i % 2 != 0 else "right"]

        for to, ed, side in zip(toi[:-1], toi[1:], sides):
            args = {
                "side": side,
                "ground_reaction_force_threshold": self.ground_reaction_force_threshold,
                "height_threshold": self.height_threshold,
                "algorithm": self.algorithm,
                "vertical_axis": self.vertical_axis,
                "antpos_axis": self.anteroposterior_axis,
            }
            step = self[self.index[to] : self.index[ed]]
            elements = [
                "_".join(i)
                for i in product(["left", "right"], ["heel", "toe", "metatarsal_head"])
            ]
            elements += ["ground_reaction_force"]
            args.update(**{i: step.get(i) for i in elements})
            self._cycles += [RunningStep(**step.extra_signals, **args)]  # type: ignore

    # * constructor

    def __init__(
        self,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
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
        process_inputs: bool = True,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a RunningTest instance.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}, optional
            Algorithm used for gait cycle detection. 'kinematics' uses marker data, 'kinetics' uses force platform data.
        left_heel, right_heel, left_toe, right_toe : Point3D or None, optional
            Marker data for the respective anatomical points.
        left_metatarsal_head : Point3D or None, optional
            Marker data for the left metatarsal head.
        right_metatarsal_head : Point3D or None, optional
            Marker data for the right metatarsal head.
        ground_reaction_force : ForcePlatform or None, optional
            Ground reaction force data.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force for contact detection.
        height_threshold : float or int, optional
            Maximum vertical height for contact detection.
        vertical_axis : {'X', 'Y', 'Z'}, optional
            The vertical axis.
        antpos_axis : {'X', 'Y', 'Z'}, optional
            The anterior-posterior axis.
        process_inputs : bool, optional
            If True, process the input data.
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
            process_inputs=process_inputs,
            **extra_signals,
        )
