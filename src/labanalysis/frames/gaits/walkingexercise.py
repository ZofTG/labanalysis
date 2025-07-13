"""
Module for running test analysis.

This module defines classes for performing running test analysis,
including step detection and summary plots.
"""

#! IMPORTS


from typing import Literal

import numpy as np
import pandas as pd

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...signalprocessing import find_peaks, psd
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from .gaitexercise import GaitExercise
from .walkingstride import WalkingStride

__all__ = ["WalkingExercise"]


#! CLASSESS


class WalkingExercise(GaitExercise):
    """
    Represents a walking test.

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
            If any required marker is missing or no toe-offs have been found.
        """

        # get toe-off times
        time = self.index
        fsamp = float(1 / np.mean(np.diff(time)))
        for lbl in ["left_toe", "right_toe"]:

            # get the vertical coordinates of the toe markers
            arr = np.asarray(self[self.vertical_axis], float).flatten()

            # filter and rescale
            ftoe = arr / np.max(arr)

            # get the minimum reasonable contact time for each step
            frq, pwr = psd(ftoe, fsamp)
            ffrq = frq[np.argmax(pwr)]
            dsamples = int(round(fsamp / ffrq / 2))

            # get the peaks at each cycle
            pks = find_peaks(ftoe, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            tos = []
            side = lbl.split("_")[0]
            for pk in pks:
                idx = np.where(ftoe[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    line = pd.Series({"time": time[idx[-1]], "side": side})
                    tos += [pd.DataFrame(line).T]

            # wrap the events
            if len(tos) == 0:
                raise ValueError("no toe-offs have been found.")
            tos = pd.concat(tos, ignore_index=True)
            tos = tos.drop_duplicates()
            tos = tos.sort_values("time")
            tos = tos.reset_index(drop=True)

            # check the alternation of the steps
            for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
                t0 = float(tos.time.values[i0])
                t1 = float(tos.time.values[i1])
                sub = self[t0:t1]
                stride = WalkingStride(
                    side,  # type: ignore
                    sub.algorithm,
                    sub.get("left_heel"),
                    sub.get("right_heel"),
                    sub.get("left_toe"),
                    sub.get("right_toe"),
                    sub.get("left_metatarsal_head"),
                    sub.get("right_metatarsal_head"),
                    sub.get("ground_reaction_force"),
                    sub.ground_reaction_force_threshold,
                    sub.height_threshold,
                    sub.vertical_axis,
                    sub.antpos_axis,  # type: ignore
                    **sub.extra_signals,  # type: ignore
                )
                self._cycles += [stride]

        # sort the cycles
        cycle_index = np.argsort([i.init_s for i in self.cycles])
        self._cycles = [self.cycles[i] for i in cycle_index]

    def _find_cycles_kinetics(self):
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ground_reaction_force not found.
        """

        # get the relevant data
        res = self.ground_reaction_force
        if res is None:
            raise ValueError("ground_reaction_force not found")
        time = res.index
        res_ap = np.asarray(res["force"][self.anteroposterior_axis], float).flatten()
        res_ap -= np.nanmean(res_ap)

        # get the dominant frequency
        fsamp = float(1 / np.mean(np.diff(time)))
        frq, pwr = psd(res_ap, fsamp)
        ffrq = frq[np.argmax(pwr)]

        # find peaks
        min_samp = int(fsamp / ffrq / 2)
        pks = find_peaks(res_ap, 0, min_samp)

        # for each peak pair get the range and obtain the toe-off
        # as the last value occurring before the peaks within the
        # 1 - height_threshold of that range
        toi = []
        for pk in pks:
            thresh = (1 - self.height_threshold) * res_ap[pk]
            loc = np.where(res_ap[:pk] < thresh)[0]
            if len(loc) > 0:
                toi += [loc[-1]]

        # get the latero-lateral centre of pressure
        cop = self.centre_of_pressure
        cop_ml = np.asarray(cop[self.lateral_axis], float).flatten()
        cop_ml -= np.nanmean(cop_ml)

        # get the sin function best fitting the cop_ml
        def _sin_fitted(arr: np.ndarray):
            """fit a sine over arr"""
            rfft = np.fft.rfft(arr - np.mean(arr))
            pwr = psd(arr)[1]
            rfft[pwr < np.max(pwr)] = 0
            return np.fft.irfft(rfft, len(arr))

        sin_ml = self._sin_fitted(cop_ml)

        # get the mean latero-lateral position of each toe-off interval
        cnt = [np.arange(i, j + 1) for i, j in zip(toi[:-1], toi[1:])]
        pos = [np.nanmean(sin_ml[i]) for i in cnt]

        # get the sides
        sides = ["left" if i > 0 else "right" for i in pos]

        # generate the steps
        toi_evens = toi[0:-1:2]
        sides_evens = sides[0:-1:2]
        toi_odds = toi[1:-1:2]
        sides_odds = sides[1:-1:2]
        for ti, si in zip([toi_evens, toi_odds], [sides_evens, sides_odds]):
            for to, ed, side in zip(ti[:-1], ti[1:], si):
                sub = self[self.index[to] : self.index[ed]]
                stride = WalkingStride(
                    side,  # type: ignore
                    sub.algorithm,
                    sub.get("left_heel"),
                    sub.get("right_heel"),
                    sub.get("left_toe"),
                    sub.get("right_toe"),
                    sub.get("left_metatarsal_head"),
                    sub.get("right_metatarsal_head"),
                    sub.get("ground_reaction_force"),
                    sub.ground_reaction_force_threshold,
                    sub.height_threshold,
                    sub.vertical_axis,
                    sub.antpos_axis,  # type: ignore
                    **sub.extra_signals,  # type: ignore
                )
                self._cycles += [stride]

        # sort the cycles
        idx = np.argsort([i.init_s for i in self._cycles])
        self._cycles = [self._cycles[i] for i in idx]

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
        Initialize a WalkingTest instance.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}, optional
            Algorithm used for gait cycle detection. 'kinematics' uses marker data, 'kinetics' uses force platform data.
        left_heel : Point3D or None, optional
            Marker data for the left heel.
        right_heel : Point3D or None, optional
            Marker data for the right heel.
        left_toe : Point3D or None, optional
            Marker data for the left toe.
        right_toe : Point3D or None, optional
            Marker data for the right toe.
        left_metatarsal_head : Point3D or None, optional
            Marker data for the left metatarsal head.
        right_metatarsal_head : Point3D or None, optional
            Marker data for the right metatarsal head.
        grf : ForcePlatform or None, optional
            Ground reaction force data.
        grf_threshold : float or int, optional
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
