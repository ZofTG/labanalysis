"""basic gait module

This module provides classes and utilities for gait analysis, including
GaitObject, GaitCycle, and GaitTest, which support kinematic and kinetic
cycle detection, event extraction, and biofeedback summary generation.
"""

#! IMPORTS


from typing import Literal

import numpy as np

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...signalprocessing import butterworth_filt, rms_filt
from ..processingpipeline import ProcessingPipeline
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from .gaitcycle import GaitCycle
from .gaitobject import GaitObject

#! CONSTANTS


__all__ = ["GaitExercise"]


#! CLASSESS


class GaitExercise(GaitObject):
    """
    Detect steps and strides from kinematic/kinetic data and extract
    biofeedback info.

    Parameters
    ----------
    algorithm : Literal['kinematics', 'kinetics'], optional
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
    vertical_axis : Literal['X', 'Y', 'Z'], optional
        The vertical axis.
    antpos_axis : Literal['X', 'Y', 'Z'], optional
        The anterior-posterior axis.
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals to include.
    """

    # * class variables

    _cycles: list[GaitCycle]

    # * attributes

    @property
    def cycles(self):
        """
        Get the detected gait cycles.

        Returns
        -------
        list of GaitCycle
        """
        return self._cycles

    @property
    def processing_pipeline(self):
        """
        Get the default processing pipeline for this test.

        Returns
        -------
        ProcessingPipeline
        """

        # emg
        def process_emg(channel: EMGSignal):
            channel -= channel.mean()
            fsamp = 1 / np.mean(np.diff(channel.index))
            channel.apply(
                butterworth_filt,
                fcut=[20, 450],
                fsamp=fsamp,
                order=4,
                ftype="bandpass",
                phase_corrected=True,
                inplace=True,
                axis=1,
            )
            channel.apply(
                rms_filt,
                order=int(0.2 * fsamp),
                pad_style="reflect",
                offset=0.5,
                inplace=True,
                axis=1,
            )
            return channel

        # points3d
        def process_point3d(point: Point3D):
            point.fillna(inplace=True)
            fsamp = 1 / np.mean(np.diff(point.index))
            point = point.apply(
                butterworth_filt,
                fcut=6,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
            )
            return point

        # forceplatforms
        def process_forceplatforms(fp: ForcePlatform):

            def process_signal3d(signal: Signal3D):
                signal.fillna(inplace=True, value=0)
                fsamp = 1 / np.mean(np.diff(signal.index))
                signal = signal.apply(
                    butterworth_filt,
                    fcut=[10, 100],
                    fsamp=fsamp,
                    order=4,
                    ftype="bandstop",
                    phase_corrected=True,
                )
                return signal

            force_platforms_processing_pipeline = ProcessingPipeline(
                point3d_funcs=[process_point3d],
                signal3d_funcs=[process_signal3d],
            )

            fp.apply(force_platforms_processing_pipeline, inplace=True)
            return fp

        return ProcessingPipeline(
            emgsignal_funcs=[process_emg],
            point3d_funcs=[process_point3d],
            forceplatform_funcs=[process_forceplatforms],
        )

    # * methods

    def _find_cycles_kinetics(self) -> None:
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _find_cycles_kinematics(self) -> None:
        """
        Find the gait cycles using the kinematics algorithm.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """
        Set the gait cycle detection algorithm.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            Algorithm label.
        """
        super().set_algorithm(algorithm)

        # update cycles
        self._cycles = []
        if self.algorithm == "kinetics":
            self._find_cycles_kinetics()
        elif self.algorithm == "kinematics":
            self._find_cycles_kinematics()

    # * constructors

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
        process_inputs: bool = True,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a GaitTest.

        Parameters
        ----------
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
        process_inputs: bool, optional
            If True, the ProcessPipeline integrated within this instance is
            applied. Otherwise raw data are retained.
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
        if not isinstance(process_inputs, bool):
            raise TypeError("'process_inputs' must be True or False.")
        if process_inputs:
            self.apply(self.processing_pipeline, inplace=True)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        left_heel: str | None = "lHeel",
        right_heel: str | None = "rHeel",
        left_toe: str | None = "lToe",
        right_toe: str | None = "rToe",
        left_metatarsal_head: str | None = "lMid",
        right_metatarsal_head: str | None = "rMid",
        ground_reaction_force: str | None = "fRes",
        ground_reaction_force_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
        strip: bool = True,
        reset_time: bool = True,
        process_inputs: bool = True,
    ):
        """
        Generate a GaitTest object directly from a .tdf file.

        Parameters
        ----------
        file : str
            Path to a ".tdf" file.
        algorithm : {'kinematics', 'kinetics'}, optional
            The cycle detection algorithm.
        left_heel : str or None, optional
            Name of the left heel marker in the tdf file.
        right_heel : str or None, optional
            Name of the right heel marker in the tdf file.
        left_toe : str or None, optional
            Name of the left toe marker in the tdf file.
        right_toe : str or None, optional
            Name of the right toe marker in the tdf file.
        left_metatarsal_head : str or None, optional
            Name of the left metatarsal head marker in the tdf file.
        right_metatarsal_head : str or None, optional
            Name of the right metatarsal head marker in the tdf file.
        ground_reaction_force : str or None, optional
            Name of the ground reaction force data in the tdf file.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force for contact detection.
        height_threshold : float or int, optional
            Maximum vertical height for contact detection.
        vertical_axis : {'X', 'Y', 'Z'}, optional
            The vertical axis.
        antpos_axis : {'X', 'Y', 'Z'}, optional
            The anterior-posterior axis.
        process_inputs: bool, optional
            If True, the ProcessPipeline integrated within this instance is
            applied. Otherwise raw data are retained.

        Returns
        -------
        GaitTest
        """
        record = super().from_tdf(file)
        return cls(
            algorithm=algorithm,
            left_heel=record.get(left_heel),
            right_heel=record.get(right_heel),
            left_toe=record.get(left_toe),
            right_toe=record.get(right_toe),
            left_metatarsal_head=record.get(left_metatarsal_head),
            right_metatarsal_head=record.get(right_metatarsal_head),
            ground_reaction_force=record.get(ground_reaction_force),
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,
            strip=strip,
            reset_time=reset_time,
            process_inputs=process_inputs,
        )
