"""Squat Jump Test module"""

#! IMPORTS


from typing import Iterable, Literal
import numpy as np
import pandas as pd

from ..frames import StateFrame
from ..statictests import StaticUprightStance
from .counter_movement_jump import CounterMovementJump, CounterMovementJumpTest

__all__ = ["SideJump", "SideJumpTest"]


#! CLASSES


class SideJump(CounterMovementJump):
    """
    class defining a single side jump collected by markers, forceplatforms
    and (optionally) emg signals.

    Parameters
    ----------
    markers: pd.DataFrame
        a DataFrame being composed by:
            * one or more triplets of columns like:
                | <NAME> | <NAME> | <NAME> |
                |    X   |   Y    |    Z   |
                |    m   |   m    |    m   |
            * the time instant of each sample in seconds as index.

    forceplatforms: pd.DataFrame
        a DataFrame being composed by:
            * one or more packs of columns like:
                | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> | <NAME> |
                | ORIGIN | ORIGIN | ORIGIN |  FORCE | FORCE  | FORCE  | TORQUE | TORQUE | TORQUE |
                |    X   |   Y    |    Z   |    X   |   Y    |    Z   |    X   |   Y    |    Z   |
                |    m   |   m    |    m   |    N   |   N    |    N   |    Nm  |   Nm   |   Nm   |
            * the time instant of each sample in seconds as index.

    emgs: pd.DataFrame
        a DataFrame being composed by:
            * one or more packs of columns like:
                | <NAME> |
                |    V   |
            * the time instant of each sample in seconds as index.


    Attributes
    ----------
    markers
        the kinematic data

    forceplatforms
        the force data

    emgs
        the EMG data

    emg_processing_options
        the parameters to set the filtering of the EMG signal

    forceplatform_processing_options
        the parameters to set the filtering of the force signal

    marker_processing_options
        the parameters to set the filtering of the kinematic signals

    eccentric_phase
        a StateFrame representing the eccentric phase of the jump

    concentric_phase
        a StateFrame representing the concentric phase of the jump

    flight_phase
        a StateFrame representing the flight phase of the jump

    loading_response_phase
        a StateFrame representing the loading response phase of the jump

    rate_of_force_development
        return the rate of force development over the concentric phase of the
        jump

    velocity_at_toeoff
        return the vertical velocity at the toeoff in m/sù

    concentric_power
        return the mean power in W generated during the concentric phase

    jump_height
        return the height of the jump in cm

    side
        return the side of the jump

    Methods
    -------
    to_dataframe
        return the available data as single pandas DataFrame.

    to_stateframe
        return the available data as StateFrame.

    copy
        return a copy of the object.

    slice
        return a subset of the object.

    process_data
        process internal data to remove/replace missing values and smooth the
        signals.

    is_processed
        returns True if the actual object already run the process data method

    to_reference_frame
        rotate the actual object to a new reference frame defined by
        the provided origin and axes.

    resize
        resize the available data to the relevant phases of the jump.
    """

    # * class variables

    _side: Literal["Right", "Left"]

    # * attributes

    @property
    def side(self):
        """return the side of the jump"""
        return self._side

    # * methods

    def copy(self):
        """create a copy of the object"""
        return self.from_stateframe(self, self.side)

    # * constructors

    def __init__(
        self,
        markers_raw: pd.DataFrame,
        forceplatforms_raw: pd.DataFrame,
        emgs_raw: pd.DataFrame,
        side: Literal["Left", "Right"],
        process_data: bool = True,
        ignore_index: bool = True,
        markers_fcut: int | float | None = 6,
        forces_fcut: int | float | None = 100,
        emgs_fband: tuple[int | float, int | float] | None = (30, 400),
        emgs_rms_win: int | float | None = 0.2,
    ):
        """
        generate an instance of a Side Jump object

        Parameters
        ----------
        markers_raw: pd.DataFrame
            a dataframe containing raw markers data.

        forceplatforms_raw: pd.DataFrame
            a raw dataframe containing raw forceplatforms data.

        emgs_raw: pd.DataFrame
            a raw dataframe containing raw emg data.

        side: Literal['Left', 'Right']
            the side of the jump.

        process_data: bool = True
            if True, process the data according to the options provided below

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
            1. the data in between the start and end of the marker data are
            retained.
            2. missing values in the middle of the data are replaced by zeros
            3. Force and Torque data are low-pass filtered by means of a
            lowpass, Butterworth filter with the entered force options.
            4. Force vector origin's coordinates are low-pass filtered by means
            of a lowpass, Butterworth filter with the entered marker options.

        EMGs (optional)
            1. the data in between the start and end of the markers are
            retained.
            2. the signals are bandpass filtered with the provided emg options
            3. the root-mean square filter with the given time window is
            applied to get the envelope of the signals.

        All
            1. if 'ignore_index=True' then the time indices of all components is
            adjusted to begin with zero.
        """
        super().__init__(
            markers_raw=markers_raw,
            forceplatforms_raw=forceplatforms_raw,
            emgs_raw=emgs_raw,
            process_data=process_data,
            ignore_index=ignore_index,
            markers_fcut=markers_fcut,
            forces_fcut=forces_fcut,
            emgs_fband=emgs_fband,
            emgs_rms_win=emgs_rms_win,
        )
        if not isinstance(side, (Literal, str)):
            raise ValueError("'side' must be 'Left' or 'Right'.")
        self._side = side

    @classmethod
    def from_tdf_file(
        cls,
        file: str,
        side: Literal["Left", "Right"],
        process_data: bool = True,
        ignore_index: bool = True,
        markers_fcut: int | float | None = 6,
        forces_fcut: int | float | None = 100,
        emgs_fband: tuple[int | float, int | float] | None = (30, 400),
        emgs_rms_win: int | float | None = 0.2,
    ):
        """
        generate a SideJump from a .tdf file

        Parameters
        ----------
        file : str
            a valid .tdf file containing (tracked) markers, force platforms and
            (optionally) EMG data

        side: Literal['Left', 'Right']
            the side of the jump

        process_data: bool = True
            if True, process the data according to the options provided below

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

        Returns
        -------
        frame: SideJump
            a SideJump instance of the data contained in the .tdf file.

        Processing procedure
        --------------------

        Markers
            1. missing values at the beginning and end of the data are removed
            2. missing values in the middle of the trial are replaced by cubic
            spline interpolation
            3. the data are low-pass filtered by means of a lowpass, Butterworth
            filter with the entered marker options

        Force Platforms
            1. the data in between the start and end of the marker data are
            retained.
            2. missing values in the middle of the data are replaced by zeros
            3. Force and Torque data are low-pass filtered by means of a
            lowpass, Butterworth filter with the entered force options.
            4. Force vector origin's coordinates are low-pass filtered by means
            of a lowpass, Butterworth filter with the entered marker options.

        EMGs (optional)
            1. the data in between the start and end of the markers are
            retained.
            2. the signals are bandpass filtered with the provided emg options
            3. the root-mean square filter with the given time window is
            applied to get the envelope of the signals.

        All
            1. if 'ignore_index=True' then the time indices of all components is
            adjusted to begin with zero.
        """
        obj = StateFrame.from_tdf_file(file=file)
        if not isinstance(process_data, bool):
            raise ValueError("'process_data' must be a boolean")
        if process_data:
            obj.process_data(
                inplace=True,
                ignore_index=ignore_index,
                markers_fcut=markers_fcut,
                forces_fcut=forces_fcut,
                emgs_fband=emgs_fband,
                emgs_rms_win=emgs_rms_win,
            )
        return cls.from_stateframe(obj, side)

    @classmethod
    def from_stateframe(cls, obj: StateFrame, side: Literal["Left", "Right"]):
        """
        generate a CounterMovementJump from a StateFrame object

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        side: Literal['Left', 'Right']
            the test side

        Returns
        -------
        an instance of the jump.
        """
        # check the input
        if not isinstance(obj, StateFrame):
            raise ValueError("obj must be a StateFrame object.")

        # create the object instance
        out = cls(
            markers_raw=obj.markers,
            forceplatforms_raw=obj.forceplatforms,
            emgs_raw=obj.emgs,
            side=side,
            process_data=False,
        )
        out._processed = obj.is_processed()
        out._marker_processing_options = obj.marker_processing_options
        out._forceplatform_processing_options = obj.forceplatform_processing_options
        out._emg_processing_options = obj.emg_processing_options

        return out


class SideJumpTest(CounterMovementJumpTest):
    """
    Class handling the data processing and analysis of the collected data about
    a side jump test.

    Parameters
    ----------
    baseline: StaticUprightStance
        a StaticUprightStance instance defining the baseline acquisition.

    left_jumps: Iterable[SideJump]
        a variable number of SideJump objects

    right_jumps: Iterable[SideJump]
        a variable number of SideJump objects

    Attributes
    ----------
    baseline
        the StaticUprightStance instance of the test

    left_jumps
        the list of available (left) SideJump objects.

    right_jumps
        the list of available (right) SideJump objects.

    summary_table
        A table with summary statistics about the test. The table
        includes the following elements:
            * jump height
            * concentric power
            * rate of force development
            * velocity at toeoff
            * muscle symmetry (for each tested muscle)

    summary_plot
        a plotly FigureWidget summarizing the results of the test
    """

    # * class variables

    _left_jumps: list[SideJump]
    _right_jumps: list[SideJump]

    # * methods

    def _check_valid_inputs(self):
        # check the baseline
        if not isinstance(self._baseline, StaticUprightStance):
            raise ValueError("baseline must be a StaticUprightStance instance.")

        # check for the jumps
        if not isinstance(self._left_jumps, Iterable):
            raise ValueError("'left_jumps' must be a list of SideJump objects.")
        self._left_jumps = list(self._left_jumps)
        if not isinstance(self._right_jumps, Iterable):
            raise ValueError("'right_jumps' must be a list of SideJump objects.")
        self._right_jumps = list(self._right_jumps)
        for jump in self._left_jumps + self._right_jumps:
            if not isinstance(jump, SideJump):
                msg = f"All jumps must be SideJump instances."
                raise ValueError(msg)

    # * constructors

    def __init__(
        self,
        baseline: StaticUprightStance,
        left_jumps: list[SideJump],
        right_jumps: list[SideJump],
    ):
        self._baseline = baseline
        self._left_jumps = left_jumps
        self._right_jumps = right_jumps
        self._check_valid_inputs()
