"""Squat Jump Test module"""

#! IMPORTS


import numpy as np
import pandas as pd

from ... import signalprocessing as sp
from ..frames import StateFrame
from ..statictests import StaticUprightStance
from .squat_jump import SquatJump, SquatJumpTest

__all__ = ["CounterMovementJump", "CounterMovementJumpTest"]


#! CLASSES


class CounterMovementJump(SquatJump):
    """
    class defining a single CounterMovementJump collected by markers,
    forceplatforms and (optionally) emg signals.

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

    # * attributes

    @property
    def eccentric_phase(self):
        """
        return a StateFrame denoting the eccentric phase of the jump

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the concentric
            phase of the jump

        Procedure
        ---------
            1. define 'time_end' as the time instant corresponding to the start
            of the concentric phase. Please looka the 'concentric_phase'
            documentation to have a detailed description of the procedure used
            to extract this phase.
            2. look at the last positive speed value in the vertical S2 signal
            occurring before 'time_end'.
            3. define 'time_end' as the last peak in the grf occurring before
            the time defined in 2.
        """
        # get the time instant corresponding to the start of the concentric
        # phase
        t_end = self.to_dataframe().index.to_numpy()
        t_end = t_end[t_end < self.concentric_phase.to_dataframe().index[0]]
        t_end = float(round(t_end[-1], 3))

        # get the vertical S2 velocity
        s2y = self.markers.S2.Y.values.astype(float).flatten()
        s2t = self.markers.index.to_numpy()
        s2v = sp.winter_derivative1(s2y, s2t)
        s2y = s2y[1:-1]
        s2t = s2t[1:-1]

        # look at the last positive vertical speed value occuring before t_end
        batches = sp.continuous_batches(s2v[s2t < t_end] <= 0)
        if len(batches) == 0:
            raise RuntimeError("No eccentric phase has been found.")
        s2y_0 = float(round(s2t[batches[-1][0]], 3))  # type: ignore

        # take the last peak in vertical grf occurring before s2y_0
        grfy = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grft = self.forceplatforms.index.to_numpy()
        idx = np.where(grft < s2y_0)[0]
        grf_pks = sp.find_peaks(grfy[idx])
        if len(grf_pks) == 0:
            t_start = float(round(grft[0], 3))
        else:
            t_start = float(round(grft[grf_pks[-1]], 3))  # type: ignore

        # get the time corresponding phase
        return self.slice(t_start, t_end)

    # * methods

    def copy(self):
        """create a copy of the object"""
        return super().from_stateframe(self)

    def resize(
        self,
        extra_time_window: float | int = 0.2,
        inplace: bool = True,
    ):
        """
        resize the available data to the relevant phases of the jump.

        This function removes the data at the beginning and at the end of the
        jump leaving just the selected 'extra_time_window' at both sides.
        The jump is assumed to start at the beginning of the 'eccentric_phase'
        and to end when the 'loading_response_phase' is concluded.

        Parameters
        ----------
        extra_time_window : float | int (default = 0.2)
            the extra time allowed at both sides of the available data that is
            retained from resizing.

        inplace: bool (default = True)
            if True, the function resizes the current jump instance. Otherwise
            it returns a resized copy.

        Returns
        -------
        if 'inplace=True', it returns nothing. Otherwise a new instance of
        SquatJump is returned.
        """
        # check the input data
        if not isinstance(extra_time_window, (int, float)):
            raise ValueError("'extra_time_window' has to be an int or float.")
        if not isinstance(inplace, bool):
            raise ValueError("'inplace' has to be a boolean.")

        # set the start and end of the test
        t_start = self.eccentric_phase.to_dataframe().index.to_numpy()[0]
        t_start = max(t_start - extra_time_window, 0)
        t_end = self.loading_response_phase.to_dataframe().index.to_numpy()[-1]
        t_last = self.to_dataframe().index.to_numpy()[-1]
        t_end = min(t_end + extra_time_window, t_last)

        # handle the inplace option
        if not inplace:
            return self.from_stateframe(self.slice(t_start, t_end))

        # markers
        mrk_idx = self.markers.index.to_numpy()
        mrk_loc = np.where((mrk_idx >= t_start) & (mrk_idx <= t_end))[0]
        self._markers = self._markers.iloc[mrk_loc]

        # forceplatforms
        fps_idx = self.forceplatforms.index.to_numpy()
        fps_loc = np.where((fps_idx >= t_start) & (fps_idx <= t_end))[0]
        self._forceplatforms = self._forceplatforms.iloc[fps_loc]

        # emgs
        if self.emgs.shape[0] > 0:
            emg_idx = self.emgs.index.to_numpy()
            emg_loc = np.where((emg_idx >= t_start) & (emg_idx <= t_end))[0]
            self._emgs = self._emgs.iloc[emg_loc]

    # * constructors

    def __init__(
        self,
        markers_raw: pd.DataFrame,
        forceplatforms_raw: pd.DataFrame,
        emgs_raw: pd.DataFrame,
        process_data: bool = True,
        ignore_index: bool = True,
        markers_fcut: int | float | None = 6,
        forces_fcut: int | float | None = 100,
        emgs_fband: tuple[int | float, int | float] | None = (30, 400),
        emgs_rms_win: int | float | None = 0.2,
    ):
        """
        generate an instance of a Squat Jump object

        Parameters
        ----------
        markers_raw: pd.DataFrame
            a dataframe containing raw markers data.

        forceplatforms_raw: pd.DataFrame
            a raw dataframe containing raw forceplatforms data.

        emgs_raw: pd.DataFrame
            a raw dataframe containing raw emg data.

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

    @classmethod
    def from_tdf_file(
        cls,
        file: str,
        process_data: bool = True,
        ignore_index: bool = True,
        markers_fcut: int | float | None = 6,
        forces_fcut: int | float | None = 100,
        emgs_fband: tuple[int | float, int | float] | None = (30, 400),
        emgs_rms_win: int | float | None = 0.2,
    ):
        """
        generate a SquatJump from a .tdf file

        Parameters
        ----------
        file : str
            a valid .tdf file containing (tracked) markers, force platforms and
            (optionally) EMG data


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
        frame: SquatJump
            a SquatJump instance of the data contained in the .tdf file.

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
        return super().from_tdf_file(
            file=file,
            process_data=process_data,
            ignore_index=ignore_index,
            markers_fcut=markers_fcut,
            forces_fcut=forces_fcut,
            emgs_fband=emgs_fband,
            emgs_rms_win=emgs_rms_win,
        )

    @classmethod
    def from_stateframe(cls, obj: StateFrame):
        """
        generate a CounterMovementJump from a StateFrame object

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        Returns
        -------
        frame: CounterMovementJump
            a CounterMovementJump instance.
        """
        return super().from_stateframe(obj)


class CounterMovementJumpTest(SquatJumpTest):
    """
    Class handling the data processing and analysis of the collected data about
    a counter movement jump test.

    Parameters
    ----------
    baseline: StaticUprightStance
        a StaticUprightStance instance defining the baseline acquisition.

    *jumps: CounterMovementJump
        a variable number of jump objects

    Attributes
    ----------
    baseline
        the StaticUprightStance instance of the test

    jumps
        the list of available SquatJump objects.

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

    _jumps: list[CounterMovementJump]

    # * methods

    def _check_valid_inputs(self):
        # check the baseline
        if not isinstance(self._baseline, StaticUprightStance):
            raise ValueError("baseline must be a StaticUprightStance instance.")

        # check for the jumps
        for i, jump in enumerate(self._jumps):
            if not isinstance(jump, CounterMovementJump):
                msg = f"jump {i + 1} is not a CounterMovementJump instance."
                raise ValueError(msg)

    # * constructors

    def __init__(
        self,
        baseline: StaticUprightStance,
        *jumps: CounterMovementJump,
    ):
        super().__init__(baseline, *jumps)
