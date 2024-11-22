"""Counter Movement Jump Test module"""

#! IMPORTS


from typing import Iterable

import numpy as np
import pandas as pd

from ... import signalprocessing as sp
from ..posturaltests.upright import UprightStance
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

    grf
        return the vertical ground reaction force

    eccentric_phase
        a StateFrame representing the eccentric phase of the jump

    concentric_phase
        a StateFrame representing the concentric phase of the jump

    flight_phase
        a StateFrame representing the flight phase of the jump

    loading_response_phase
        a StateFrame representing the loading response phase of the jump

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

        # get the last peak in vertical position before the concentric phase
        s2y = self.markers.S2.Y.values.astype(float).flatten()
        s2t = self.markers.index.to_numpy()
        """
        fsamp = float(1 / np.mean(np.diff(s2t)))
        s2f = sp.butterworth_filt(s2y, 1, fsamp, 6, "lowpass", True)
        pks = s2t[sp.find_peaks(s2f)]
        if len(pks) == 0:
            raise RuntimeError("No eccentric phase has been found.")
        t_start = float(pks[-1])
        """
        # look at the last positive vertical speed value occuring before t_end
        s2v = sp.winter_derivative1(s2y)
        s2t = s2t[1:-1]
        batches = sp.continuous_batches(s2v[s2t < t_end] <= 0)
        if len(batches) == 0:
            raise RuntimeError("No eccentric phase has been found.")
        s2y_0 = float(round(s2t[batches[-1][0]], 3))  # type: ignore

        # take the last peak in vertical grf occurring before s2y_0
        grfy = self.grf.values.astype(float).flatten()
        grft = self.grf.index.to_numpy()
        idx = np.where(grft < s2y_0)[0]
        grf_pks = sp.find_peaks(grfy[idx])
        if len(grf_pks) == 0:
            t_start = float(round(grft[0], 3))
        else:
            t_start = float(round(grft[grf_pks[-1]], 3))  # type: ignore

        # get the time corresponding phase
        return self.slice(t_start, t_end)

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


class CounterMovementJumpTest(SquatJumpTest):
    """
    Class handling the data processing and analysis of the collected data about
    a counter movement jump test.

    Parameters
    ----------
    baseline: StaticUprightStance
        a StaticUprightStance instance defining the baseline acquisition.

    *jumps: CounterMovementJump
        a variable number of CounterMovementJump objects

    Attributes
    ----------
    baseline
        the StaticUprightStance instance of the test

    jumps
        the list of available CounterMovementJump objects.

    Methods
    -------
    results
        return a plotly figurewidget highlighting the resulting data
        and a table with the resulting outcomes as pandas DataFrame.

    summary
        return a plotly bar plot highlighting the test summary and a table
        reporting the summary data.
    """

    # * class variables

    _jumps: list[CounterMovementJump]

    # * attributes

    @property
    def jumps(self):
        """return the jumps contained in the test"""
        return self._jumps

    def _make_results_table(self):
        raw = super()._make_results_table()
        new = []
        phase_col = ("Phase", "", "", "", "")
        time_col = ("Time", "", "", "", "")
        jump_col = ("Jump", "", "", "", "")
        for i, jump in enumerate(self.jumps):
            dfe = jump.eccentric_phase.to_dataframe().dropna()
            dfe.insert(0, phase_col, np.tile("Eccentric", dfe.shape[0]))
            jmp = f"Jump {i + 1}"
            lbl = np.tile(jmp, dfe.shape[0])
            dfe.insert(0, jump_col, lbl)
            time = dfe.index.to_numpy() - dfe.index[0]
            dfe.insert(0, time_col, time)
            dfe = self._simplify_table(dfe)
            dfr = raw.loc[raw.Jump == jmp]
            offset = round(float(np.mean(np.diff(time)) + time[-1]), 3)
            dfr.loc[dfr.index, ["Time"]] += offset  # type: ignore
            new += [dfe, dfr]
        new = pd.concat(new, ignore_index=True)
        return new.sort_values("Time")

    # * methods

    def _check_valid_inputs(self):
        # check the baseline
        if not isinstance(self._baseline, UprightStance):
            raise ValueError("baseline must be a StaticUprightStance instance.")

        # check for the jumps
        if not isinstance(self._jumps, Iterable):
            msg = "'jumps' must be a list of CounterMovementJump objects."
            raise ValueError(msg)
        for i, jump in enumerate(self._jumps):
            if not isinstance(jump, CounterMovementJump):
                msg = f"jump {i + 1} is not a CounterMovementJump instance."
                raise ValueError(msg)

    # * constructors

    def __init__(
        self,
        baseline: UprightStance,
        jumps: list[CounterMovementJump],
    ):
        super().__init__(baseline, jumps)  # type: ignore
