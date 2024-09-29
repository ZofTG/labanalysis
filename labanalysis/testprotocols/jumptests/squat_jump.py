"""Squat Jump Test module"""

#! IMPORTS


from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ... import signalprocessing as sp
from ...constants import G
from ..base import LabTest
from ..frames import StateFrame
from ..posturaltests.upright import UprightStance

__all__ = ["SquatJump", "SquatJumpTest", "get_jump_features"]


#! CLASSES


class SquatJump(StateFrame):
    """
    class defining a single SquatJump collected by markers, forceplatforms
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

    grf
        return the vertical ground reaction force

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
    def grf(self):
        """return the grf"""
        grfy = self.forceplatforms.lFoot.FORCE.Y.values.astype(float).flatten()
        grfy += self.forceplatforms.rFoot.FORCE.Y.values.astype(float).flatten()
        grf = pd.Series(grfy, index=self.forceplatforms.index.to_numpy())
        return grf.astype(float)

    @property
    def concentric_phase(self):
        """
        return a StateFrame representing the concentric phase of the jump

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the concentric
            phase of the jump

        Procedure
        ---------
            1. get the longest countinuous batch with positive acceleration
            of S2 occurring before con_end.
            2. define 'con_start' as the last local minima in the vertical grf
            occurring before the beginning of the batch defined in 2.
            3. define 'con_end' as the end of the concentric phase as the time
            instant immediately before the flight phase. Please look at the
            concentric_phase documentation to have a detailed view about how
            it is detected.
        """
        # take the end of the concentric phase as the time instant immediately
        # before the flight phase
        flight_start = self.flight_phase.to_dataframe().index.to_numpy()[0]
        time = self.to_dataframe().index.to_numpy()
        con_end = float(round(time[time < flight_start][-1], 3))

        # get the longest countinuous batch with positive acceleration of S2
        # occurring before con_end
        s2y = self.markers.S2.Y.values.astype(float).flatten()
        s2t = self.markers.index.to_numpy()
        s2a = sp.winter_derivative2(s2y, s2t)
        s2t = s2t[1:-1]
        batches = sp.continuous_batches((s2a > 0) & (s2t < con_end))
        if len(batches) == 0:
            raise RuntimeError("No concentric phase has been found")
        batch = batches[np.argmax([len(i) for i in batches])]

        # take the last local minima in the vertical grf occurring before
        # start_s2
        start_s2 = float(round(s2t[batch[0]], 3))  # type: ignore
        grfy = self.grf.values.astype(float).flatten()
        grft = self.grf.index.to_numpy()
        mins = grft[sp.find_peaks(-grfy[grft < start_s2])]
        if len(mins) == 0:
            con_start = start_s2
        else:
            con_start = float(round(mins[-1], 3))

        # return a slice of the available data
        return self.slice(con_start, con_end)

    @property
    def flight_phase(self):
        """
        return a StateFrame representing the flight phase of the jump

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the flight
            phase of the jump

        Procedure
        ---------
            1. get the longest batch with grf lower than 30N.
            2. define 'flight_start' as the first local minima occurring after
            the start of the detected batch.
            3. define 'flight_end' as the last local minima occurring before the
            end of the detected batch.
        """

        # get the longest batch with grf lower than 30N
        grfy = self.grf.values.astype(float)
        grft = self.grf.index.to_numpy()
        batches = sp.continuous_batches(grfy <= 30)
        msg = "No flight phase found."
        if len(batches) == 0:
            raise RuntimeError(msg)
        batch = batches[np.argmax([len(i) for i in batches])]

        # check the length of the batch is at minimum 2 samples
        if len(batch) < 2:
            raise RuntimeError(msg)

        # get the local minima in grfy
        mins = sp.find_peaks(-grfy)

        # look at the first local minima occurring after the start of the
        # detected batch
        time_start = float(round(grft[mins[mins > batch[0]]][0], 3))

        # look at the last local minima occurring befor the end of the
        # detected batch
        time_stop = float(round(grft[mins[mins < batch[-1]]][-1], 3))

        # return a slice of the available data
        return self.slice(time_start, time_stop)

    @property
    def loading_response_phase(self):
        """
        return the loading response phase, i.e. the phase during which the
        load is absobed and dissipated after landing.

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the loading
            response phase of the jump

        Procedure
        ---------
            1. define 'time_start' as the time instant immediately after the
            end of the flight phase. Please refer to the 'flight_phase'
            documentation to have a detailed view about how it is detected
            2. look at the longest continuous batch in the vertical velocity of
            S2 with positive sign which occurs after time_start.
            3. define 'time_end' as the time of the last element of the batch.
        """
        # take the time instant immediately after the end of the flight phase
        time = self.to_dataframe().index.to_numpy()
        flight_time = self.flight_phase.to_dataframe().index.to_numpy()
        time = time[time > flight_time[-1]]
        time_start = float(round(time[0], 3))

        # look at the longest continuous batch in the vertical velocity of
        # S2 with positive sign which occurs after time_start
        s2y = self.markers.S2.Y.values.astype(float).flatten()
        s2t = self.markers.index.to_numpy()
        s2v = sp.winter_derivative1(s2y, s2t)
        s2t = s2t[1:-1]
        batches = sp.continuous_batches((s2v > 0) & (s2t > time_start))
        if len(batches) == 0:
            raise RuntimeError("No loading response phase has been found")
        batch = batches[np.argmax([len(i) for i in batches])]

        # take the time of the last element of the batch
        time_end = float(round(s2t[batch[-1]], 3))  # type: ignore

        # return the range
        return self.slice(time_start, time_end)

    # * methods

    def to_stateframe(self):
        """return the actual object as StateFrame"""
        return super().copy()

    def copy(self):
        """create a copy of the object"""
        return self.from_stateframe(self)

    def resize(
        self,
        extra_time_window: float | int = 0.2,
        inplace: bool = True,
    ):
        """
        resize the available data to the relevant phases of the jump.

        This function removes the data at the beginning and at the end of the
        jump leaving just the selected 'extra_time_window' at both sides.
        The jump is assumed to start at the beginning of the 'concentric_phase'
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
        t_start = self.concentric_phase.to_dataframe().index.to_numpy()[0]
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

    def _check_inputs(self):
        """check the validity of the entered data"""
        # ensure that the 'rFoot' and 'lFoot' force platform objects exist
        lbls = np.unique(self.forceplatforms.columns.get_level_values(0))
        required_fp = ["lFoot", "rFoot"]
        for lbl in required_fp:
            if not any([i == lbl for i in lbls]):
                msg = f"the data does not contain the required '{lbl}'"
                msg += " forceplatform object."
                raise ValueError(msg)
        self._forceplatforms = self._forceplatforms[required_fp]

        # ensure that the 'S2' marker exists
        lbls = np.unique(self.markers.columns.get_level_values(0))
        if not any([i == "S2" for i in lbls]):
            msg = "the data does not contain the 'S2' marker."
            raise ValueError(msg)
        self._markers = self._markers[["S2"]]

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
        )

        # check the inputs are valid
        self._check_inputs()
        if not isinstance(process_data, bool):
            raise ValueError("'process_data' must be a boolean object.")

        # process the data if required
        if process_data:
            self.process_data(
                ignore_index=ignore_index,
                inplace=True,
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
        out = super().from_tdf_file(file)
        return cls(
            markers_raw=out.markers,
            forceplatforms_raw=out.forceplatforms,
            emgs_raw=out.emgs,
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
        generate a SquatJump from a StateFrame object

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        Returns
        -------
        frame: SquatJump
            a SquatJump instance.
        """
        # check the input
        if not isinstance(obj, StateFrame):
            raise ValueError("obj must be a StateFrame object.")

        # create the object instance
        out = cls(
            markers_raw=obj.markers,
            forceplatforms_raw=obj.forceplatforms,
            emgs_raw=obj.emgs,
            process_data=False,
        )
        out._processed = obj.is_processed()
        out._marker_processing_options = obj.marker_processing_options
        out._forceplatform_processing_options = obj.forceplatform_processing_options
        out._emg_processing_options = obj.emg_processing_options

        return out


class SquatJumpTest(LabTest):
    """
    Class handling the data processing and analysis of the collected data about
    a squat jump test.

    Parameters
    ----------
    baseline: StaticUprightStance
        a StaticUprightStance instance defining the baseline acquisition.

    *jumps: SquatJump
        a variable number of SquatJump objects

    Attributes
    ----------
    baseline
        the StaticUprightStance instance of the test

    jumps
        the list of available SquatJump objects.

    results_table
        a table containing the metrics resulting from each jump

    summary_table
        A table with summary statistics about the test.

    summary_plot
        a plotly FigureWidget summarizing the results of the test
    """

    # * class variables

    _baseline: UprightStance
    _jumps: list[SquatJump]

    # * attributes

    @property
    def baseline(self):
        """return the baseline acquisition of the test"""
        return self._baseline

    @property
    def jumps(self):
        """return the jumps performed during the test"""
        return self._jumps

    @property
    def results_table(self):
        """Return a table containing the test results."""

        # get the required metrics from each jump
        base = self.baseline
        res = [pd.DataFrame(get_jump_features(i, base)).T for i in self.jumps]

        # convert the results to table
        table = pd.concat(res, ignore_index=True)
        table.index = pd.Index([f"Jump {i + 1}" for i in range(table.shape[0])])

        return table

    @property
    def summary_table(self):
        """Return a table with summary statistics about the test."""
        # generate a long format table
        res = self.results_table.unstack()
        idx = res.index.to_frame()
        tbl = pd.concat([idx, res], axis=1).reset_index(drop=True)
        tbl.columns = pd.Index(["METRIC", "UNIT", "JUMP", "VALUE"])

        # get the mean and std stats
        grp = tbl.groupby(["METRIC", "UNIT"])
        ref = grp.describe([])["VALUE"][["mean", "std"]]
        ref.columns = pd.Index(["MEAN", "STD"])

        # add the values from the best jump
        best_jump_idx = np.argmax(res.Elevation.values.astype(float).flatten())
        best_jump = res.index[best_jump_idx][-1]  # type: ignore
        vals = res.loc[[i for i in res.index if i[-1] == best_jump]]
        vals.index = vals.index.droplevel(2)
        ref.loc[vals.index, "BEST"] = vals.values
        ref = pd.concat([ref.index.to_frame(), ref], axis=1)
        ref = ref.reset_index(drop=True)

        return ref

    @property
    def summary_plot(self):
        """return a matplotlib figure highlighting the test results"""

        # get the summary results in long format
        raw = self.summary_table
        best = raw[["METRIC", "UNIT", "BEST"]].copy()
        best.columns = best.columns.map(lambda x: x.replace("BEST", "VALUE"))
        best.insert(0, "TYPE", np.tile("BEST JUMP", best.shape[0]))
        mean = raw[["METRIC", "UNIT", "MEAN", "STD"]].copy()
        mean.columns = pd.Index(["METRIC", "UNIT", "VALUE", "ERROR"])
        mean.insert(0, "TYPE", np.tile("MEAN PERFORMANCE", mean.shape[0]))
        long = pd.concat([best, mean], ignore_index=True)

        # separate performance data from muscle symmetry data
        idx = long.index
        idx = [i for i in idx if str(long.loc[i, "METRIC"]).endswith("Imbalance")]
        vals = long.loc[[i for i in long.index if i not in idx]]
        syms = long.loc[idx]

        # generate the figure and the subplot grid
        muscles = np.unique(syms.METRIC.values.astype(str)).tolist()
        feats = np.unique(vals.METRIC.values.astype(str)).tolist()
        ncols = len(feats) * len(muscles)
        nrows = 1
        row0 = []
        for _ in feats:
            line = [{"colspan": len(muscles)}]
            line += [None for i in range(len(muscles) - 1)]
            row0 += line
        specs = [row0]
        if len(muscles) > 0:
            nrows += 1
            row1 = []
            for _ in muscles:
                line = [{"colspan": len(feats)}]
                line += [None for i in range(len(feats) - 1)]
                row1 += line
            specs += [row1]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=feats + muscles,
            specs=specs,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
            row_titles=None,
            column_titles=None,
            x_title=None,
            y_title=None,
        )

        # plot the jump properties
        for i, param in enumerate(feats):
            dfr = vals.loc[vals.METRIC == param]
            dfr.insert(0, "TEXT", [f"{i:0.1f}" for i in dfr.VALUE])
            mean = dfr.loc[dfr.TYPE == "MEAN"]
            base = min(dfr.VALUE.min(), (mean.VALUE - mean.ERROR.values).min())
            base = float(base * 0.9)
            dfr.insert(0, "BASE", np.tile(base, dfr.shape[0]))
            maxval = max(dfr.VALUE.max(), (mean.VALUE + mean.ERROR.values).max())
            maxval = float(maxval * 1.1)
            dfr.loc[dfr.index, "VALUE"] = dfr.loc[dfr.index, "VALUE"] - base
            fig0 = px.bar(
                data_frame=dfr,
                x="METRIC",
                y="VALUE",
                text="TEXT",
                error_y="ERROR",
                color="TYPE",
                barmode="group",
                base="BASE",
            )
            fig0.update_traces(
                showlegend=bool(i == 0),
                legendgroup="METRIC",
                legendgrouptitle_text="METRIC",
            )
            for trace in fig0.data:
                fig.add_trace(row=1, col=i * len(muscles) + 1, trace=trace)
            fig.update_yaxes(
                row=1,
                col=i * len(muscles) + 1,
                title=dfr.UNIT.values.astype(str).flatten()[0],
                range=[base, maxval],
            )
        fig.update_xaxes(row=1, showticklabels=False)

        # get symmetry data
        left = syms.copy()
        left.VALUE = 50 - left.VALUE.values.astype(float)
        left.insert(0, "SIDE", np.tile("Left", left.shape[0]))
        right = syms.copy()
        right.VALUE = 50 + right.VALUE.values.astype(float)
        right.insert(0, "SIDE", np.tile("Right", right.shape[0]))
        syms = pd.concat([left, right], ignore_index=True)
        syms.insert(0, "TEXT", [f"{i:0.1f}" for i in syms.VALUE])
        valmean = syms.loc[syms.TYPE == "MEAN", ["VALUE", "ERROR"]]
        valmax = max(syms.VALUE.max(), valmean.sum(axis=1).max()) * 1.1
        base = min(syms.VALUE.min(), (valmean.VALUE - valmean.ERROR.values).min())
        base = float(base * 0.9)
        syms.insert(0, "BASE", np.tile(base, syms.shape[0]))
        syms.loc[syms.index, "VALUE"] = syms.loc[syms.index, "VALUE"] - base
        for i, param in enumerate(muscles):
            dfr = syms.loc[syms.METRIC == param]
            fig0 = px.bar(
                data_frame=dfr,
                x="TYPE",
                y="VALUE",
                text="TEXT",
                color="SIDE",
                error_y="ERROR",
                barmode="group",
                base="BASE",
                color_discrete_sequence=px.colors.qualitative.Plotly[2:],
            )
            fig0.update_traces(
                showlegend=bool(i == 0),
                legendgroup="SIDE",
                legendgrouptitle_text="SIDE",
            )
            for trace in fig0.data:
                fig.add_trace(row=2, col=i * len(feats) + 1, trace=trace)
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_width=2,
                opacity=0.5,
                showlegend=False,
                row=2,  # type: ignore
                col=int(i * len(feats)) + 1,  # type: ignore
            )
            if i == 0:
                fig.update_yaxes(row=2, col=i * len(feats) + 1, title="%")
            else:
                fig.update_yaxes(row=2, col=i * len(feats) + 1, visible=False)
        fig.update_yaxes(row=2, range=[base, valmax])

        # update the layout and return
        fig.update_layout(
            legend={
                "x": 1,
                "y": 0.4,
                "xref": "container",
                "yref": "container",
            },
            template="simple_white",
            height=600,
            width=1200,
        )

        return go.FigureWidget(fig)

    # * methods

    def _check_valid_inputs(self):
        # check the baseline
        if not isinstance(self._baseline, UprightStance):
            raise ValueError("baseline must be a StaticUprightStance instance.")

        # check for the jumps
        if not isinstance(self._jumps, Iterable):
            msg = "'jumps' must be a list of SquatJump objects."
            raise ValueError(msg)
        for i, jump in enumerate(self._jumps):
            if not isinstance(jump, SquatJump):
                raise ValueError(f"jump {i + 1} is not a SquatJump instance.")

    # * constructors

    def __init__(
        self,
        baseline: UprightStance,
        jumps: list[SquatJump],
    ):
        super().__init__()
        self._baseline = baseline
        self._jumps = jumps
        self._check_valid_inputs()


#! FUNCTIONS


def get_jump_features(jump: SquatJump, baseline: UprightStance):
    """
    get the properties of a jump

    Parameters
    ----------
    jump: Any Jump instance
        the jump from which the features have to be extracted

    baseline: UprightStance
        an upright stance instance from which the user weight and EMG normative
        data can be extracted

    Returns
    -------
    feats: pd.Series
        a pandas Series containing the extracted features
    """
    # check the inputs
    if not isinstance(jump, SquatJump):
        raise ValueError("'jump' must be a SquatJump instance.")
    if not isinstance(baseline, UprightStance):
        raise ValueError("'baseline' must be an UprightStance instance.")

    # get the EMG norms and user weight
    weight = baseline.weight

    # get the required metrics from each jump
    con = jump.concentric_phase
    t0, t1 = con.forceplatforms.index.to_numpy()[[0, -1]]
    grf = jump.grf
    grf = grf.loc[(grf.index >= t0) & (grf.index <= t1)]
    time = grf.index.to_numpy()
    grf = grf.values.astype(float).flatten()

    # impulse
    imp = float(np.trapezoid(grf, time))
    weight_integral = np.tile(weight * G, len(time))
    weight_integral = float(np.trapezoid(weight_integral, time))
    net_imp = imp - weight_integral

    # velocity
    velocity = net_imp / weight

    # acceleration
    dtime = float(time[-1] - time[0])
    acceleration = velocity / dtime

    # power
    power = net_imp / dtime * velocity

    # elevation
    s2y = jump.flight_phase.markers.S2.Y.values.astype(float).flatten()
    elevation = float(np.max(s2y) - s2y[0]) * 100

    # efficiency
    # pot_energy = weight * G * elevation / 100
    # work = power * dtime
    # efficiency = pot_energy / work * 100

    # get the output data
    line = {
        ("Elevation", "cm"): elevation,
        ("Velocity", "m/s"): velocity,
        ("Acceleration", "m/s^2"): acceleration,
        ("Relative Power", "W/kg"): power / weight,
    }

    # add EMG data
    if jump.emgs.shape[0] > 0:

        # get normalized EMG amplitudes
        emg_norms = baseline.emg_norms.loc["median"]
        emgs = con.emgs / emg_norms

        # get muscle symmetries
        syms_emg = emgs.apply(np.trapezoid, x=emgs.index, axis=0, raw=True)
        syms_emg.sort_index(inplace=True)
        for muscle in np.unique(syms_emg.index.get_level_values(0)):
            splits = muscle.split("_")
            lbl = [i[0].upper() + i[1:].lower() for i in splits]
            lbl = " ".join(lbl + ["Imbalance"])
            val = syms_emg[muscle]["right"] - syms_emg[muscle]["left"]
            val /= syms_emg[muscle]["right"] + syms_emg[muscle]["left"]
            line[(lbl, "%")] = float(val.values * 100)

    return pd.Series(line)
