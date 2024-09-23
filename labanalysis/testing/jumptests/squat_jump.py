"""Squat Jump Test module"""

#! IMPORTS


from typing import Iterable
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ... import signalprocessing as sp
from ..base import LabTest, G
from ..frames import StateFrame
from ..statictests import StaticUprightStance

__all__ = ["SquatJump", "SquatJumpTest"]


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
        grfy = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grft = self.forceplatforms.index.to_numpy()
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
        grfy = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grft = self.forceplatforms.index.to_numpy()
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

    @property
    def rate_of_force_development(self):
        """
        return the maximum rate of force development over the concentric phase
        of the jump in N/s
        """
        # get the vertical force data
        confp = self.concentric_phase.forceplatforms.fRes.FORCE
        grf = confp.Y.values.astype(float).flatten()
        time = confp.index.to_numpy()

        # get the rfd value
        return float(np.max(sp.winter_derivative1(grf, time)))

    @property
    def velocity_at_toeoff(self):
        """return the vertical velocity at the toeoff in m/s"""

        # get the vertical velocity
        pos = self.markers.S2.Y.values.astype(float).flatten()
        time = self.markers.index.to_numpy()
        vel = sp.winter_derivative1(pos, time)

        # remove the first and last sample from time to be aligned with vel
        time = time[1:-1]

        # get the velocity at the last time instant of the concentric phase
        loc = np.where(time >= self.concentric_phase.markers.index[-1])[0][0]
        return float(vel[loc])

    @property
    def jump_height(self):
        """return the jump height in m"""

        # get the vertical position of S2 at the flight phase
        pos = self.flight_phase.markers.S2.Y.values.astype(float).flatten()

        # get the difference between the first and highest sample
        maxh = np.max(pos)
        toeoffh = pos[0]
        return float(maxh - toeoffh)

    @property
    def concentric_power(self):
        """return the mean power in W generated during the concentric phase"""
        # get the concentric phase grf and vertical velocity
        con = self.concentric_phase
        s2y = con.markers.S2.Y.values.astype(float).flatten()
        s2t = con.markers.index.to_numpy()
        s2v = sp.winter_derivative1(s2y, s2t)
        s2t = s2t[1:-1]
        grf = con.forceplatforms.loc[s2t].fRes.FORCE.Y.values.astype(float).flatten()

        # return the mean power output
        return float(np.mean(grf * s2v))

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

        # ensure that the 'fRes', 'rFoot' and 'lFoot' force platform objects exist
        lbls = np.unique(self.forceplatforms.columns.get_level_values(0))
        required_fp = ["fRes", "lFoot", "rFoot"]
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

        # check the process data attribute
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

    _baseline: StaticUprightStance
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
    def _raw_summary_table(self):
        """
        private attribute used internally to derive the summary metrics from
        each jump. It should not be called except from the 'summary_table'
        and the 'summary_plot' attribute.
        """
        # get the EMG norms and user weight
        weight = self.baseline.weight

        # get the required metrics from each jump
        summary_list = []
        for jump in self.jumps:
            line = {
                ("jump height", "", "cm"): jump.jump_height * 100,
                ("concentric power", "", "W/kg"): jump.concentric_power / weight,
                ("force development", "", "kgf/s"): jump.rate_of_force_development / G,
                ("velocity at toe-off", "", "m/s"): jump.velocity_at_toeoff,
                **self._jump_symmetry(jump).to_dict(),
            }
            summary_list += [pd.DataFrame(pd.Series(line)).T]

        # conver the results to table
        table = pd.concat(summary_list, ignore_index=True)
        table.index = pd.Index([f"JUMP{i + 1}" for i in range(table.shape[0])])

        return table

    @property
    def summary_table(self):
        """
        Return a table with summary statistics about the test. The table
        includes the following elements:
            * jump height
            * concentric power
            * rate of force development
            * velocity at toeoff
            * muscle symmetry
        """
        # get the raw table
        table = self._raw_summary_table

        # append mean and max values between jumps
        mean = table.mean(axis=0).values
        best = table.max(axis=0).values
        table.loc["MEAN", table.columns] = mean
        table.loc["BEST", table.columns] = best
        table.loc["BEST", [i for i in table.columns if i[-1] == "%"]] = None

        return table

    @property
    def summary_plot(self):
        """return a matplotlib figure highlighting the test results"""

        # generate the figure and the subplot grid
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["JUMP HEIGHT", "POWER", "SYMMETRIES"],
            specs=[[{}, {}], [{"colspan": 2}, None]],
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.20,
            vertical_spacing=0.15,
            row_titles=None,
            column_titles=None,
            x_title=None,
            y_title=None,
        )

        # get the raw results
        raw = self._raw_summary_table

        # plot the jump height
        for trace in self._get_bar_traces(raw["jump height"]):
            fig.add_trace(row=1, col=1, trace=trace)
        fig.update_yaxes(row=1, col=1, title="cm")

        # plot the concentric power
        for trace in self._get_bar_traces(raw["concentric power"]):
            fig.add_trace(row=1, col=2, trace=trace)
        fig.update_yaxes(row=1, col=2, title="W/kg")

        # get symmetry mean data
        sym_objs = np.unique([i[0] for i in raw.columns if i[-1] == "%"])
        avg = raw.mean(axis=0)[sym_objs].astype(float)
        std = raw.std(axis=0)[sym_objs].astype(float)
        base = [max((i - j) * 0.9, 0) for i, j in zip(avg.values, std.values)]
        base = float(np.min(base))
        sym_list = []
        lbls = np.unique(avg.index.get_level_values(0)).tolist()
        for lbl in lbls:
            for side in ["left", "right"]:
                line = {
                    "SIDE": side.upper(),
                    "METRIC": lbl.upper().replace("_", " "),
                    "VALUE": float(avg[lbl][side].values) - base,
                    "ERROR": float(std[lbl][side].values),
                    "TEXT": f"{float(avg[lbl][side].values):0.1f}",
                    "BASE": base,
                }
                sym_list += [pd.DataFrame(pd.Series(line)).T]
        sym_df = pd.concat(sym_list, ignore_index=True)

        # plot symmetry
        fig0 = px.bar(
            data_frame=sym_df,
            x="METRIC",
            y="VALUE",
            color="SIDE",
            base="BASE",
            error_y="ERROR",
            text="TEXT",
            barmode="group",
            opacity=1,
            color_discrete_sequence=[
                px.colors.qualitative.Plotly[2],
                px.colors.qualitative.Plotly[4],
            ],
        )
        fig0.update_layout(showlegend=False)
        for trace in fig0.data:
            fig.add_trace(row=2, col=1, trace=trace)
        fig.update_yaxes(row=2, col=1, title="%")

        # add the target mean symmetry line
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_width=2,
            line_color=px.colors.qualitative.Plotly[5],
            opacity=0.5,
            showlegend=False,
            row=2,  # type: ignore
            col=1,  # type: ignore
        )

        # update the layout and return
        fig.update_traces(error_y_color="rgba(0, 0, 0, 0.3)")
        fig.update_layout(
            legend={
                "x": 1,
                "y": 0.4,
                "xref": "container",
                "yref": "container",
                "title": "SIDE",
            },
            template="simple_white",
            height=600,
            width=600,
        )

        return go.FigureWidget(fig)

    # * methods

    def _jump_symmetry(self, jump: SquatJump):
        """
        private method used to extract jump symmetries from emg data and grf.

        Parameters
        ----------
        jump: SquatJump
            the jump to be compared

        Returns
        -------
        syms: dict[str, float]
            the relative contribution of each side for each available pair of
            data.

        Procedure
        ---------
        the method integrates left and right GRF (and optionally EMG signals)
        and provides the relative ratio between the two sides.
        """
        # get the jump concentric phase
        con = jump.concentric_phase

        # get GRF integrals over time
        fps = con.forceplatforms
        lgrf = fps.lFoot.FORCE.Y.values.astype(float).flatten()
        rgrf = fps.rFoot.FORCE.Y.values.astype(float).flatten()
        time = fps.index.to_numpy()
        syms = {
            ("grf", "left", "%"): float(np.trapezoid(lgrf, x=time)),
            ("grf", "right", "%"): float(np.trapezoid(rgrf, x=time)),
        }
        syms = pd.Series(syms)

        # check if EMG data exists
        if jump.emgs.shape[0] > 0:

            # get normalized EMG amplitudes
            emg_norms = self.baseline.emg_norms.loc["median"]
            emgs = con.emgs / emg_norms

            # get integral of EMG signals
            syms_emg = emgs.apply(np.trapezoid, x=emgs.index, axis=0, raw=True)
            syms_emg.sort_index(inplace=True)
            for i in syms_emg.index:
                syms[(i[0], i[1], "%")] = float(syms_emg[i])

        # calculate percentage of each side
        labels = np.unique(syms.index.get_level_values(0))
        for label in labels:
            syms[label] = syms[label] / syms[label].sum() * 100

        return syms

    def _get_bar_traces(self, obj: pd.Series):
        """
        private method used to extract the traces to be plotted

        Parameters
        ----------
        obj: pd.Series
            the series of the data

        Returns
        -------
        traces: tuple
            a tuple containing the traces.
        """
        vals = obj.values.astype(float).flatten()
        base = min(np.min(vals) * 0.9, np.mean(vals) - np.std(vals))
        df = {
            "BASE": np.tile(base, len(vals)),
            "VALUE": vals - base,
            "METRIC": obj.index,
            "TEXT": [f"{i:0.1f}" for i in vals],
        }
        fig1 = px.bar(
            data_frame=pd.DataFrame(df),
            x="METRIC",
            base="BASE",
            y="VALUE",
            text="TEXT",
            barmode="group",
            opacity=1,
        )
        fig1.update_traces(marker_color=px.colors.qualitative.Plotly[0])
        fig1.add_trace(
            trace=go.Bar(
                x=["MEAN"],
                y=[np.mean(vals) - base],
                base=[base],
                text=[f"{np.mean(vals):0.1f}"],
                error_y=go.bar.ErrorY(array=[np.std(vals)]),
                showlegend=False,
                marker_color=px.colors.qualitative.Plotly[1],
            ),
        )

        return fig1.data

    def _check_valid_inputs(self):
        # check the baseline
        if not isinstance(self._baseline, StaticUprightStance):
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
        baseline: StaticUprightStance,
        jumps: list[SquatJump],
    ):
        super().__init__()
        self._baseline = baseline
        self._jumps = jumps
        self._check_valid_inputs()
