"""Squat Jump Test module"""

#! IMPORTS


from abc import abstractmethod
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import *
from ... import signalprocessing as sp
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
    """

    # * attributes

    @property
    def concentric_phase(self):
        """
        return a StateFrame representing the concentric phase of the jump

        Procedure
        ---------
            1. get the index of the peak vertical velocity of S2 marker
            2. check for the last zero velocity occuring before the peak.
            3. look for the last sample with the feet in touch to the ground.
            4. return a slice of the data containing the subset defined by
            points 2 and 3.

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the concentric
            phase of the jump
        """
        # get the vertical velocity in S2
        s2_y = self.markers.S2.Y.values.astype(float).flatten()
        s2_t = self.markers.index.to_numpy()
        s2_v = sp.winter_derivative1(s2_y, s2_t)
        s2_y = s2_y[1:-1]
        s2_t = s2_t[1:-1]

        # get the index of the peak velocity
        max_v = np.argmax(s2_v)

        # look at the zeros in the vertical velocity occurring before max_v
        zeros_s2 = sp.crossings(s2_v[:max_v], 0)[0].tolist()

        # look at the last local minima in vertical velocity occuring before
        # max_v
        vlocal_min = sp.find_peaks(-s2_v[:max_v]).tolist()

        # get the latest between zeros_s2 and vlocal_min
        start_idx = np.sort(vlocal_min + zeros_s2)
        if len(start_idx) == 0:
            msg = "No local minima or zeros have been found in vertical velocity."
            raise RuntimeError(msg)

        # get the time instants corresponding to ground reaction force = zero
        grf_y = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grf_t = self.forceplatforms.index.to_numpy()
        zeros_grf = sp.crossings(grf_y, 30)[0]
        msg = "No zero values found in ground reaction force"
        if len(zeros_grf) == 0:
            raise RuntimeError(msg)

        # set the time start as the last zero in the vertical velocity
        time_start = float(round(s2_t[start_idx][-1], 3))

        # set the time stop as the time instant occurring immediately before
        # the first zero in grf occurring after time_start
        time_stop = grf_t[zeros_grf]
        time_stop = time_stop[time_stop > time_start]
        if len(time_stop) == 0:
            msg += " after the start of the concentric phase."
            raise RuntimeError(msg)
        time_stop = float(round(grf_t[grf_t < time_stop[0]][-1], 3))

        # return a slice of the available data
        return self.slice(time_start, time_stop)

    @property
    def flight_phase(self):
        """
        return a StateFrame representing the flight phase of the jump

        Procedure
        ---------
            1. get the batches of samples with ground reaction force being zero.
            2. take the longed batch.
            3. take the time corresponding to the start and stop of the batch.
            4. return a slice containing only the data corresponding to the
            detected start and stop values.

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the flight
            phase of the jump
        """

        # get the indices of the largest interval in the ground reaction force
        # with zeros
        grf_y = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grf_t = self.forceplatforms.index.to_numpy()
        zeros_batches = sp.continuous_batches(grf_y <= 30)
        msg = "No zero values found in ground reaction force"
        if len(zeros_batches) == 0:
            raise RuntimeError(msg)

        # take the time corresponding to the start and stop of the batch
        zeros_batch = zeros_batches[np.argmax([len(i) for i in zeros_batches])]
        if len(zeros_batch) < 2:
            raise RuntimeError("no flight phase detected")
        times = grf_t[zeros_batch][[0, -1]]
        time_start, time_stop = [float(round(i, 3)) for i in times]

        # return a slice of the available data
        return self.slice(time_start, time_stop)

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

    # * constructors

    def __init__(
        self,
        markers_raw: pd.DataFrame,
        forceplatforms_raw: pd.DataFrame,
        emgs_raw: pd.DataFrame,
        process_data: bool = True,
        ignore_index: bool = True,
        markers_fcut: int | float | None = 6,
        forces_fcut: int | float | None = 50,
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

        forces_fcut: int | float | None = 50
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
        forces_fcut: int | float | None = 50,
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

        forces_fcut: int | float | None = 50
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
        generate a StaticUprightStance from a StateFrame object

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

        # get the raw results
        raw = self._raw_summary_table

        # get the mean and the standard deviation
        avg = raw.mean(axis=0)
        std = raw.std(axis=0)

        # generate the figure and the subplot grid
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=None,
            vertical_spacing=None,
            subplot_titles=["SYMMETRIES", "JUMP HEIGHT", "POWER"],
            row_titles=None,
            column_titles=None,
            x_title=None,
            y_title=None,
            specs=[[{"colspan": 2}, None], [{}, {}]],
        )

        # get the symmetry data
        sym_objs = np.unique([i[0] for i in raw.columns if i[-1] == "%"])
        cols = raw.columns.get_level_values(0)
        idx = [np.where(i == cols)[0][0] for i in sym_objs]
        df_sym = []
        base = avg[sym_objs].values.astype(float)
        base = base - std[sym_objs].values.astype(float)
        base = min(avg[sym_objs].min(), np.min(base)) * 0.9
        for name in sym_objs[np.argsort(idx)]:
            for side in ["left", "right"]:
                line = {
                    "SIDE": side.upper(),
                    "METRIC": name.upper().replace("_", " "),
                    "VALUE": avg[(name, side, "%")] - base,
                    "ERROR": std[(name, side, "%")],
                    "TEXT": f"{avg[(name, side, "%")]:0.1f}",
                    "BASE": base,
                }
                df_sym += [pd.DataFrame(pd.Series(line)).T]

        # plot the symmetries
        fig0 = px.bar(
            data_frame=pd.concat(df_sym, ignore_index=True),
            x="METRIC",
            y="VALUE",
            base="BASE",
            color="SIDE",
            error_y="ERROR",
            text="TEXT",
            barmode="group",
            opacity=1,
        )
        for trace in fig0.data:
            fig.add_trace(
                row=1,
                col=1,
                trace=trace,
            )
        fig.update_yaxes(row=1, col=1, title="%")

        # plot the jump height
        for trace in self._get_bar_traces(raw["jump height"]):
            fig.add_trace(
                row=2,
                col=1,
                trace=trace,
            )
        fig.update_yaxes(row=2, col=1, title="cm")

        # plot the concentric power
        for trace in self._get_bar_traces(raw["concentric power"]):
            fig.add_trace(
                row=2,
                col=2,
                trace=trace,
            )
        fig.update_yaxes(row=2, col=2, title="W")

        # update the layout and return
        fig.update_traces(error_y_color="rgba(0, 0, 0, 0.3)")
        fig.update_layout(
            legend_title="SIDE",
            template="simple_white",
            height=600,
            width=800,
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

    def _get_bar_traces(self, obj:pd.Series):
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
            'BASE': np.tile(base, len(vals)),
            'VALUE': vals - base,
            'METRIC': obj.index,
            'TEXT': [f"{i:0.1f}" for i in vals],
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
        fig1.update_traces(marker_color="rgba(200, 200, 200, 1)")
        fig1.add_trace(
            trace=go.Bar(
                x=["MEAN"],
                y=[np.mean(vals) - base],
                base=[base],
                text=[f"{np.mean(vals):0.1f}"],
                error_y=go.bar.ErrorY(array=[np.std(vals)]),
                marker_color="rgba(150, 100, 100, 1)",
                name="MEAN",
                showlegend=False,
            ),
        )

        return fig1.data

    # * constructors

    def __init__(self, baseline: StaticUprightStance, *jumps: SquatJump):
        super().__init__()

        # check the baseline
        if not isinstance(baseline, StaticUprightStance):
            raise ValueError("baseline must be a StaticUprightStance instance.")
        self._baseline = baseline

        # check for the jumps
        for i, jump in enumerate(jumps):
            if not isinstance(jump, SquatJump):
                raise ValueError(f"jump {i + 1} is not a SquatJump instance.")
        self._jumps = list(jumps)
