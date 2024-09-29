"""Side Jump Test module"""

#! IMPORTS


from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .squat_jump import get_jump_features

from ..base import LabTest
from ..frames import StateFrame
from ..posturaltests.upright import UprightStance
from .counter_movement_jump import CounterMovementJump

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

    @property
    def grf(self):
        """return the grf"""
        grfy = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grf = pd.Series(grfy, index=self.forceplatforms.index.to_numpy())
        return grf.astype(float)

    # * methods

    def copy(self):
        """create a copy of the object"""
        return self.from_stateframe(self, self.side)

    def _check_inputs(self):
        """check the validity of the entered data"""
        # ensure that the 'fRes' force platform objects exist
        lbls = np.unique(self.forceplatforms.columns.get_level_values(0))
        required_fp = ["fRes"]
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
        if not isinstance(side, str):
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
        generate a SideJump from a StateFrame object

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


class SideJumpTest(LabTest):
    """
    Class handling the data processing and analysis of the collected data about
    a side jump test.

    Parameters
    ----------
    baseline: UprightStance
        a UprightStance instance defining the baseline acquisition.

    left_jumps: Iterable[SideJump]
        a variable number of SideJump objects

    right_jumps: Iterable[SideJump]
        a variable number of SideJump objects

    Attributes
    ----------
    baseline
        the UprightStance instance of the test

    left_jumps
        the list of available (left) SideJump objects.

    right_jumps
        the list of available (right) SideJump objects.

    results_table
        a table containing the metrics resulting from each jump

    summary_table
        A table with summary statistics about the test.

    summary_plot
        a plotly FigureWidget summarizing the results of the test
    """

    # * class variables

    _baseline: UprightStance
    _left_jumps: list[SideJump]
    _right_jumps: list[SideJump]

    # * attributes

    @property
    def baseline(self):
        """return the baseline acquisition of the test"""
        return self._baseline

    @property
    def left_jumps(self):
        """return the left jumps performed during the test"""
        return self._left_jumps

    @property
    def right_jumps(self):
        """return the right jumps performed during the test"""
        return self._right_jumps

    @property
    def results_table(self):
        """Return a table containing the test results."""

        # get the required metrics from each jump
        res = []
        base = self.baseline
        for jump in self.left_jumps + self.right_jumps:
            dfr = pd.DataFrame(get_jump_features(jump, base)).T
            dfr = dfr[[i for i in dfr.columns if not i[0].endswith("Imbalance")]]
            dfr.insert(0, "SIDE", np.tile(jump.side, dfr.shape[0]))
            res += [dfr]

        # convert the results to table
        table = pd.concat(res, ignore_index=True)
        table.index = pd.Index([f"Jump {i + 1}" for i in range(table.shape[0])])

        return table

    @property
    def summary_table(self):
        """Return a table with summary statistics about the test."""
        # generate a long format table
        res = self.results_table
        res.insert(0, "JUMP", res.index)
        tbl = []
        for i in res.columns[2:]:
            dfr = res[[("JUMP", ""), ("SIDE", ""), i]].copy()
            dfr.columns = pd.Index(["JUMP", "SIDE", "VALUE"])
            dfr.insert(0, "UNIT", np.tile(i[1], dfr.shape[0]))
            dfr.insert(0, "METRIC", np.tile(i[0], dfr.shape[0]))
            tbl += [dfr]
        tbl = pd.concat(tbl, ignore_index=True)

        # get the mean and std stats
        grp = tbl.groupby(["METRIC", "UNIT", "SIDE"])
        ref = grp.describe([])["VALUE"][["mean", "std"]]
        ref.columns = pd.Index(["MEAN", "STD"])

        # add the values from the best jump on each side
        for side in res.SIDE.unique():
            val = res.loc[res.SIDE == side]
            idx = np.argmax(val.Elevation.values.astype(float).flatten())
            best = str(val.JUMP.values[idx])
            vals = res.loc[[i for i in res.JUMP if i == best]]
            for i, (metric, unit) in enumerate(vals.columns[2:]):
                ref_idx = (metric, unit, side)
                ref.loc[ref_idx, "BEST"] = vals[(metric, unit)].values
        ref = pd.concat([ref.index.to_frame(), ref], axis=1)
        ref = ref.reset_index(drop=True)

        return ref

    @property
    def summary_plot(self):
        """return a matplotlib figure highlighting the test results"""

        # get the summary results in long format
        raw = self.summary_table
        best = raw[["METRIC", "UNIT", "SIDE", "BEST"]].copy()
        best.columns = best.columns.map(lambda x: x.replace("BEST", "VALUE"))
        best.insert(0, "TYPE", np.tile("BEST JUMP", best.shape[0]))
        mean = raw[["METRIC", "UNIT", "SIDE", "MEAN", "STD"]].copy()
        mean.columns = pd.Index(["METRIC", "UNIT", "SIDE", "VALUE", "ERROR"])
        mean.insert(0, "TYPE", np.tile("MEAN PERFORMANCE", mean.shape[0]))
        long = pd.concat([best, mean], ignore_index=True)

        # generate the figure and the subplot grid
        feats = np.unique(long.METRIC.values.astype(str)).tolist()
        fig = make_subplots(
            rows=2,
            cols=len(feats),
            subplot_titles=feats,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
            row_titles=["PERFORMANCE", "SYMMETRY"],
            column_titles=None,
            x_title=None,
            y_title=None,
        )

        # plot the jump properties
        for i, param in enumerate(feats):
            dfr = long.loc[long.METRIC == param]
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
                x="TYPE",
                y="VALUE",
                text="TEXT",
                error_y="ERROR",
                color="SIDE",
                barmode="group",
                base="BASE",
            )
            fig0.update_traces(
                showlegend=bool(i == 0),
                legendgroup="SIDE",
                legendgrouptitle_text="SIDE",
            )
            for trace in fig0.data:
                fig.add_trace(row=1, col=i + 1, trace=trace)
            fig.update_yaxes(
                row=1,
                col=i + 1,
                title=dfr.UNIT.values.astype(str).flatten()[0],
                range=[base, maxval],
            )

        # get the symmetries
        syms = []
        for grp, dfa in long.groupby(["METRIC", "TYPE"]):
            idx_sx = dfa.SIDE == "Left"
            idx_dx = dfa.SIDE == "Right"
            sx = float(dfa.loc[idx_sx].VALUE.values[0])
            dx = float(dfa.loc[idx_dx].VALUE.values[0])
            total = sx + dx
            delta = 100 * (dx - sx) / total
            dfa.loc[idx_sx, "VALUE"] = 50 - delta
            dfa.loc[idx_dx, "VALUE"] = 50 + delta
            syms += [dfa[["METRIC", "TYPE", "SIDE", "VALUE"]].copy()]
        syms = pd.concat(syms, ignore_index=True)
        syms.insert(0, "TEXT", [f"{i:0.1f}" for i in syms.VALUE])
        base = float(syms.VALUE.min() * 0.9)
        maxv = float(syms.VALUE.max() * 1.1)
        syms.insert(0, "BASE", np.tile(base, syms.shape[0]))
        syms.loc[syms.index, "VALUE"] = syms.VALUE - syms.BASE.values
        for i, feat in enumerate(feats):
            dfr = syms.loc[syms.METRIC == feat]
            fig0 = px.bar(
                data_frame=dfr,
                x="TYPE",
                y="VALUE",
                text="TEXT",
                color="SIDE",
                barmode="group",
                base="BASE",
            )
            fig0.update_traces(showlegend=False)
            for trace in fig0.data:
                fig.add_trace(row=2, col=i + 1, trace=trace)
        fig.update_yaxes(row=2, visible=False, range=[base, maxv])
        fig.update_yaxes(row=2, col=1, visible=True, title="%")

        # add mean symmetry line
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_width=2,
            line_color=px.colors.qualitative.Plotly[2],
            opacity=0.5,
            row=2,  # type: ignore
            showlegend=False,
        )

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
            raise ValueError("baseline must be a UprightStance instance.")

        # check for the jumps
        if not isinstance(self._left_jumps, list):
            raise ValueError("'left_jumps' must be a list of SideJump objects.")
        if not isinstance(self._right_jumps, list):
            raise ValueError("'right_jumps' must be a list of SideJump objects.")
        for jump in self._left_jumps + self._right_jumps:
            if not isinstance(jump, SideJump):
                msg = f"All jumps must be SideJump instances."
                raise ValueError(msg)

    # * constructors

    def __init__(
        self,
        baseline: UprightStance,
        left_jumps: list[SideJump],
        right_jumps: list[SideJump],
    ):
        self._baseline = baseline
        self._left_jumps = left_jumps
        self._right_jumps = right_jumps
        self._check_valid_inputs()
