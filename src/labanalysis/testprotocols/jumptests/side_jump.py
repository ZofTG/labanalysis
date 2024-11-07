"""Side Jump Test module"""

#! IMPORTS


from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ...constants import G

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
        raw = []
        for i, jump in enumerate(self.left_jumps):
            dfj = self._get_jump_results(jump)
            lbl = np.tile(f"Jump {i + 1}", dfj.shape[0])
            dfj.insert(0, ("Jump", "", "", "", ""), lbl)
            raw += [dfj]
        for i, jump in enumerate(self.right_jumps):
            dfj = self._get_jump_results(jump)
            lbl = np.tile(f"Jump {i + 1}", dfj.shape[0])
            dfj.insert(0, ("Jump", "", "", "", ""), lbl)
            raw += [dfj]

        return pd.concat(raw, ignore_index=True)

    @property
    def summary_table(self):
        """Return a table with summary statistics about the test."""

        # get the required metrics from each jump
        out = []
        for i, jump in enumerate(self.left_jumps):
            new = get_jump_features(jump, self.baseline)
            new = new.loc[new.Unit != "%"]
            new.insert(0, "Jump", np.tile(i + 1, new.shape[0]))
            new.insert(0, "Side", np.tile(jump.side, new.shape[0]))
            out += [new]
        for i, jump in enumerate(self.right_jumps):
            new = get_jump_features(jump, self.baseline)
            new = new.loc[new.Unit != "%"]
            new.insert(0, "Jump", np.tile(i + 1, new.shape[0]))
            new.insert(0, "Side", np.tile(jump.side, new.shape[0]))
            out += [new]
        res = pd.concat(out, ignore_index=True)

        # get the mean and std stats
        grp = res.groupby(["Parameter", "Side", "Unit"])
        ref = grp.describe([])["Value"][["mean", "std"]]
        ref.columns = pd.Index(["Mean", "Std"])

        # add the values from the best jump for each side
        for side, dfr in res.groupby("Side"):
            edf = dfr.loc[dfr.Parameter == "Elevation"]
            hight = edf.Value.values.astype(float).flatten()
            best_jump = edf.iloc[np.argmax(hight)].Jump  # type: ignore
            best = res.loc[(res.Jump == best_jump) & (res.Side == side)]
            best = best.drop("Jump", axis=1)
            index = pd.MultiIndex.from_frame(best[["Parameter", "Side", "Unit"]])
            ref.loc[index, "Best"] = best.Value.values

        return pd.DataFrame(ref)

    @property
    def results_plot(self):
        """return a plotly figurewidget highlighting the resulting data"""

        # get the results table
        res = self.results_table
        cols = [("MARKER", "S2", "COORDINATE", "Y", "m")]
        cols += [("FORCE_PLATFORM", "fRes", "FORCE", "Y", "N")]
        raw = []
        jumps = res.Jump.values.astype(str).flatten()
        time = res.Time.values.astype(float).flatten()
        phase = res.Phase.values.astype(str).flatten()
        sides = res.Side.values.astype(str).flatten()
        for col in cols:
            typ = col[1] if col[0] == "MARKER" else "Force"
            typ = typ.split("_")[0].capitalize()
            val = res[col].values.astype(float).flatten()
            if col[-1] == "N":
                unit = "kgf"
                val = val / G
            else:
                unit = "cm"
                val = val * 1e2
            new = {
                "Type": np.tile(typ, len(val)),
                "Unit": np.tile(unit, len(val)),
                "Value": val,
                "Time": time,
                "Jump": jumps,
                "Phase": phase,
                "Side": sides,
            }
            raw += [pd.DataFrame(new)]
        raw = pd.concat(raw, ignore_index=True)

        # generate the figure
        fig = px.line(
            data_frame=raw,
            x="Time",
            y="Value",
            color="Jump",
            line_dash="Phase",
            facet_row="Type",
            facet_col="Side",
            template="simple_white",
            height=600,
            width=1200,
        )

        # update the layout and return
        fig.update_traces(opacity=0.5)
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
        fig.update_yaxes(title="", matches=None)
        fig.update_yaxes(showticklabels=False, col=2, visible=False)
        fig.update_xaxes(showticklabels=False, row=2, visible=False)
        for i, unit in enumerate(raw.Unit.unique()[::-1]):
            fig.update_yaxes(row=i + 1, col=1, title=unit)
        cm_range = raw.loc[raw.Unit == "cm"].Value.values.astype(float).flatten()
        cm_range = [np.min(cm_range), np.max(cm_range)]
        fig.update_yaxes(row=2, range=cm_range)
        kgf_range = raw.loc[raw.Unit == "kgf"].Value.values.astype(float).flatten()
        kgf_range = [np.min(kgf_range), np.max(kgf_range)]
        fig.update_yaxes(row=1, range=kgf_range)

        return go.FigureWidget(fig)

    @property
    def summary_plots(self):
        """return a plotly figurewidget highlighting the test summary"""

        # get the summary results in long format
        raw = self.summary_table
        best = raw[["Best"]].copy()
        best.columns = pd.Index(["Value"])
        best.insert(0, "Type", np.tile("Best", best.shape[0]))
        mean = raw[["Mean", "Std"]].copy()
        mean.columns = pd.Index(["Value", "Error"])
        mean.insert(0, "Type", np.tile("Mean", mean.shape[0]))
        long = pd.concat([best, mean])
        long = pd.concat([long.index.to_frame(), long], axis=1)
        vals = long.reset_index(drop=True)

        # prepare the data for being rendered
        vals.insert(0, "Text", vals.Value.map(lambda x: f"{x:0.2f}"))
        for parameter, dfr in vals.groupby("Parameter"):
            base = min(np.min(dfr.Value), np.nanmin(dfr.Value - dfr.Error)) * 0.9
            vals.loc[dfr.index, "Base"] = base
        vals.loc[vals.index, "Value"] -= vals.Base

        # generate the figure and the subplot grid
        fig = px.bar(
            data_frame=vals,
            x="Type",
            y="Value",
            error_y="Error",
            color="Side",
            text="Text",
            base="Base",
            facet_col="Parameter",
            facet_col_spacing=0.1,
            barmode="group",
            template="simple_white",
            height=300,
            width=1200,
        )

        # update the layout
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[1]))
        fig.update_traces(error_y_color="rgba(0, 0, 0, 0.3)")
        fig.update_xaxes(title="")
        fig.update_yaxes(matches=None, showticklabels=True)
        for i, (parameter, dfr) in enumerate(vals.groupby("Parameter")):
            fig.update_yaxes(col=i + 1, title=dfr.Unit.values[0])
        fig.update_layout(template="simple_white", height=600, width=1200)

        return go.FigureWidget(fig)

    # * methods

    def _get_jump_results(self, jump: SideJump):
        """private method used to obtain jump results"""
        col = ("Phase", "", "", "", "")
        try:
            dfe = jump.eccentric_phase.to_dataframe().dropna()
            dfe.insert(0, col, np.tile("Eccentric", dfe.shape[0]))
        except Exception:
            dfe = pd.DataFrame()
        try:
            dfc = jump.concentric_phase.to_dataframe().dropna()
            dfc.insert(0, col, np.tile("Concentric", dfc.shape[0]))
        except Exception:
            dfc = pd.DataFrame()
        try:
            dff = jump.flight_phase.to_dataframe().dropna()
            dff.insert(0, col, np.tile("Flight", dff.shape[0]))
        except Exception:
            dff = pd.DataFrame()
        try:
            dfl = jump.loading_response_phase.to_dataframe().dropna()
            dfl.insert(0, col, np.tile("Loading Response", dfl.shape[0]))
        except Exception:
            dfe = pd.DataFrame()

        dfj = pd.concat([dfe, dfc, dff, dfl])
        time = dfj.index.to_numpy() - dfj.index[0]
        dfj.insert(0, ("Time", "", "", "", ""), time)
        dfj.reset_index(inplace=True, drop=True)
        dfj.insert(0, ("Side", "", "", "", ""), np.tile(jump.side, dfj.shape[0]))
        return dfj

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
