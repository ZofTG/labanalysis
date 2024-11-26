"""Side Jump Test module"""

#! IMPORTS


from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...constants import G
from ...plotting.plotly import bars_with_normative_bands
from ..frames import StateFrame
from ..posturaltests.upright import UprightStance
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


class SideJumpTest(CounterMovementJumpTest):
    """
    Class handling the data processing and analysis of the collected data about
    a side jump test.

    Attributes
    ----------
    baseline: UprightStance
        a UprightStance instance defining the baseline acquisition.

    left_jumps: Iterable[SideJump]
        a variable number of SideJump objects

    right_jumps: Iterable[SideJump]
        a variable number of SideJump objects

    jumps
        the list of available SquatJump objects.

    name: str
        the name of the test

    Methods
    -------
    results
        return a plotly figurewidget highlighting the resulting data
        and a table with the resulting outcomes as pandas DataFrame.

    summary
        return a dictionary with the figures highlighting the test summary
        and a table reporting the summary data.

    save
        a method allowing the saving of the data in an appropriate format.

    load
        a class method to load a LabTest object saved in its own format.
    """

    # * class variables

    _left_jumps: list[SideJump]
    _right_jumps: list[SideJump]

    # * attributes

    @property
    def left_jumps(self):
        """return the left jumps performed during the test"""
        return self._left_jumps

    @property
    def right_jumps(self):
        """return the right jumps performed during the test"""
        return self._right_jumps

    # * methods

    def _simplify_table(self, table: pd.DataFrame):
        """
        get a simplified table containing only the reference data for the
        jumps

        Parameters
        ----------
        table: pd.DataFrame
            the table to be simplified

        Returns
        -------
        simple: pd.DataFrame
            the simplified table
        """
        cols = [("MARKER", "S2", "COORDINATE", "Y", "m")]
        cols += [("FORCE_PLATFORM", "fRes", "FORCE", "Y", "N")]
        cols += [i for i in table.columns if i[0] == "EMG"]
        raw = []
        jumps = table.Jump.values.astype(str).flatten()
        sides = table.Side.values.astype(str).flatten()
        time = table.Time.values.astype(float).flatten()
        phase = table.Phase.values.astype(str).flatten()
        for col in cols:
            typ = col[1] if col[0] == "EMG" else col[0]
            typ = typ.split("_")[0].capitalize()
            if col[3] == "Left" or col[3] == "Right":
                name = col[3]
            else:
                name = col[1]
            val = table[col].values.astype(float).flatten()
            if col[-1] == "V":
                unit = "uV"
                val = val * 1e6
            elif col[-1] == "N":
                unit = "kgf"
                val = val / G
            else:
                unit = "cm"
                val = val * 1e2
            new = {
                "Type": np.tile(typ, len(val)),
                "Parameter": np.tile(name, len(val)),
                "Unit": np.tile(unit, len(val)),
                "Value": val,
                "Time": time,
                "Jump": jumps,
                "Phase": phase,
                "Side": sides,
            }
            raw += [pd.DataFrame(new)]
        return pd.concat(raw, ignore_index=True)

    def _make_results_table(self):
        """
        private method used to generate the table required for creating
        the results figure
        """
        # get the results table
        raw = []
        for i, jump in enumerate(self.right_jumps):
            dfj = self._get_single_jump_results(jump)
            lbl = np.tile(f"Jump {i + 1}", dfj.shape[0])
            dfj.insert(0, ("Jump", "", "", "", ""), lbl)
            dfj.insert(0, "Side", np.tile(jump.side, dfj.shape[0]))
            raw += [dfj]
        for i, jump in enumerate(self.left_jumps):
            dfj = self._get_single_jump_results(jump)
            lbl = np.tile(f"Jump {i + 1}", dfj.shape[0])
            dfj.insert(0, ("Jump", "", "", "", ""), lbl)
            dfj.insert(0, "Side", np.tile(jump.side, dfj.shape[0]))
            raw += [dfj]
        return self._simplify_table(pd.concat(raw))

    def _make_results_plot(self):
        """generate a figure according to the test's data"""

        raw = self._make_results_table()
        raw.loc[raw.index, "Label"] = [
            " ".join([i, v]).replace(" ", "<br>")
            for i, v in zip(raw.Type, raw.Parameter)
        ]
        raw.loc[raw.index, "Side"] = raw.Side.map(lambda x: x + " Jump(s)")
        fig = px.line(
            data_frame=raw,
            x="Time",
            y="Value",
            line_dash="Jump",
            color="Phase",
            facet_row="Label",
            facet_col="Side",
        )
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
        fig.update_traces(opacity=0.33)
        fig.update_yaxes(matches=None, showticklabels=True)
        for row in np.arange(len(raw.Label.unique())):
            fig.update_yaxes(
                matches="y" if row == 0 else f"y{row * 2 + 1}",
                row=row + 1,
            )
        fig.update_xaxes(showticklabels=True)
        fig.update_xaxes(matches="x", col=1)
        fig.update_xaxes(matches="x2", col=2)
        combs = raw[["Unit", "Label"]].drop_duplicates().values[::-1]
        for i, (unit, lbl) in enumerate(combs):
            fig.update_yaxes(title=unit, row=i + 1, col=1)

        # update the layout and return
        fig.update_layout(
            template="simple_white",
            height=300 * len(raw.Label.unique()),
            width=1200,
        )
        return go.FigureWidget(fig)

    def _make_summary_table(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ):
        """
        return a table highlighting the test summary and a table reporting
        the summary data.

        Parameters
        ----------
        normative_intervals: pd.DataFrame, optional
            all the normative intervals. The dataframe must have the following
            columns:

                Parameter: str
                    the name of the parameter

                Rank: str
                    the label defining the interpretation of the value

                Lower: int | float
                    the lower bound of the interval.

                Upper: int | float
                    the upper bound of the interval.

                Color: str
                    code that can be interpreted as a color.

        Returns
        -------
        table: pd.DataFrame
            return the summary table
        """
        # get the EMG norms and user weight
        weight = self.baseline.weight

        # get the features for each jump
        out = []
        for j, jump in enumerate(self.right_jumps):
            jump_df = self._get_single_jump_feats(jump, weight)
            jump_df.insert(0, "Jump", np.tile(f"Jump {j + 1}", jump_df.shape[0]))
            jump_df.insert(0, "Side", np.tile(jump.side, jump_df.shape[0]))
            out += [jump_df]
        for j, jump in enumerate(self.left_jumps):
            jump_df = self._get_single_jump_feats(jump, weight)
            jump_df.insert(0, "Jump", np.tile(f"Jump {j + 1}", jump_df.shape[0]))
            jump_df.insert(0, "Side", np.tile(jump.side, jump_df.shape[0]))
            out += [jump_df]
        out = pd.concat(out, ignore_index=True)

        # add the normative bands
        for (jump, param), dfr in out.groupby(["Jump", "Parameter"]):
            if normative_intervals.shape[0] > 0:
                idx = normative_intervals.Parameter == str(param)
                norms = normative_intervals.loc[idx]
                for row in np.arange(norms.shape[0]):
                    rnk, low, upp, clr = norms.iloc[row].values.flatten()
                    val = dfr.Value.values.astype(float)
                    if val >= low and val <= upp:
                        out.loc[dfr.index, "Rank"] = rnk
                        out.loc[dfr.index, "Color"] = clr

        return out

    def _make_summary_plot(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ):
        """
        return a dictionary of plotly FigureWidget objects highlighting the
        test summary and a table reporting the summary data.

        Parameters
        ----------
        normative_intervals: pd.DataFrame, optional
            all the normative intervals. The dataframe must have the following
            columns:

                Parameter: str
                    the name of the parameter

                Rank: str
                    the label defining the interpretation of the value

                Lower: int | float
                    the lower bound of the interval.

                Upper: int | float
                    the upper bound of the interval.

                Color: str
                    code that can be interpreted as a color.

        Returns
        -------
        figures: dict[str, FigureWidget]
            return a dictionary of plotly FigureWidget objects summarizing the
            results of the test.
        """
        # get the summary data
        data = self._make_summary_table(normative_intervals)

        # build the output figures
        figures: dict[str, go.FigureWidget] = {}
        for parameter in data.Parameter.unique():

            # get the data and the normative bands
            dfr = data.loc[data.Parameter == parameter]
            if normative_intervals.shape[0] > 0:
                idx = normative_intervals.Parameter == parameter
                norms = normative_intervals[idx]
            else:
                norms = normative_intervals

            # get a bar plot with optional normative bands
            fig = bars_with_normative_bands(
                data_frame=dfr,
                yarr="Jump" if "Imbalance" in parameter else "Value",
                xarr="Value" if "Imbalance" in parameter else "Jump",
                patterns="Side",
                orientation="h" if "Imbalance" in parameter else "v",
                unit=dfr.Unit.values[0],
                intervals=norms,  # type: ignore
            )[0]
            fig.update_layout(title=parameter, template="simple_white")
            figures[parameter] = go.FigureWidget(fig)

        return figures

    # * constructors

    def __init__(
        self,
        baseline: UprightStance,
        left_jumps: list[SideJump],
        right_jumps: list[SideJump],
    ):
        # check for the jumps
        if not isinstance(left_jumps, list):
            raise ValueError("'left_jumps' must be a list of SideJump objects.")
        if not isinstance(right_jumps, list):
            raise ValueError("'right_jumps' must be a list of SideJump objects.")
        for jump in left_jumps + right_jumps:
            if not isinstance(jump, SideJump):
                msg = f"All jumps must be SideJump instances."
                raise ValueError(msg)

        # build
        super().__init__(
            baseline=baseline,
            jumps=left_jumps + right_jumps,  # type: ignore
        )
        self._left_jumps = left_jumps
        self._right_jumps = right_jumps
