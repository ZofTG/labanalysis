"""Side Jump Test module"""

#! IMPORTS


from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
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

    def _make_results_table(self):
        """Return a table containing the test results."""

        def get_jump_results(jump: SideJump):
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

        raw = []
        for i, jump in enumerate(self.left_jumps):
            dfj = get_jump_results(jump)
            lbl = np.tile(f"Jump {i + 1}", dfj.shape[0])
            dfj.insert(0, ("Jump", "", "", "", ""), lbl)
            raw += [dfj]
        for i, jump in enumerate(self.right_jumps):
            dfj = get_jump_results(jump)
            lbl = np.tile(f"Jump {i + 1}", dfj.shape[0])
            dfj.insert(0, ("Jump", "", "", "", ""), lbl)
            raw += [dfj]

        return pd.concat(raw, ignore_index=True)

    def summary(
        self,
        normative_intervals: dict[
            str, dict[str, tuple[int | float, int | float, str]]
        ] = {},
    ):
        """
        return a plotly bar plot highlighting the test summary and a table
        reporting the summary data.

        Parameters
        ----------
        normative_intervals: dict[str, dict[str, tuple[int | float, int | float, str]]]
            one or more key-valued dictionaries defining the properties
            returned by the test. The keys should be:
                "Elevation"
                "Takeoff velocity"
                "<muscle> Imbalance"
            Where <muscle> denotes an (optional) investigated muscle.

            For each key, a dict shall be provided as value having structure:
                band_name: (lower_bound, upper_bound, color)

            Here the upper and lower bounds should be considered as inclusive
            of the provided values, and the color should be a string object
            that can be interpreted as color.

        Returns
        -------
        fig: plotly FigureWidget
            return a plotly FigureWidget object summarizing the results of the
            test.

        tab: pandas DataFrame
            return a pandas dataframe with a summary of the test results.
        """
        # check the inputs
        if not isinstance(normative_intervals, dict):
            raise ValueError("normative_intervals must be a dict object.")

        # get the summary results in long format
        out = []
        for i, jump in enumerate(self.jumps):
            new = self._get_jump_features()
            new.insert(0, "Jump", np.tile(f"Jump {i + 1}", new.shape[0]))
            out += [new]

        res = pd.concat(out, ignore_index=True)

        # build the output figure
        parameters = ["Elevation", "Takeoff Velocity"]
        parameters += [i for i in res.Parameter.unique() if i.endswith("Imbalance")]
        fig = make_subplots(
            rows=1,
            cols=len(parameters),
            subplot_titles=parameters,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.1,
            row_titles=None,
            column_titles=None,
            x_title=None,
            y_title=None,
        )

        # populate the figure
        out = []
        for i, parameter in enumerate(parameters):

            # get the data and the normative bands
            dfr = res.loc[res.Parameter == parameter]
            if any([i == parameter for i in normative_intervals.keys()]):
                norms = normative_intervals[parameter]
            else:
                norms = {}

            # get a bar plot with optional normative bands
            fig0, dfs = bars_with_normative_bands(
                data_frame=dfr,
                yarr="Jump" if parameter.endswith("Imbalance") else "Value",
                xarr="Value" if parameter.endswith("Imbalance") else "Jump",
                orientation="h" if parameter.endswith("Imbalance") else "v",
                unit=dfr.Unit.values[0],
                **norms,  # type: ignore
            )
            dfs.insert(0, "Parameter", np.tile(parameter, dfs.shape[0]))
            out += [dfs]

            # add the figure data and annotations to the proper figure
            for trace in fig0.data:
                fig.add_trace(row=1, col=i + 1, trace=trace)
            for shape in fig0.layout["shapes"]:  # type: ignore
                showlegend = not any(
                    (
                        i["name"] == shape["name"]  # type: ignore
                        for i in fig.layout["shapes"]  # type: ignore
                    )
                )
                shape.update(  # type: ignore
                    legendgroup=shape["name"],  # type: ignore
                    showlegend=showlegend,
                )
            for shape in fig0.layout.shapes:  # type: ignore
                fig.add_shape(shape, row=1, col=i + 1)
            if parameter.endswith("Imbalance"):
                fig.update_xaxes(
                    row=1,
                    col=i + 1,
                    range=fig0.layout["xaxis"].range,  # type: ignore
                )
            else:
                fig.update_yaxes(
                    row=1,
                    col=i + 1,
                    range=fig0.layout["yaxis"].range,  # type: ignore
                )

        return go.FigureWidget(fig), pd.concat(out, ignore_index=True)

    # * constructors

    def __init__(
        self,
        baseline: UprightStance,
        left_jumps: list[SideJump],
        right_jumps: list[SideJump],
    ):
        # check for the jumps
        if not isinstance(self._left_jumps, list):
            raise ValueError("'left_jumps' must be a list of SideJump objects.")
        if not isinstance(self._right_jumps, list):
            raise ValueError("'right_jumps' must be a list of SideJump objects.")
        for jump in self._left_jumps + self._right_jumps:
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
