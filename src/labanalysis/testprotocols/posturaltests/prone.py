"""static tests module containing Static Tests"""

#! IMPORTS

import numpy as np
import pandas as pd

from ...constants import G
from os.path import dirname, join

from ..frames import StateFrame
from .upright import UprightStance
from ..base import LabTest
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

__all__ = ["ProneStance", "PlankTest"]


#! CLASSES


class ProneStance(UprightStance):
    """
    class defining a static prone stance.

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

    weight
        the bodyweight of the subject in kg

    emg_norms
        descriptive statistics of the processed emg amplitude

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
    def weight(self):
        """the subject's weight in kg"""

        # warn about data processing
        self._check_processed()

        # return the outcomes
        grf = self.forceplatforms.fRes.FORCE.Z.values.astype(float)
        return float(np.median(grf)) / G

    # * methods

    def _check_inputs(self):
        """check the validity of the entered data"""
        # ensure that the 'rFoot', 'lFoot', 'rHand', 'lHand' and 'fRes'
        # force platform objects exist
        lbls = np.unique(self.forceplatforms.columns.get_level_values(0))
        required_fp = ["lFoot", "rFoot", "lHand", "rHand", "fRes"]
        for lbl in required_fp:
            if not any([i == lbl for i in lbls]):
                msg = f"the data does not contain the required '{lbl}'"
                msg += " forceplatform object."
                raise ValueError(msg)

        # self._forceplatforms = self._forceplatforms[required_fp]

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
        generate an instance of a prone stance object

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
        forces_fcut: int | float | None = 50,
        emgs_fband: tuple[int | float, int | float] | None = (30, 400),
        emgs_rms_win: int | float | None = 0.2,
    ):
        """
        generate the object from a .tdf file

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
        frame: ProneStance
            a ProneStance instance of the data contained in the .tdf file.

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
        generate the object from a StateFrame

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        Returns
        -------
        frame: ProneStance
            a ProneStance instance.
        """
        return super().from_stateframe(obj)


class PlankTest(ProneStance, LabTest):
    """
    Class handling the data processing and analysis of the collected data about
    a plank test.

    Parameters
    ----------
    stance: ProneStance
        a ProneStance object

    Attributes
    ----------
    results_table
        a table containing the metrics resulting from each jump

    summary_table
        A table with summary statistics about the test.

    summary_plot
        a plotly FigureWidget summarizing the results of the test

    to_pronestance
        return the object as ProneStance instance
    """

    # * attributes

    @property
    def results_table(self):
        """Return a table containing the test results."""
        # take a dataframe of the available data and remove the unnecessary
        # elements
        fps = self.forceplatforms
        valid_columns = []
        for column in fps.columns:
            if ((column[1] == "ORIGIN") and (column[2] in ["X", "Y"])) or (
                (column[1] == "FORCE") and (column[2] == "Z")
            ):
                valid_columns += [column]
        fps = fps[valid_columns]

        # normalize the force to the bodyweight
        forces = fps.loc[fps.index, [i for i in fps.columns if i[1] == "FORCE"]]
        fps.loc[fps.index, forces.columns] = forces.values / self.weight / G * 100
        coords = fps.loc[fps.index, [i for i in fps.columns if i[1] == "ORIGIN"]]
        fps.loc[fps.index, coords.columns] = coords.values * 1000
        cols = []
        for i in fps.columns:
            if i[1] == "ORIGIN":
                cols += [(i[0], "COP", i[2], "mm")]
            else:
                cols += [(i[0], "GRF", i[2], "% Weight")]
        fps.columns = pd.MultiIndex.from_tuples(cols)
        return fps

    @property
    def summary_table(self):
        """Return a table with summary statistics about the test."""
        # generate a long format table
        res = self.results_table

        # get the load distribution
        forces = res[[i for i in res.columns if i[1] == "GRF" and i[0] != "fRes"]]
        forces.columns = pd.Index([i[0] for i in forces.columns])
        upper = forces.rHand + forces.lHand
        lower = forces.rFoot + forces.lFoot
        left = forces.rHand + forces.rFoot
        right = forces.lHand + forces.lFoot
        out = [
            {
                "Parameter": "Weight distribution",
                "Side": "Upper Body",
                "Unit": "%",
                "Value": upper.mean(),
            },
            {
                "Parameter": "Weight distribution",
                "Side": "Lower Body",
                "Unit": "%",
                "Value": lower.mean(),
            },
            {
                "Parameter": "Weight distribution",
                "Side": "Left",
                "Unit": "%",
                "Value": left.mean(),
            },
            {
                "Parameter": "Weight distribution",
                "Side": "Right",
                "Unit": "%",
                "Value": right.mean(),
            },
            {
                "Parameter": "CoP Lateral Displacement",
                "Side": "-",
                "Unit": "mm",
                "Value": res.fRes.COP.mean(axis=0).X.values[0],
            },
        ]

        # stability
        pos = res[[i for i in res.columns if i[1] == "COP" and i[0] == "fRes"]]
        pos.columns = pd.Index(["X", "Y"])
        avg = pos[["X", "Y"]].mean(axis=0)
        stability = (((pos[["X", "Y"]] - avg) ** 2).sum(axis=1) ** 0.5).mean()
        stability = {
            "Parameter": "Stability",
            "Unit": "mm",
            "Value": stability,
        }
        out += [stability]

        # wrap up and return
        out = [pd.DataFrame(pd.Series(i)).T for i in out]
        out = pd.concat(out, ignore_index=True)

        return out

    @property
    def summary_plots(self):
        """return dict with summary figures highlighting the test results"""

        # plot the sway
        results = self.results_table
        cop_coords = results.fRes.COP
        cop_coords.columns = pd.Index([i[0] for i in cop_coords])
        cop_coords.loc[cop_coords.index, "Y"] -= cop_coords.Y.mean()
        rng1 = [cop_coords.min().min(), cop_coords.max().max()]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cop_coords.X.values.astype(float).flatten(),
                y=cop_coords.Y.values.astype(float).flatten(),
                name="Weight Displacement",
                mode="lines",
                line_color=px.colors.qualitative.Plotly[0],
                opacity=0.5,
                line_width=2,
            ),
        )

        # plot the mean
        avg = cop_coords.mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=[avg.X],
                y=[avg.Y],
                name="Mean Position",
                mode="markers",
                marker_color=px.colors.qualitative.Plotly[4],
                opacity=1,
                marker_size=20,
            ),
        )

        # add the vertical and horizontal axes
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_width=2,
            line_color="black",
            opacity=0.3,
            showlegend=False,
            col=1,  # type: ignore
            row=1,  # type: ignore
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_width=2,
            line_color="black",
            opacity=0.3,
            showlegend=False,
            col=1,  # type: ignore
            row=1,  # type: ignore
        )

        # get the normative values
        norm_file = join(dirname(dirname(__file__)), "normative_values.xlsx")
        norms = pd.read_excel(io=norm_file, sheet_name="PlankTest")
        xref, yref = norms[["X", "Y"]].values.astype(float).flatten()

        # get the figure ranges
        amax = cop_coords.abs().max(axis=0).max()
        amax = float(np.max([amax, avg.Y + yref, avg.X + xref]))
        rng = [-amax, amax]

        # plot the normative bands
        colors = {
            "Poor": px.colors.qualitative.Plotly[1],
            "Normal": px.colors.qualitative.Plotly[2],
        }
        fig.add_shape(
            type="rect",
            x0=rng[0],
            x1=rng[1],
            y0=rng[0],
            y1=rng[1],
            fillcolor=colors["Poor"],
            showlegend=True,
            line_width=0,
            name="Poor stability range",
            opacity=0.1,
        )
        fig.add_shape(
            type="circle",
            x0=avg.X - xref,
            x1=avg.X + xref,
            y0=avg.Y - yref,
            y1=avg.Y + yref,
            fillcolor=colors["Normal"],
            showlegend=True,
            line_width=None,
            name="Normal stability range",
            opacity=0.3,
        )

        # update the layout
        ntraces = int(len(fig.data))  # type: ignore
        for i in range(ntraces):
            fig.data[i].update(zorder=int(ntraces - i - 1))  # type: ignore
        fig.update_layout(
            template="simple_white",
            title="Plank Test",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            legend=dict(
                x=0.95,
                y=0.95,
                xanchor="right",
                yanchor="top",
            ),
            height=800,
            width=800,
        )
        fig.update_xaxes(range=rng, title="mm")
        fig.update_yaxes(range=rng, title="mm")

        # plot the left/right distribution
        summ = self.summary_table
        symm = summ.loc[summ.Parameter == "CoP Lateral Displacement"]

        return {"Sway": go.FigureWidget(fig)}

    # * methods

    def _check_inputs(self):
        """check the validity of the entered data"""
        # ensure that the 'rFoot', 'lFoot', 'rHand', 'lHand' and 'fRes'
        # force platform objects exist
        lbls = np.unique(self.forceplatforms.columns.get_level_values(0))
        required_fp = ["lFoot", "rFoot", "lHand", "rHand", "fRes"]
        for lbl in required_fp:
            if not any([i == lbl for i in lbls]):
                msg = f"the data does not contain the required '{lbl}'"
                msg += " forceplatform object."
                raise ValueError(msg)

        # self._forceplatforms = self._forceplatforms[required_fp]

    def to_pronestance(self):
        """return the object as ProneStance instance"""
        return ProneStance.from_stateframe(self.to_stateframe())

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
        generate a Plank Test instance

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
            process_data=process_data,
            ignore_index=ignore_index,
            markers_fcut=markers_fcut,
            forces_fcut=forces_fcut,
            emgs_fband=emgs_fband,
            emgs_rms_win=emgs_rms_win,
        )

        # rotate the data and center it to the mid-point between all the
        # contact points
        rhand = self.forceplatforms["rHand"]["ORIGIN"].median(axis=0).values
        lhand = self.forceplatforms["lHand"]["ORIGIN"].median(axis=0).values
        rfoot = self.forceplatforms["rFoot"]["ORIGIN"].median(axis=0).values
        lfoot = self.forceplatforms["lFoot"]["ORIGIN"].median(axis=0).values
        origin = (rhand + lhand + rfoot + lfoot) / 4
        self.to_reference_frame(
            origin=origin,
            axis1=(rhand + rfoot) / 2 - origin,
            axis2=(rhand + lhand) / 2 - origin,
            axis3=[0, 1, 0],
            inplace=True,
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
        generate the object from a .tdf file

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
        frame: ProneStance
            a ProneStance instance of the data contained in the .tdf file.

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
        generate the object from a StateFrame

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        Returns
        -------
        frame: ProneStance
            a ProneStance instance.
        """
        return super().from_stateframe(obj)

    @classmethod
    def from_pronestance(cls, obj: ProneStance):
        """
        generate the object from a StateFrame

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        Returns
        -------
        frame: ProneStance
            a ProneStance instance.
        """
        if not isinstance(obj, ProneStance):
            raise ValueError("'obj' must be a ProneStance instance.")
        return super().from_stateframe(obj.to_stateframe())
