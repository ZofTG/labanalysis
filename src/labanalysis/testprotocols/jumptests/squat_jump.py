"""Squat Jump Test module"""

#! IMPORTS


from os.path import join, dirname
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ... import signalprocessing as sp
from ...constants import G
from ...frames import EMGSignal, ForcePlatform, Point3D, Signal1D, Signal3D, StateFrame
from ...plotting.plotly import bars_with_normative_bands
from ...testing import TestProtocol
from scipy.stats.distributions import norm as normal_distribution

__all__ = ["SquatJump", "SquatJumpTest"]


#! CLASSES


class SquatJump(StateFrame):

    @property
    def grf(self):
        """return the vertical ground reaction force"""
        grfl = self["left_foot"].force[self.vertical_axis]
        grfl = grfl.values.astype(float).flatten()
        grfr = self["right_foot"].force[self.vertical_axis]
        grfr = grfr.values.astype(float).flatten()
        grf = pd.Series(grfr + grfl, index=self.index.to_numpy())
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
        flight_start = self.flight_phase.index.to_numpy()[0]
        time = self.index.to_numpy()
        con_end = float(round(time[time < flight_start][-1], 3))

        # get the last local minima in the filtered vertical position
        # occurring before con_end
        s2 = self["s2"].dropna()
        s2y = s2[self.vertical_axis].values.astype(float).flatten()
        s2t = s2.index.to_numpy()
        fsamp = float(1.0 / np.mean(np.diff(s2t)))
        s2f = sp.butterworth_filt(s2y, 2, fsamp, 6, "lowpass", True)
        mns = time[sp.find_peaks(-s2f)]
        mns = mns[mns < con_end]
        if len(mns) == 0:
            con_start = 0
        else:
            con_start = mns[-1]

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

        # # get the time samples corresponding to the start and end of each
        # batch
        time_start = float(np.round(grft[batch[0]], 3))
        time_stop = float(np.round(grft[batch[-1]], 3))

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
        time = self.index.to_numpy()
        flight_time = self.flight_phase.index.to_numpy()
        time = time[time > flight_time[-1]]
        time_start = float(round(time[0], 3))

        # look at the longest continuous batch in the vertical velocity of
        # S2 with positive sign which occurs after time_start
        s2 = self["s2"].dropna()
        s2y = s2[self.vertical_axis].values.astype(float).flatten()
        s2t = s2.index.to_numpy()
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
    def takeoff_velocity_ms(self):

        # get the ground reaction force during the concentric phase
        con = self.concentric_phase
        con_time = con["s2"].dropna().index.to_numpy()
        con_grf = self.grf.loc[con_time].values.astype(float).flatten()

        # get the output velocity
        weight = con_grf[0] / G
        net_grf = con_grf - weight * G
        return float(np.trapezoid(net_grf, con_time) / weight)

    @property
    def elevation_cm(self):
        flight = self.flight_phase
        yflight = (
            flight["s2"].dropna()[self.vertical_axis].values.astype(float).flatten()
        )
        return 100 * float(np.max(yflight) - yflight[0])

    @property
    def emg_activity(self):

        # get the muscle activations
        emgs = self.concentric_phase.emgsignals
        activations = []
        for (muscle, side, unit), data in emgs.items():
            value = np.trapezoid(
                y=data.values.astype(float).flatten(),
                x=data.index.to_numpy(),
            )
            line = {"muscle": muscle, "side": side, "value": value}
            activations += [pd.DataFrame(pd.Series(line))]

        # add imbalance
        df = pd.DataFrame()
        if len(activations) > 0:
            sub = pd.concat(activations).pivot_table(
                columns=["muscle", "side"],
                values="value",
            )
            for muscle in sub.columns.get_level_values(0):
                vals = sub.loc[sub.index, muscle].values.astype(float)
                lbl = muscle + " Balance (%)"
                df.loc[df.index, lbl] = vals[0] / vals.sum() * 100 - 50

        return df

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
        t_start = self.concentric_phase.index.to_numpy()[0]
        t_start = max(t_start - extra_time_window, 0)
        t_end = self.loading_response_phase.index.to_numpy()[-1]
        t_last = self.index.to_numpy()[-1]
        t_end = min(t_end + extra_time_window, t_last)

        # handle the inplace option
        if not inplace:
            return self.slice(t_start, t_end)
        obj = self.copy()
        return obj.slice(t_start, t_end)

    def __init__(
        self,
        s2: Point3D,
        left_foot: ForcePlatform,
        right_foot: ForcePlatform,
        **signals: Union[Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform],
    ):

        # check the inputs
        if (
            not isinstance(s2, Point3D)
            or not isinstance(left_foot, ForcePlatform)
            or not isinstance(right_foot, ForcePlatform)
        ):
            msg = "s2 must be a Point3D object, while left_foot and right_foot "
            msg += "have to be ForcePlatform objects."
            raise TypeError(msg)

        # build the object
        super().__init__(
            strip=True,
            reset_index=True,
            s2=s2,
            left_foot=left_foot,
            right_foot=right_foot,
            **signals,
        )
        self.resize(inplace=True)


class SquatJumpTest(TestProtocol):
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

    _jumps: list[SquatJump]

    @property
    def jumps(self):
        """return the jumps performed during the test"""
        return self._jumps

    @property
    def normative_values(self):
        filepath = join(dirname(__file__), "squat_jump_references.csv")
        data = pd.read_csv(filepath)
        for row, line in data.iterrows():
            row = int(row)  # type: ignore
            p01, p20, p40, p60, p80, p99 = normal_distribution.ppf(
                [0.01, 0.2, 0.4, 0.6, 0.8, 0.99],
                loc=line.mean,
                scale=line.std,
            )
            data.loc[row, "p1"] = p01
            data.loc[row, "p20"] = p20
            data.loc[row, "p40"] = p40
            data.loc[row, "p60"] = p60
            data.loc[row, "p80"] = p80
            data.loc[row, "p99"] = p99
        return data

    def _make_results_table(self):
        """
        private method used to generate the table required for creating
        the results figure
        """

        # separate into phases
        out = []
        for i, jump in enumerate(self.jumps):
            dfc = jump.concentric_phase.dropna()
            col_depth = len(dfc.columns[0])
            col = ["Phase"] + ["" for i in range(col_depth - 1)]
            col = tuple(*col)
            dfc.insert(0, col, np.tile("Concentric", dfc.shape[0]))
            dff = jump.flight_phase.dropna()
            dff.insert(0, col, np.tile("Flight", dff.shape[0]))
            dfl = jump.loading_response_phase.dropna()
            dfl.insert(0, col, np.tile("Loading Response", dfl.shape[0]))
            dfj = pd.concat([dfc, dff, dfl]).reset_index(drop=True)
            col = ["Time"] + ["" for i in range(col_depth - 2)] + ["s"]
            col = tuple(*col)
            dfj.insert(0, col, dfj.index.to_numpy())

            # flat the columns
            cols = ["_".join([i for i in j if i != ""]) for j in dfj.columns]
            dfj.columns = pd.Index(cols)
            dfj.insert(0, "Jump", f"Jump {i + 1}")
            out += [dfj]

        # melt
        out = pd.concat(out, ignore_index=True).melt(
            id_vars=["Jump", "Time_s", "Phase"],
            var_name="Parameter",
            value_name="Value",
        )

        return out

    def _make_results_plot(self):
        """generate a figure according to the test's data"""
        raw = self._make_results_table()
        fig = px.line(
            data_frame=raw,
            x="Time_s",
            y="Value",
            color="Phase",
            line_dash="Jump",
            facet_row="Parameter",
        )
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
        fig.update_traces(opacity=0.33)
        fig.update_xaxes(showticklabels=True)
        fig.update_yaxes(matches=None, showticklabels=True)

        # update the layout and return
        fig.update_layout(
            template="simple_white",
            height=300 * len(raw.Parameter.unique()),
        )
        return go.Figure(fig)

    def _make_summary_table(self):
        out = []
        for i, jump in enumerate(self.jumps):
            new = jump.emg_activity
            new.insert(0, "Takeoff Velocity (m/s)", [jump.takeoff_velocity_ms])
            new.insert(0, "Elevation (cm)", [jump.elevation_cm])
            new = new.T
            new.columns = pd.Index(["Value"])
            new.insert(0, "Parameter", jump.index)
            new.insert(0, "Jump", f"Jump {i + 1}")
            new.reset_index(inplace=True, drop=True)
            out += [new]
        out = pd.concat(out, ignore_index=True)

        # add normative values
        norms = self.normative_values
        for row, line in out.iterrows():
            param = line.Parameter
            row = int(row)  # type: ignore
            if param in norms.parameter:
                avg, std = (
                    norms.loc[norms.parameter == param][["mean", "std"]]
                    .values.astype(float)
                    .flatten()
                )
                p01, p20, p40, p60, p80, p99 = normal_distribution.ppf(
                    [0.01, 0.2, 0.4, 0.6, 0.8, 0.99], loc=avg, scale=std
                )
                out.loc[row, "min"] = min(p01, line.Value)
                out.loc[row, "poor"] = p20
                out.loc[row, "fair"] = p40
                out.loc[row, "normal"] = p60
                out.loc[row, "good"] = p80
                out.loc[row, "max"] = max(p99, line.Value)
                if line.Value <= p20:
                    out.loc[row, "rank"] = "poor"
                elif line.Value <= p40:
                    out.loc[row, "rank"] = "fair"
                elif line.Value <= p60:
                    out.loc[row, "rank"] = "normal"
                elif line.Value <= p80:
                    out.loc[row, "rank"] = "good"
                else:
                    out.loc[row, "rank"] = "excellent"

        return out

    def _make_summary_plot(self):
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

        # get the data
        data = self._make_summary_table()

        # build the figure
        figures: dict[str, go.Figure] = {}
        for parameter in data.Parameter.unique():

            # get the data and the normative bands
            dfr = data.loc[data.Parameter == parameter]
            norms = []
            if parameter.endswith("Imbalance (%)"):
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Excellent",
                                "Lower": -dfr.p20 + 50,
                                "Upper": dfr.p20 + 50,
                                "Color": "lightblue",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Good",
                                "Lower": dfr.p20 + 50,
                                "Upper": dfr.p40 + 50,
                                "Color": "green",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Good",
                                "Lower": -dfr.p40 + 50,
                                "Upper": -dfr.p20 + 50,
                                "Color": "green",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Normal",
                                "Lower": dfr.p40 + 50,
                                "Upper": dfr.p60 + 50,
                                "Color": "lightgreen",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Normal",
                                "Lower": -dfr.p60 + 50,
                                "Upper": -dfr.p40 + 50,
                                "Color": "lightgreen",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Fair",
                                "Lower": dfr.p60 + 50,
                                "Upper": dfr.p80 + 50,
                                "Color": "orange",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Fair",
                                "Lower": -dfr.p80 + 50,
                                "Upper": -dfr.p60 + 50,
                                "Color": "orange",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Poor",
                                "Lower": dfr.p80 + 50,
                                "Upper": 100,
                                "Color": "lightred",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Poor",
                                "Lower": -100,
                                "Upper": -dfr.p80 + 50,
                                "Color": "lightred",
                            }
                        )
                    )
                ]
            else:
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Poor",
                                "Lower": min(dfr.Value * 0.95, dfr.p01),
                                "Upper": dfr.p20,
                                "Color": "lightred",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Fair",
                                "Lower": dfr.p20,
                                "Upper": dfr.p40,
                                "Color": "orange",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Normal",
                                "Lower": dfr.p40,
                                "Upper": dfr.p60,
                                "Color": "lightgree",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Good",
                                "Lower": dfr.p60,
                                "Upper": dfr.p80,
                                "Color": "green",
                            }
                        )
                    )
                ]
                norms += [
                    pd.DataFrame(
                        pd.Series(
                            {
                                "Rank": "Excellent",
                                "Lower": dfr.p80,
                                "Upper": max(dfr.Value * 1.05, dfr.p99),
                                "Color": "lightblue",
                            }
                        )
                    )
                ]
            norms = pd.concat(norms, ignore_index=True)

            # get a bar plot with optional normative bands
            fig = bars_with_normative_bands(
                data_frame=dfr,
                yarr="Jump" if "Imbalance" in parameter else "Value",
                xarr="Value" if "Imbalance" in parameter else "Jump",
                orientation="h" if "Imbalance" in parameter else "v",
                unit=parameter.split("(")[-1][:-1],
                intervals=norms,  # type: ignore
            )[0]
            fig.update_layout(title=parameter, template="simple_white")
            figures[parameter] = go.Figure(fig)

        return figures

    def __init__(
        self,
        jumps: list[SquatJump],
    ):
        super().__init__()
        self._jumps = jumps
