"""kinematics module"""

#! IMPORTS


import warnings
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.express.colors as colors
import plotly.graph_objects as go

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...frames import EMGSignal, ForcePlatform, Point3D, Signal1D, Signal3D
from ...plotting.plotly import bars_with_normative_bands
from ...signalprocessing import find_peaks
from .gait import GaitCycle, GaitTest

__all__ = ["RunningStep", "RunningTest"]


#! CLASSESS


class RunningStep(GaitCycle):
    """
    basic running step class.

    Parameters
    ----------
    side: Literal['LEFT', 'RIGHT']
        the side of the cycle

    frame: StateFrame
        a stateframe object containing all the available kinematic, kinetic
        and emg data related to the cycle

    algorithm: Literal['kinematics', 'kinetics'] = 'kinematics'
        If algorithm = 'kinematics' and markers are available an algorithm
        based just on kinematic data is adopted to detect the gait cycles.
        To run the kinematics algorithm, the following markers must be available:
            - left_heel
            - right_heel
            - left_toe
            - right_toe
            - left_meta_head (Optional)
            - right_meta_head (Optional)
        If algorithm = 'kinetics' and forceplatforms data is available,
        only kinetic data are used to detect the gait cycles. If this is the
        case, the following force vector must be available:
            - grf
        If any of the two algorithms cannot run, a warning is thrown.

    left_heel: str = 'lHeel'
        the left heel label

    right_heel: str = 'rHeel'
        the right heel label

    left_toe: str = 'lToe'
        the left toe label

    right_toe: str = 'rToe'
        the right toe label

    left_meta_head: str | None = 'lMid'
        the left metatarsal head label

    right_meta_head: str | None = 'rMid'
        the right metatarsal head label

    grf: str = 'fRes'
        the ground reaction force label

    grf_threshold: float | int = GRF_THRESHOLD_DEFAULT
        the minimum ground reaction force value (in N) to evaluate the impact
        of one foot on the ground.

    height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT
        the maximum vertical height of one marker (in m) from the ground
        to be assumed in contact with it.

    vertical_axis: Literal['X', 'Y', 'Z'] = 'Y'
        the vertical axis

    antpos_axis: Literal['X', 'Y', 'Z'] = 'Z'
        the anterior-posterior axis

    Note
    ----
    the cycle starts from the toeoff and ends at the next toeoff of the
    same foot.
    """

    @property
    def flight_frame(self):
        """return a stateframe corresponding to the flight phase"""
        return self.slice(self.init_s, self.footstrike_s)

    @property
    def contact_frame(self):
        """return a stateframe corresponding to the contact phase"""
        return self.slice(self.footstrike_s, self.end_s)

    @property
    def loading_response_frame(self):
        """return a stateframe corresponding to the loading response phase"""
        return self.slice(self.footstrike_s, self.midstance_s)

    @property
    def propulsion_frame(self):
        """return a stateframe corresponding to the propulsive phase"""
        return self.slice(self.midstance_s, self.end_s)

    @property
    def flight_time_s(self):
        """return the flight time in seconds"""
        return self.footstrike_s - self.init_s

    @property
    def loadingresponse_time_s(self):
        """return the loading response time in seconds"""
        return self.midstance_s - self.footstrike_s

    @property
    def propulsion_time_s(self):
        """return the propulsion time in seconds"""
        return self.end_s - self.midstance_s

    @property
    def contact_time_s(self):
        """return the contact time in seconds"""
        return self.end_s - self.footstrike_s

    # * methods

    def _footstrike_kinetics(self):
        """find the footstrike time using the kinetics algorithm"""

        # get the contact phase samples
        grf = self.resultant_force
        if grf is None:
            raise ValueError("no ground reaction force data available.")
        vgrf = grf[self.vertical_axis].values.astype(float).flatten()
        time = grf.index.to_numpy()
        grff = self._filter_kinetics(vgrf, time)
        grfn = grff / np.max(grff)
        mask = np.where((grfn < self.height_threshold)[: np.argmax(grfn)])[0]

        # extract the first contact time
        if len(mask) == 0:
            raise ValueError("no footstrike has been found.")

        return float(time[mask[-1]])

    def _footstrike_kinematics(self):
        """find the footstrike time using the kinematics algorithm"""

        # get the relevant vertical coordinates
        vcoords = {}
        contact_foot = self.side.lower()
        for marker in ["heel", "meta_head"]:
            lbl = f"{contact_foot}_{marker}"
            val = self[f"{contact_foot}_{marker}"]
            if val is None:
                continue
            vcoords[lbl] = val[self.vertical_axis].values.astype(float).flatten()

        # filter the signals and extract the first contact time
        time = self.index.to_numpy()
        fs_time = []
        for val in vcoords.values():
            val = self._filter_kinematics(val, time)
            val = val / np.max(val)
            fsi = np.where(val < self.height_threshold)[0]
            if len(fsi) == 0 or fsi[0] == 0:
                raise ValueError("not footstrike has been found.")
            fs_time += [time[fsi[0]]]

        # get output time
        if len(fs_time) > 0:
            return float(np.min(fs_time))
        raise ValueError("no footstrike has been found.")

    def _midstance_kinetics(self):
        """find the midstance time using the kinetics algorithm"""
        grf = self.resultant_force
        if grf is None:
            raise ValueError("no ground reaction force data available.")
        vgrf = grf[self.vertical_axis].values.astype(float).flatten()
        time = grf.index.to_numpy()
        grff = self._filter_kinetics(vgrf, time)
        return float(time[np.argmax(grff)])

    def _midstance_kinematics(self):
        """find the midstance time using the kinematics algorithm"""
        # get the available markers
        lbls = [f"{self.side.lower()}_{i}" for i in ["heel", "toe"]]
        meta_lbl = f"{self.side.lower()}_metatarsal_head"
        meta_dfr = getattr(self, meta_lbl)
        if meta_dfr is not None:
            lbls += [meta_lbl]

        # get the mean vertical signal
        time = self.index.to_numpy()
        ref = np.zeros_like(time)
        for lbl in lbls:
            val = self[lbl][self.vertical_axis]
            val = val.values.astype(float).flatten()
            ref += self._filter_kinematics(val, time)
        ref /= len(lbls)

        # return the time corresponding to the minimum value
        return float(time[np.argmin(val)])

    # * constructor

    def __init__(
        self,
        side: Literal["left", "right"],
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_meta_head: Point3D | None = None,
        right_meta_head: Point3D | None = None,
        grf: ForcePlatform | None = None,
        grf_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        super().__init__(
            side=side,
            algorithm=algorithm,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_meta_head=left_meta_head,
            right_meta_head=right_meta_head,
            grf=grf,
            grf_threshold=grf_threshold,
            height_threshold=height_threshold,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,
            **extra_signals,
        )


class RunningTest(GaitTest):
    """
    generate a RunningTest instance

    Parameters
    ----------
    frame: StateFrame
        a stateframe object containing all the available kinematic, kinetic
        and emg data related to the cycle

    algorithm: Literal['kinematics', 'kinetics'] = 'kinematics'
        If algorithm = 'kinematics' and markers are available an algorithm
        based just on kinematic data is adopted to detect the gait cycles.
        To run the kinematics algorithm, the following markers must be available:
            - left_heel
            - right_heel
            - left_toe
            - right_toe
            - left_meta_head (Optional)
            - right_meta_head (Optional)
        If algorithm = 'kinetics' and forceplatforms data is available,
        only kinetic data are used to detect the gait cycles. If this is the
        case, the following force vector must be available:
            - grf
        If any of the two algorithms cannot run, a warning is thrown.

    left_heel: str = 'lHeel'
        the left heel label

    right_heel: str = 'rHeel'
        the right heel label

    left_toe: str = 'lToe'
        the left toe label

    right_toe: str = 'rToe'
        the right toe label

    left_meta_head: str | None = 'lMid'
        the left metatarsal head label

    right_meta_head: str | None = 'rMid'
        the right metatarsal head label

    grf: str = 'fRes'
        the ground reaction force label

    grf_threshold: float | int = GRF_THRESHOLD_DEFAULT
        the minimum ground reaction force value (in N) to evaluate the impact
        of one foot on the ground.

    height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT
        the maximum vertical height of one marker (in m) from the ground
        to be assumed in contact with it.

    vertical_axis: Literal['X', 'Y', 'Z'] = 'Y'
        the vertical axis

    antpos_axis: Literal['X', 'Y', 'Z'] = 'Z'
        the anterior-posterior axis
    """

    # * methods

    def _find_cycles_kinematics(self):
        """find the gait cycles using the kinematics algorithm"""

        # get toe-off times
        tos = []
        time = self.index.to_numpy()
        fsamp = float(1 / np.mean(np.diff(time)))
        for lbl in ["left_toe", "right_toe"]:

            # get the vertical coordinates of the toe markers
            obj = self[lbl]
            if obj.shape[1] == 0:
                raise ValueError(f"{lbl} is missing.")
            arr = obj[self.vertical_axis].values.astype(float).flatten()

            # filter and rescale
            ftoe = self._filter_kinematics(arr, time)
            ftoe = ftoe / np.max(ftoe)

            # get the minimum reasonable contact time for each step
            dsamples = int(round(fsamp / 2))

            # get the peaks at each cycle
            pks = find_peaks(ftoe, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            side = lbl.split("_")[0]
            for pk in pks:
                idx = np.where(ftoe[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    line = pd.Series({"Time": time[idx[-1]], "Side": side})
                    tos += [pd.DataFrame(line).T]

        # wrap the events
        if len(tos) == 0:
            raise ValueError("no toe-offs have been found.")
        tos = pd.concat(tos, ignore_index=True)
        tos = tos.drop_duplicates()
        tos = tos.sort_values("Time")
        tos = tos.reset_index(drop=True)

        # check the alternation of the steps
        sides = tos.Side.values
        if not all(s0 != s1 for s0, s1 in zip(sides[:-1], sides[1:])):
            warnings.warn("Left-Right steps alternation not guaranteed.")
        for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
            t0 = float(tos.Time.values[i0])
            t1 = float(tos.Time.values[i1])
            step = self.slice(t0, t1)
            args = {
                "side": tos.Side.values[i1],
                "grf_threshold": self.grf_threshold,
                "height_threshold": self.height_threshold,
                "algorithm": self.algorithm,
                "vertical_axis": self.vertical_axis,
                "antpos_axis": self.anteroposterior_axis,
            }
            args["left_heel"] = step.left_heel
            args["right_heel"] = step.right_heel
            args["left_toe"] = step.left_toe
            args["right_toe"] = step.right_toe
            args["left_meta_head"] = step.left_meta_head
            args["right_meta_head"] = step.right_meta_head
            args["grf"] = step.ground_reaction_force
            self._cycles += [RunningStep(**step.extra_signals, **args)]  # type: ignore

    def _find_cycles_kinetics(self):
        """find the gait cycles using the kinetics algorithm"""
        if self.ground_reaction_force is None:
            raise ValueError("no ground reaction force data available.")

        # get the grf and the latero-lateral COP
        time = self.ground_reaction_force.index.to_numpy()
        axs = [self.vertical_axis, self.anteroposterior_axis]
        axs = [i for i in ["X", "Y", "Z"] if i not in axs]
        cop_ml = self.centre_of_pressure
        grf = self.resultant_force
        if cop_ml is None or grf is None:
            raise RuntimeError("ground_reaction_force data not found.")
        grf = grf[self.vertical_axis].values.astype(float).flatten()
        cop_ml = cop_ml[axs].values.astype(float).flatten()
        grff = self._filter_kinetics(grf, time)
        mlcf = self._filter_kinetics(cop_ml, time)

        # check if there are flying phases
        flights = grf <= self.grf_threshold
        if not any(flights):
            raise ValueError("No flight phases have been found on data.")

        # get the minimum reasonable contact time for each step
        fsamp = float(1 / np.mean(np.diff(time)))
        dsamples = int(round(fsamp / 4))

        # get the peaks in the normalized grf, then return toe-offs and foot
        # strikes
        grfn = grff / np.max(grff)
        toi = []
        fsi = []
        pks = find_peaks(grfn, 0.5, dsamples)
        for pk in pks:
            to = np.where(grfn[pk:] < self.height_threshold)[0]
            fs = np.where(grfn[:pk] < self.height_threshold)[0]
            if len(fs) > 0 and len(to) > 0:
                toi += [to[0] + pk]
                if len(toi) > 1:
                    fsi += [fs[-1]]
        toi = np.unique(toi)
        fsi = np.unique(fsi)

        # get the mean latero-lateral position of each contact
        contacts = [np.arange(i, j + 1) for i, j in zip(fsi, toi[1:])]
        pos = [np.nanmean(mlcf[i]) for i in contacts]

        # get the mean value of alternated contacts and set the step sides
        # accordingly
        evens = np.mean(pos[0:-1:2])
        odds = np.mean(pos[1:-1:2])
        sides = []
        for i in np.arange(len(pos)):
            if evens < odds:
                sides += ["left" if i % 2 == 0 else "right"]
            else:
                sides += ["left" if i % 2 != 0 else "right"]

        for to, ed, side in zip(toi[:-1], toi[1:], sides):
            args = {
                "side": side,
                "grf_threshold": self.grf_threshold,
                "height_threshold": self.height_threshold,
                "algorithm": self.algorithm,
                "vertical_axis": self.vertical_axis,
                "antpos_axis": self.anteroposterior_axis,
            }
            args["left_heel"] = self.left_heel
            args["right_heel"] = self.right_heel
            args["left_toe"] = self.left_toe
            args["right_toe"] = self.right_toe
            args["left_meta_head"] = self.left_meta_head
            args["right_meta_head"] = self.right_meta_head
            args["grf"] = self.ground_reaction_force
            self._cycles += [RunningStep(*self.extra_signals, **args)]  # type: ignore

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

        # get the data
        data = self._make_summary_table()
        data = data.groupby(
            by=["Algorithm", "Parameter", "Unit", "Side"],
            as_index=False,
        )
        data = data.mean()

        # generate the step phases figure
        figures: dict[str, go.FigureWidget] = {}
        cmap = colors.qualitative.Plotly
        figures["phases"] = go.FigureWidget()
        sides = ["RIGHT", "LEFT"]
        events = ["flight", "loadingresponse", "propulsion", "cycle"]
        for i, param in enumerate(events):
            arr = []
            txt = []
            for side in sides:
                dfr = data.loc[(data.Side == side) & (data.Parameter == param)]
                perc = dfr.loc[data.Unit == "%"].Value.values.astype(float)[0]
                time = dfr.loc[data.Unit == "s"].Value.values.astype(float)[0]
                arr += [0.001 if param == "cycle" else time]
                txt += [f"{time:0.3f}<br>({perc:0.1f}%)"]
            trace = go.Bar(
                x=arr,
                y=sides,
                text=txt,
                textposition="inside" if param != "cycle" else "outside",
                name=param,
                legendgroup=param + " time",
                marker_color=cmap[i],
                orientation="h",
            )
            figures["phases"].add_trace(trace)

        # update the layout
        figures["phases"].update_xaxes(title="Time (s)")
        figures["phases"].update_layout(
            barmode="stack",
            template="plotly_white",
            title="Running Phases",
            legend_title_text="Phase",
        )

        # create the imbalance plots
        for param in events:

            # get the data and the normative bands
            dfr = data.loc[(data.Parameter == param) & (data.Unit == "s")]
            left = dfr.loc[dfr.Side == "left"].Value.values.astype(float)[0]
            right = dfr.loc[dfr.Side == "right"].Value.values.astype(float)[0]
            val = 200 * (right - left) / (right + left)
            if normative_intervals.shape[0] > 0:
                idx = normative_intervals.Parameter == param
                norms = normative_intervals[idx]
            else:
                norms = normative_intervals

            # get a bar plot with optional normative bands
            fig = bars_with_normative_bands(
                yarr=[0],
                xarr=[val],
                orientation="h",
                unit="%",
                intervals=norms,  # type: ignore
            )[0]
            fig.update_yaxes(showticklabels=False, visible=False)
            fig.update_layout(title=param, template="simple_white")
            figures[param + " imbalance"] = go.FigureWidget(fig)

        return figures

    def _make_results_plot(self):
        """
        generate a view with allowing to understand the detected gait cycles
        """
        # get the data to be plotted
        data = []
        labels = [
            "resultant_force",
            "centre_of_pressure",
            "left_heel",
            "right_heel",
            "left_toe",
            "right_toe",
            "left_meta_head",
            "right_meta_head",
        ]
        for label in labels:
            dfr = getattr(self, label)
            if dfr is not None:
                if label in ["resultant_force"]:
                    ffun = self._filter_kinetics
                    yaxes = [self.vertical_axis, self.anteroposterior_axis]
                elif label in ["centre_of_pressure"]:
                    ffun = self._filter_kinetics
                    yaxes = [self.vertical_axis, self.anteroposterior_axis]
                    yaxes += [i for i in ["X", "Y", "Z"] if i not in yaxes]
                else:
                    ffun = self._filter_kinematics
                    yaxes = [self.vertical_axis]
                for yaxis in yaxes:
                    arr = dfr[yaxis].values.astype(float).flatten()
                    time = dfr.index.to_numpy()
                    filt = ffun(arr, time)
                    if label in ["centre_of_pressure"]:
                        arr -= np.nanmean(arr)
                        filt -= np.nanmean(filt)
                    elif label not in ["resultant_force", "centre_of_pressure"]:
                        arr -= np.nanmin(arr)
                    unit = dfr.columns.to_list()[0][-1]
                    row = {"Raw": arr, "Filtered": filt, "Time": time, "Unit": unit}
                    row = pd.DataFrame(row)
                    row = row.melt(
                        id_vars=["Time", "Unit"],
                        var_name="Type",
                        value_name="Value",
                    )
                    if yaxis == self.vertical_axis:
                        orientation = "VERTICAL"
                    elif yaxis == self.anteroposterior_axis:
                        orientation = "ANTERIOR-POSTERIOR"
                    else:
                        orientation = "LATERO-LATERAL"
                    source = f'{label.upper().replace("_", " ")}<br>'
                    source += f"({orientation})"
                    row.insert(0, "Source", np.tile(source, row.shape[0]))
                    data.append(row)
        data = pd.concat(data, ignore_index=True)
        labels, units = np.unique(
            data[["Source", "Unit"]].values.astype(str),
            axis=0,
        ).T

        # generate the output figure
        fig = px.line(
            data_frame=data,
            x="Time",
            y="Value",
            color="Type",
            facet_row="Source",
            template="simple_white",
            title=self.name,
            height=300 * len(labels),
        )

        # update the layout
        fig.update_xaxes(showticklabels=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_traces(opacity=0.5)
        fig.update_yaxes(matches=None)

        # add cycles and thresholds
        time = self.index.to_numpy()[[0, -1]]
        for row in np.arange(len(labels)):

            # update the y-axis label
            ref = fig.layout.annotations[row].text  # type: ignore
            unit = str(units[np.where(labels == ref)[0][0]])
            fig.update_yaxes(title_text=unit, row=row + 1)

            # get the y-axis range
            yaxis = "y" + ("" if row == 0 else str(row + 1))
            traces = [i for i in fig.data if i.yaxis == yaxis]  # type: ignore
            minv = np.min([np.nanmin(i.y) for i in traces])  # type: ignore
            maxv = np.max([np.nanmax(i.y) for i in traces])  # type: ignore
            y_range = [minv, maxv]

            # plot the cycles
            for i, cycle in enumerate(self.cycles):
                color = "purple" if cycle.side == "LEFT" else "green"
                styles = ["solid", "dash", "dot", "dashdot"]
                for n, line in cycle.time_events.iterrows():
                    event, unit, value = line.values
                    fig.add_trace(
                        row=row + 1,
                        col=1,
                        trace=go.Scatter(
                            x=[value, value],
                            y=y_range,
                            line_dash=styles[int(n) % len(styles)],  # type: ignore
                            line_color=color,
                            opacity=0.3,
                            mode="lines",
                            name=f"{event} ({cycle.side})",
                            showlegend=bool((row == 0) & (i < 2)),
                            legendgroup=f"{event} ({cycle.side})",
                        ),
                    )

            # plot the thresholds
            if "CENTRE OF PRESSURE" in ref:
                thres = 0
            else:
                thres = self.height_threshold
            thres = float(thres * np.max(y_range))
            fig.add_trace(
                row=row + 1,
                col=1,
                trace=go.Scatter(
                    x=time,
                    y=[thres, thres],
                    mode="lines",
                    line_dash="dot",
                    line_color="black",
                    opacity=0.3,
                    name="Threshold",
                    showlegend=bool(row == 0),
                ),
            )

        return fig

    # * constructor

    def __init__(
        self,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_meta_head: Point3D | None = None,
        right_meta_head: Point3D | None = None,
        grf: ForcePlatform | None = None,
        grf_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        super().__init__(
            algorithm=algorithm,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_meta_head=left_meta_head,
            right_meta_head=right_meta_head,
            grf=grf,
            grf_threshold=grf_threshold,
            height_threshold=height_threshold,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,
            **extra_signals,
        )
