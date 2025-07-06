"""kinematics module"""

#! IMPORTS


import warnings
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express.colors as colors
import plotly.graph_objects as go
import plotly.express as px

from ... import signalprocessing as labsp
from ...plotting.plotly import bars_with_normative_bands
from ...frames import StateFrame
from . import gait

__all__ = ["WalkingStride", "WalkingTest"]


#! CLASSESS


class WalkingStride(gait.GaitCycle):
    """
    basic walking stride class.

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

    # * class variables

    _opposite_footstrike_s: float
    _absolute_time_events = [
        "footstrike_s",
        "opposite_footstrike_s",
        "midstance_s",
        "init_s",
        "end_s",
    ]

    # * attributes

    @property
    def swing_frame(self):
        """return a stateframe corresponding to the swing phase of the step"""
        return self.slice(self.init_s, self.footstrike_s)

    @property
    def stance_frame(self):
        """return a stateframe corresponding to the contact phase"""
        return self.slice(self.footstrike_s, self.end_s)

    @property
    def swing_time_s(self):
        """return the flight time in seconds"""
        return self.footstrike_s - self.init_s

    @property
    def stance_time_s(self):
        """return the stance time in seconds"""
        return self.end_s - self.footstrike_s

    @property
    def opposite_footstrike_s(self):
        """return the time corresponding to the toeoff of the opposite leg"""
        return self._opposite_footstrike_s

    @property
    def first_double_support_frame(self):
        """return a stateframe corresponding to the first double support phase"""
        return self.slice(self.footstrike_s, self.midstance_s)

    @property
    def first_double_support_time_s(self):
        """return the first double support time in seconds"""
        return self.midstance_s - self.footstrike_s

    @property
    def second_double_support_frame(self):
        """return a stateframe corresponding to the second double support phase"""
        return self.slice(self.opposite_footstrike_s, self.end_s)

    @property
    def second_double_support_time_s(self):
        """return the second double support time in seconds"""
        return self.end_s - self.opposite_footstrike_s

    @property
    def single_support_frame(self):
        """return a stateframe corresponding to the single support phase"""
        return self.slice(self.midstance_s, self.opposite_footstrike_s)

    @property
    def single_support_time_s(self):
        """return the single support time in seconds"""
        return self.opposite_footstrike_s - self.midstance_s

    # * methods

    def _get_grf_positive_crossing_times(self):
        """find the positive crossings over the mean force"""
        # get the ground reaction force
        res = self.resultant_force
        if res is None:
            raise ValueError("no ground reaction force data available.")
        vres = res[res.columns[0][0]][self.vertical_axis]
        time = vres.index.to_numpy()
        vres = vres.values.astype(float).flatten()
        vres = self._filter_kinetics(vres, time)
        vres -= np.nanmean(vres)
        vres /= np.max(vres)

        # get the zero-crossing points
        zeros, signs = labsp.crossings(vres, 0)
        return time[zeros[signs > 0]].astype(float)

    def _footstrike_kinetics(self):
        """find the footstrike time using the kinetics algorithm"""
        positive_zeros = self._get_grf_positive_crossing_times()
        if len(positive_zeros) == 0:
            raise ValueError("no footstrike has been found.")
        return float(positive_zeros[0])

    def _footstrike_kinematics(self):
        """find the footstrike time using the kinematics algorithm"""

        # get the relevant vertical coordinates
        vcoords = {}
        for marker in ["heel", "meta_head"]:
            lbl = f"{self.side.lower()}_{marker}"
            dfr = getattr(self, lbl)
            if dfr is not None:
                dfr = dfr[dfr.columns.get_level_values(0)[0]]
                vcoords[lbl] = dfr[self.vertical_axis]
                vcoords[lbl] = vcoords[lbl].values.astype(float).flatten()

        # filter the signals and extract the first contact time
        time = self.markers.index.to_numpy()
        fs_time = []
        for val in vcoords.values():
            val = self._filter_kinematics(val, time)
            val = val / np.max(val)
            fsi = np.where(val < self.height_threshold)[0]
            if len(fsi) > 0:
                fs_time += [time[fsi[0]]]

        # return
        if len(fs_time) == 0:
            raise ValueError("not footstrike has been found")
        return float(np.min(fs_time))

    def _opposite_footstrike_kinematics(self):
        """find the opposite footstrike time using the kinematics algorithm"""

        # get the opposite leg
        noncontact_foot = "left" if self.side == "RIGHT" else "right"

        # get the relevant vertical coordinates
        vcoords = {}
        for marker in ["heel", "meta_head"]:
            lbl = f"{noncontact_foot}_{marker}"
            dfr = getattr(self, lbl)
            if dfr is not None:
                dfr = dfr[dfr.columns.get_level_values(0)[0]]
                vcoords[lbl] = dfr[self.vertical_axis]
                vcoords[lbl] = vcoords[lbl].values.astype(float).flatten()

        # filter the signals and extract the first contact time
        time = self.markers.index.to_numpy()
        fs_time = []
        for val in vcoords.values():
            val = self._filter_kinematics(val, time)
            val = val / np.max(val)
            fsi = np.where(val >= self.height_threshold)[0]
            if len(fsi) > 0 and fsi[-1] + 1 < len(time):
                fs_time += [time[fsi[-1] + 1]]

        # return
        if len(fs_time) == 0:
            raise ValueError("not opposite footstrike has been found")
        return float(np.min(fs_time))

    def _midstance_kinetics(self):
        """find the midstance time using the kinetics algorithm"""

        # get the anterior-posterior resultant force
        res = self.resultant_force
        if res is None:
            raise ValueError("resultant_force not found")
        res = res[res.columns[0][0]]
        time = res.index.to_numpy()
        res_ap = res[self.anteroposterior_axis].values.astype(float).flatten()
        res_ap = self._filter_kinetics(res_ap, time)
        res_ap -= np.nanmean(res_ap)

        # get the dominant frequency
        fsamp = float(1 / np.mean(np.diff(time)))
        frq, pwr = labsp.psd(res_ap, fsamp)
        ffrq = frq[np.argmax(pwr)]

        # find the local minima
        min_samp = int(fsamp / ffrq / 2)
        mns = labsp.find_peaks(-res_ap, 0, min_samp)
        if len(mns) != 2:
            raise ValueError("no valid mid-stance was found.")
        pk = np.argmax(res_ap[mns[0] : mns[1]]) + mns[0]

        # get the range and obtain the toe-off
        # as the last value occurring before the peaks within the
        # 1 - height_threshold of that range
        thresh = (1 - self.height_threshold) * res_ap[pk]
        loc = np.where(res_ap[pk:] < thresh)[0] + pk
        if len(loc) > 0:
            return float(time[loc[0]])
        raise ValueError("no valid mid-stance was found.")

    def _midstance_kinematics(self):
        """find the midstance time using the kinematics algorithm"""

        # get the minimum height across all foot markers
        vcoord = []
        time = self.markers.index.to_numpy()
        for lbl in ["heel", "meta_head", "toe"]:
            name = f"{self.side.lower()}_{lbl}"
            val = getattr(self, name)
            if val is not None:
                val = val[val.columns[0][0]][self.vertical_axis]
                val = val.values.astype(float).flatten()
                val = self._filter_kinematics(val, time)
                val = val / np.max(val)
                vcoord += [val]
        vcoord = np.vstack(np.atleast_2d(*vcoord)).mean(axis=0)
        idx = np.argmin(vcoord)
        return float(time[idx])

    def _opposite_footstrike_kinetics(self):
        """find the opposite footstrike time using the kinetics algorithm"""
        positive_zeros = self._get_grf_positive_crossing_times()
        if len(positive_zeros) < 2:
            raise ValueError("no opposite footstrike has been found.")
        return float(positive_zeros[1])

    def _update_events(self):
        """update gait events"""
        super()._update_events()
        if self.algorithm == "kinetics":
            try:
                self._opposite_footstrike_s = self._opposite_footstrike_kinetics()
            except Exception:
                self._opposite_footstrike_s = np.nan
        elif self.algorithm == "kinematics":
            try:
                self._opposite_footstrike_s = self._opposite_footstrike_kinematics()
            except Exception:
                self._opposite_footstrike_s = np.nan

    # * constructor

    def __init__(
        self,
        side: Literal["LEFT", "RIGHT"],
        frame: StateFrame,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        left_heel: str | None = "lHeel",
        right_heel: str | None = "rHeel",
        left_toe: str | None = "lToe",
        right_toe: str | None = "rToe",
        left_meta_head: str | None = "lMid",
        right_meta_head: str | None = "rMid",
        grf: str | None = "fRes",
        grf_threshold: float | int = gait.DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = gait.DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
    ):
        super().__init__(
            side=side,
            frame=frame,
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
        )


class WalkingTest(gait.GaitTest):
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

    def _sin_fitted(self, arr: np.ndarray):
        """fit a sine over arr"""
        rfft = np.fft.rfft(arr - np.mean(arr))
        pwr = labsp.psd(arr)[1]
        rfft[pwr < np.max(pwr)] = 0
        return np.fft.irfft(rfft, len(arr))

    def _find_cycles_kinematics(self):
        """find the gait cycles using the kinematics algorithm"""

        # get toe-off times
        time = self.markers.index.to_numpy()
        fsamp = float(1 / np.mean(np.diff(time)))
        for side in ["l", "r"]:

            # get the vertical coordinates of the toe markers
            arr = self.markers[f"{side}_toe"][self.vertical_axis]
            arr = arr.values.astype(float).flatten()

            # filter and rescale
            ftoe = self._filter_kinematics(arr, time)
            ftoe = ftoe / np.max(ftoe)

            # get the minimum reasonable contact time for each step
            frq, pwr = labsp.psd(ftoe, fsamp)
            ffrq = frq[np.argmax(pwr)]
            dsamples = int(round(fsamp / ffrq / 2))

            # get the peaks at each cycle
            pks = labsp.find_peaks(ftoe, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            tos = []
            for pk in pks:
                idx = np.where(ftoe[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    line = pd.Series({"time": time[idx[-1]], "side": side})
                    tos += [pd.DataFrame(line).T]

            # wrap the events
            if len(tos) == 0:
                raise ValueError("no toe-offs have been found.")
            tos = pd.concat(tos, ignore_index=True)
            tos = tos.drop_duplicates()
            tos = tos.sort_values("time")
            tos = tos.reset_index(drop=True)

            # check the alternation of the steps
            for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
                t0 = float(tos.time.values[i0])
                t1 = float(tos.time.values[i1])
                args = {
                    "frame": self.slice(from_time=t0, to_time=t1),
                    "side": "LEFT" if tos.side.values[i0] == "l" else "RIGHT",
                    "grf_threshold": self.grf_threshold,
                    "height_threshold": self.height_threshold,
                    "algorithm": self.algorithm,
                    "vertical_axis": self.vertical_axis,
                    "antpos_axis": self.anteroposterior_axis,
                }

                if self.left_heel is not None:
                    col = self.left_heel.columns.get_level_values(0)[0]
                    args["left_heel"] = col

                if self.right_heel is not None:
                    col = self.right_heel.columns.get_level_values(0)[0]
                    args["right_heel"] = col

                if self.left_toe is not None:
                    col = self.left_toe.columns.get_level_values(0)[0]
                    args["left_toe"] = col

                if self.right_toe is not None:
                    col = self.right_toe.columns.get_level_values(0)[0]
                    args["right_toe"] = col

                if self.left_meta_head is not None:
                    col = self.left_meta_head.columns.get_level_values(0)[0]
                    args["left_meta_head"] = col

                if self.right_meta_head is not None:
                    col = self.right_meta_head.columns.get_level_values(0)[0]
                    args["right_meta_head"] = col

                if self.resultant_force is not None:
                    args["grf"] = self.resultant_force.columns.get_level_values(0)[0]

                self._cycles += [WalkingStride(**args)]  # type: ignore

        # sort the cycles
        cycle_index = np.argsort([i.init_s for i in self.cycles])
        self._cycles = [self.cycles[i] for i in cycle_index]

    def _find_cycles_kinetics(self):
        """find the gait cycles using the kinetics algorithm"""

        # get the anterior-posterior resultant force
        res = self.resultant_force
        if res is None:
            raise ValueError("resultant_force not found")
        res = res[res.columns[0][0]]
        time = res.index.to_numpy()
        res_ap = res[self.anteroposterior_axis].values.astype(float).flatten()
        res_ap = self._filter_kinetics(res_ap, time)
        res_ap -= np.nanmean(res_ap)

        # get the dominant frequency
        fsamp = float(1 / np.mean(np.diff(time)))
        frq, pwr = labsp.psd(res_ap, fsamp)
        ffrq = frq[np.argmax(pwr)]

        # find peaks
        min_samp = int(fsamp / ffrq / 2)
        pks = labsp.find_peaks(res_ap, 0, min_samp)

        # for each peak pair get the range and obtain the toe-off
        # as the last value occurring before the peaks within the
        # 1 - height_threshold of that range
        toi = []
        for pk in pks:
            thresh = (1 - self.height_threshold) * res_ap[pk]
            loc = np.where(res_ap[:pk] < thresh)[0]
            if len(loc) > 0:
                toi += [loc[-1]]

        # get the latero-lateral centre of pressure
        cop = self.centre_of_pressure
        if cop is None:
            raise ValueError("centre_of_pressure not found")
        cop = cop[cop.columns[0][0]]
        known_axes = [self.vertical_axis, self.anteroposterior_axis]
        ml_axis = [i for i in cop.columns if i[0] not in known_axes]
        cop_ml = cop[ml_axis].values.astype(float).flatten()
        cop_ml = self._filter_kinetics(cop_ml, time)
        cop_ml -= np.nanmean(cop_ml)

        # get the sin function best fitting the cop_ml
        sin_ml = self._sin_fitted(cop_ml)

        # get the mean latero-lateral position of each toe-off interval
        cnt = [np.arange(i, j + 1) for i, j in zip(toi[:-1], toi[1:])]
        pos = [np.nanmean(sin_ml[i]) for i in cnt]

        # get the sides
        sides = ["LEFT" if i > 0 else "RIGHT" for i in pos]

        # generate the steps
        toi_evens = toi[0:-1:2]
        sides_evens = sides[0:-1:2]
        toi_odds = toi[1:-1:2]
        sides_odds = sides[1:-1:2]
        for ti, si in zip([toi_evens, toi_odds], [sides_evens, sides_odds]):
            for to, ed, side in zip(ti[:-1], ti[1:], si):
                t0 = float(time[to])
                t1 = float(time[ed])
                args = {
                    "frame": self.slice(from_time=t0, to_time=t1),
                    "side": side,
                    "grf_threshold": self.grf_threshold,
                    "height_threshold": self.height_threshold,
                    "algorithm": self.algorithm,
                    "vertical_axis": self.vertical_axis,
                    "antpos_axis": self.anteroposterior_axis,
                }
                if self.left_heel is not None:
                    col = self.left_heel.columns.get_level_values(0)[0]
                    args["left_heel"] = col
                if self.right_heel is not None:
                    col = self.right_heel.columns.get_level_values(0)[0]
                    args["right_heel"] = col
                if self.left_toe is not None:
                    col = self.left_toe.columns.get_level_values(0)[0]
                    args["left_toe"] = col
                if self.right_toe is not None:
                    col = self.right_toe.columns.get_level_values(0)[0]
                    args["right_toe"] = col
                if self.left_meta_head is not None:
                    col = self.left_meta_head.columns.get_level_values(0)[0]
                    args["left_meta_head"] = col
                if self.right_meta_head is not None:
                    col = self.right_meta_head.columns.get_level_values(0)[0]
                    args["right_meta_head"] = col
                if self.resultant_force is not None:
                    args["grf"] = self.resultant_force.columns.get_level_values(0)[0]
                self._cycles += [WalkingStride(**args)]  # type: ignore

        # sort the cycles
        idx = np.argsort([i.init_s for i in self._cycles])
        self._cycles = [self._cycles[i] for i in idx]

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
        events = [
            "swing",
            "first_double_support",
            "single_support",
            "second_double_support",
            "cycle",
        ]
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
            title="Walking Phases",
            legend_title_text="Phase",
        )

        # create the imbalance plots
        for param in events:

            # get the data and the normative bands
            dfr = data.loc[(data.Parameter == param) & (data.Unit == "s")]
            left = dfr.loc[dfr.Side == "LEFT"].Value.values.astype(float)[0]
            right = dfr.loc[dfr.Side == "RIGHT"].Value.values.astype(float)[0]
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
            dfr = getattr(self, label.lower())
            if dfr is not None:
                if label in ["resultant_force"]:
                    ffun = self._filter_kinetics
                    yaxes = [self.vertical_axis, self.anteroposterior_axis]
                elif label in ["centre_of_pressure"]:
                    ffun = self._filter_kinetics
                    yaxes = [self.vertical_axis, self.anteroposterior_axis]
                    yaxes = [i for i in ["X", "Y", "Z"] if i not in yaxes]
                    yaxes += [self.anteroposterior_axis]
                else:
                    ffun = self._filter_kinematics
                    yaxes = [self.vertical_axis]
                for yaxis in yaxes:
                    arr = dfr[dfr.columns.get_level_values(0)[0]]
                    arr = arr[yaxis].values.astype(float).flatten()
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
        time = self.to_dataframe().index.to_numpy()[[0, -1]]
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
                for n, event in enumerate(cycle.absolute_time_events):
                    time_event = getattr(cycle, event)
                    fig.add_trace(
                        row=row + 1,
                        col=1,
                        trace=go.Scatter(
                            x=[time_event, time_event],
                            y=y_range,
                            line_dash=styles[n % len(styles)],
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
            elif "RESULTANT FORCE" in ref:
                if "VERTICAL" not in ref:
                    thres = 0
                else:
                    thres = np.nan
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
        frame: StateFrame,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        left_heel: str | None = "lHeel",
        right_heel: str | None = "rHeel",
        left_toe: str | None = "lToe",
        right_toe: str | None = "rToe",
        left_meta_head: str | None = "lMid",
        right_meta_head: str | None = "rMid",
        grf: str | None = "fRes",
        grf_threshold: float | int = gait.DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = gait.DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
    ):
        super().__init__(
            frame=frame,
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
        )
