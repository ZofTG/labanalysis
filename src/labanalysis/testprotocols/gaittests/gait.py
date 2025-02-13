"""basic gait module"""

#! IMPORTS


import warnings
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)

from ... import signalprocessing as labsp
from ...plotting.plotly import bars_with_normative_bands
from ..base import LabTest
from ..frames import StateFrame

#! CONSTANTS


__all__ = [
    "GaitCycle",
    "GaitTest",
]


#! CLASSESS


class GaitObject(StateFrame):
    """
    basic gait object class.

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

    # * class variables

    _algorithm: Literal["kinematics", "kinetics"]
    _left_heel: pd.DataFrame | None
    _right_heel: pd.DataFrame | None
    _left_toe: pd.DataFrame | None
    _right_toe: pd.DataFrame | None
    _left_meta_head: pd.DataFrame | None
    _right_meta_head: pd.DataFrame | None
    _grf: pd.DataFrame | None
    _cop: pd.DataFrame | None
    _vertical_axis: Literal["X", "Y", "Z"]
    _antpos_axis: Literal["X", "Y", "Z"]
    _grf_threshold: float
    _height_threshold: float

    # * attributes

    @property
    def algorithm(self):
        """return the gait cycle detection algorithm"""
        return self._algorithm

    @property
    def init_s(self):
        """return the first toeoff time in seconds"""
        return float(self.to_dataframe().index.to_list()[0])

    @property
    def end_s(self):
        """return the toeoff time corresponding to the end of the cycle in seconds"""
        return float(self.to_dataframe().index.to_list()[-1])

    @property
    def grf(self):
        """return the ground reaction force data"""
        return self._grf

    @property
    def cop(self):
        """return the center of pressure data"""
        return self._cop

    @property
    def left_heel(self):
        """return the left heel label"""
        return self._left_heel

    @property
    def right_heel(self):
        """return the right heel label"""
        return self._right_heel

    @property
    def left_toe(self):
        """return the left toe label"""
        return self._left_toe

    @property
    def right_toe(self):
        """return the right toe label"""
        return self._right_toe

    @property
    def left_meta_head(self):
        """return the left metatarsal head label"""
        return self._left_meta_head

    @property
    def right_meta_head(self):
        """return the right metatarsal head label"""
        return self._right_meta_head

    @property
    def vertical_axis(self):
        """return the vertical axis"""
        return self._vertical_axis

    @property
    def antpos_axis(self):
        """return the anterior-posterior axis"""
        return self._antpos_axis

    @property
    def grf_threshold(self):
        """return the grf threshold"""
        return self._grf_threshold

    @property
    def height_threshold(self):
        """return the height threshold"""
        return self._height_threshold

    # * methods

    def _get_marker(self, label: str | None):
        """check if a label is available in the kinematic data and return it"""
        if label is not None and not isinstance(label, str):
            raise ValueError("'label' must be a string.")
        if label in self.markers.columns.get_level_values(0).unique():
            return pd.DataFrame(self.markers[[label]])
        return None

    def _get_forcevector(self, label: str | None):
        """check if a label is available in the forceplatforms data and return it"""
        if label is not None and not isinstance(label, str):
            raise ValueError("'label' must be a string.")
        if label in self.forceplatforms.columns.get_level_values(0).unique():
            grf = pd.DataFrame(self.forceplatforms[label].FORCE)
            cols = [(label,) + i for i in grf.columns.to_list()]  # type: ignore
            grf.columns = pd.MultiIndex.from_tuples(cols)
            cop = pd.DataFrame(self.forceplatforms[label].ORIGIN)
            cols = [(label,) + i for i in cop.columns.to_list()]  # type: ignore
            cop.columns = pd.MultiIndex.from_tuples(cols)
        return grf, cop

    def _filter_kinetics(self, grf: np.ndarray, time: np.ndarray):
        """filter the ground reaction force signal"""
        fsamp = float(1 / np.mean(np.diff(time)))
        grff = labsp.fillna(grf, value=0).astype(float).flatten()  # type: ignore
        grff = labsp.butterworth_filt(
            arr=grff.astype(float).flatten(),
            fcut=[10, 100],
            fsamp=fsamp,
            order=4,
            ftype="bandstop",
            phase_corrected=True,
        )
        return grff.astype(float).flatten()

    def _filter_kinematics(self, coord: np.ndarray, time: np.ndarray):
        """filter vertical coordinates from kinematic data"""
        fsamp = float(1 / np.mean(np.diff(time)))
        fcoord = labsp.fillna(coord).astype(float).flatten()  # type: ignore
        fcoord = labsp.butterworth_filt(
            arr=np.array([fcoord - np.min(fcoord)]).astype(float).flatten(),
            fcut=6,
            fsamp=fsamp,
            order=4,
            ftype="lowpass",
            phase_corrected=True,
        )
        return fcoord.astype(float).flatten()

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """set the gait cycle detection algorithm"""
        algorithms = ["kinematics", "kinetics"]
        if not isinstance(algorithm, str) or algorithm not in algorithms:
            msg = "'algorithm' must be any between 'kinematics' or 'kinetics'."
            raise ValueError(msg)
        algo = algorithm
        if (
            algo == "kinetics"
            and self.grf is None
            and all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"'forceplatforms data' not found. The 'algorithm' option"
            msg += " has been set to 'kinematics'."
            warnings.warn(msg)
            algo = "kinematics"
        elif (
            algo == "kinematics"
            and self.grf is not None
            and not all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"Not all left_heel, right_heel, left_toe and right_toe"
            msg += " markers have been found to run the 'kinematics' algorithm."
            msg += " The 'kinetics' algorithm has therefore been selected."
            warnings.warn(msg)
            algo = "kinetics"
        elif self.grf is None and any(
            [
                self.left_heel is None,
                self.left_toe is None,
                self.right_heel is None,
                self.right_toe is None,
            ]
        ):
            msg = "Neither ground reaction force nor left_heel, right_heel, "
            msg += "left_toe and right_toe markers have been found."
            msg += " Therefore none of the available algorithms can be used."
            raise ValueError(msg)

        self._algorithm = algo

    def set_grf_threshold(self, threshold: float | int):
        """set the grf threshold"""
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._grf_threshold = float(threshold)

    def set_height_threshold(self, threshold: float | int):
        """set the height threshold"""
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._height_threshold = float(threshold)

    def set_vertical_axis(self, axis: Literal["X", "Y", "Z"]):
        """set the vertical axis"""
        if axis not in ["X", "Y", "Z"]:
            raise ValueError("'axis' must be 'X', 'Y' or 'Z'")
        self._vertical_axis = axis

    def set_antpos_axis(self, axis: Literal["X", "Y", "Z"]):
        """set the anterior-posterior axis"""
        if axis not in ["X", "Y", "Z"]:
            raise ValueError("'axis' must be 'X', 'Y' or 'Z'")
        self._antpos_axis = axis

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
        grf_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
    ):

        # check the frame
        if not isinstance(frame, StateFrame):
            raise ValueError("'frame' must be a StateFrame instance.")

        # initialize the object
        super().__init__(
            markers_raw=frame.markers,
            forceplatforms_raw=frame.forceplatforms,
            emgs_raw=frame.emgs,
        )
        self._processed = frame.is_processed()
        self._marker_processing_options = frame.marker_processing_options
        self._forceplatform_processing_options = frame.forceplatform_processing_options
        self._emg_processing_options = frame.emg_processing_options

        # set the labels
        self._left_heel = self._get_marker(left_heel)
        self._right_heel = self._get_marker(right_heel)
        self._left_toe = self._get_marker(left_toe)
        self._right_toe = self._get_marker(right_toe)
        self._left_meta_head = self._get_marker(left_meta_head)
        self._right_meta_head = self._get_marker(right_meta_head)
        self._grf, self._cop = self._get_forcevector(grf)

        # set the thresholds
        if not isinstance(grf_threshold, (int, float)):
            raise ValueError("'grf_threshold' must be a float or int")
        self._grf_threshold = float(grf_threshold)
        if not isinstance(height_threshold, (int, float)):
            raise ValueError("'height_threshold' must be a float or int")
        self._height_threshold = float(height_threshold)

        # set the vertical and antpos axis
        if vertical_axis not in ["X", "Y", "Z"]:
            raise ValueError("'vertical_axis' must be 'X', 'Y' or 'Z'")
        self._vertical_axis = vertical_axis
        if antpos_axis not in ["X", "Y", "Z"]:
            raise ValueError("'antpos_axis' must be 'X', 'Y' or 'Z'")
        self._antpos_axis = antpos_axis

        # check the algorithm option
        self.set_algorithm(algorithm)


class GaitCycle(GaitObject):
    """
    basic gait cycle class.

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
    same foot
    """

    # * class variables

    _side: Literal["LEFT"] | Literal["RIGHT"]
    _footstrike_s: float
    _midstance_s: float

    # * attributes

    @property
    def side(self):
        """return the end time in seconds"""
        return self._side

    @property
    def cycle_time_s(self):
        """return the cycle time in seconds"""
        return self.end_s - self.init_s

    @property
    def footstrike_s(self):
        """return the foot-strike time in seconds"""
        return self._footstrike_s

    @property
    def midstance_s(self):
        """return the mid-stance time in seconds"""
        return self._midstance_s

    @property
    def time_events(self):
        """return all the time events defining the cycle"""
        evts = []
        for lbl in dir(self):
            if lbl.endswith("_s") and not lbl.startswith("_"):
                name = lbl.rsplit("_", 1)[0].strip().split(" ")[0].lower()
                time = getattr(self, lbl)
                perc = time
                if lbl in ["footstrike_s", "midstance_s", "init_s", "end_s"]:
                    perc -= self.init_s
                perc = perc / self.cycle_time_s * 100
                line_abs = {"Parameter": name, "Unit": "s", "Value": time}
                evts += [pd.DataFrame(pd.Series(line_abs)).T]
                line_rel = {"Parameter": name, "Unit": "%", "Value": perc}
                evts += [pd.DataFrame(pd.Series(line_rel)).T]
        out = pd.concat(evts, ignore_index=True)
        out.loc[out.index, "Parameter"] = out.Parameter.map(
            lambda x: x.replace("_time", "")
        )
        return out

    # * methods

    def _footstrike_kinetics(self) -> float:
        """return the foot-strike time in seconds using the kinetics algorithm"""
        raise NotImplementedError

    def _footstrike_kinematics(self) -> float:
        """return the foot-strike time in seconds using the kinematics algorithm"""
        raise NotImplementedError

    def _midstance_kinetics(self) -> float:
        """return the mid-stance time in seconds using the kinetics algorithm"""
        raise NotImplementedError

    def _midstance_kinematics(self) -> float:
        """return the mid-stance time in seconds using the kinematics algorithm"""
        raise NotImplementedError

    def _update_events(self):
        """update gait events"""
        if self.algorithm == "kinetics":
            try:
                self._midstance_s = self._midstance_kinetics()
            except Exception:
                self._midstance_s = np.nan
            try:
                self._footstrike_s = self._footstrike_kinetics()
            except Exception:
                self._footstrike_s = np.nan
        elif self.algorithm == "kinematics":
            try:
                self._midstance_s = self._midstance_kinematics()
            except Exception:
                self._midstance_s = np.nan
            try:
                self._footstrike_s = self._footstrike_kinematics()
            except Exception:
                self._footstrike_s = np.nan

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """set the gait cycle detection algorithm"""
        super().set_algorithm(algorithm=algorithm)
        self._update_events()

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
        grf_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
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

        # check the side
        if not isinstance(side, str):
            raise ValueError("'side' must be 'LEFT' or 'RIGHT'")
        if side in ["LEFT", "RIGHT"]:
            self._side = side

        # update the gait events
        self._update_events()


class GaitTest(GaitObject, LabTest):
    """
    detect steps and strides from kinematic/kinetic data and extract biofeedback
    info

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

    # * class variables

    _cycles: list[GaitCycle]

    # * attributes

    @property
    def cycles(self):
        """the detected gait cycles"""
        return self._cycles

    # * methods

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
        # get the time events for each step
        evts = []
        for c, cycle in enumerate(self.cycles):
            evt = cycle.time_events
            nsamp = evt.shape[0]
            evt.insert(0, "Side", np.tile(cycle.side, nsamp))
            evt.insert(0, "Algorithm", np.tile(cycle.algorithm, nsamp))
            evt.insert(0, "Event", np.tile(c, nsamp))
            evts += [evt]

        # concatenate the events and get the descriptive stats
        long = pd.concat(evts, ignore_index=True).sort_index(axis=1)

        # add the normative values
        if normative_intervals.shape[0] > 0:
            labels = ["Lower", "Upper", "Rank", "Color"]
            for param, dfr in long.groupby("Parameter"):
                vals = dfr.Value.values.astype(float).flatten()
                for i, val in enumerate(vals):
                    idx = dfr.index[i]
                    bands = self.get_intervals(
                        norms_table=normative_intervals,
                        param=str(param),
                        value=val,
                    )
                    for band in bands:
                        for lbl, val in zip(labels, band):
                            long.loc[idx, lbl] = val

        return long

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
        return NotImplementedError

    def _make_results_table(self):
        """return a table with the resulting data"""
        if len(self.cycles) > 0:
            out = []
            for i, cycle in enumerate(self.cycles):
                dfr = cycle.to_dataframe()
                dfr.insert(0, "Side", np.tile(str(cycle.side), dfr.shape[0]))
                dfr.insert(0, "Cycle", np.tile(i + 1, dfr.shape[0]))
                out += [dfr]
            return pd.concat(out).drop_duplicates().sort_index(axis=0)
        return pd.DataFrame()

    def _make_results_plot(self):
        """
        generate a view with allowing to understand the detected gait cycles
        """
        # get the data to be plotted
        data = []
        labels = [
            "GRF",
            "COP",
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
                if label == "COP":
                    axs = [self.vertical_axis, self.antpos_axis]
                    yaxis = [i for i in ["X", "Y", "Z"] if i not in axs][0]
                else:
                    yaxis = self.vertical_axis
                if label in ["GRF", "COP"]:
                    ffun = self._filter_kinetics
                else:
                    ffun = self._filter_kinematics
                arr = dfr[dfr.columns.get_level_values(0)[0]]
                arr = arr[yaxis].values.astype(float).flatten()
                time = dfr.index.to_numpy()
                filt = ffun(arr, time)
                if label in ["COP"]:
                    arr -= np.nanmean(arr)
                    filt -= np.nanmean(filt)
                elif label not in ["GRF", "COP"]:
                    arr -= np.nanmin(arr)
                unit = dfr.columns.to_list()[0][-1]
                row = {"Raw": arr, "Filtered": filt, "Time": time, "Unit": unit}
                row = pd.DataFrame(row)
                row = row.melt(
                    id_vars=["Time", "Unit"],
                    var_name="Type",
                    value_name="Value",
                )
                row.insert(0, "Source", np.tile(label, row.shape[0]))
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
            title="RunningTest",
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
                init = cycle.init_s
                footstrike = cycle.footstrike_s
                midstance = cycle.midstance_s
                end = cycle.end_s
                fig.add_trace(
                    row=row + 1,
                    col=1,
                    trace=go.Scatter(
                        x=[init, init],
                        y=y_range,
                        line_dash="solid",
                        line_color=color,
                        opacity=0.3,
                        mode="lines",
                        name=f"init ({cycle.side})",
                        showlegend=bool((row == 0) & (i < 2)),
                        legendgroup=f"init ({cycle.side})",
                    ),
                )
                fig.add_trace(
                    row=row + 1,
                    col=1,
                    trace=go.Scatter(
                        x=[end, end],
                        y=y_range,
                        line_dash="dashdot",
                        line_color=color,
                        opacity=0.3,
                        name=f"end ({cycle.side})",
                        mode="lines",
                        showlegend=bool((row == 0) & (i < 2)),
                        legendgroup=f"end ({cycle.side})",
                    ),
                )
                fig.add_trace(
                    row=row + 1,
                    col=1,
                    trace=go.Scatter(
                        x=[footstrike, footstrike],
                        y=y_range,
                        mode="lines",
                        line_dash="dash",
                        line_color=color,
                        opacity=0.3,
                        name=f"footstrike ({cycle.side})",
                        showlegend=bool((row == 0) & (i < 2)),
                        legendgroup=f"footstrike ({cycle.side})",
                    ),
                )
                fig.add_trace(
                    row=row + 1,
                    col=1,
                    trace=go.Scatter(
                        x=[midstance, midstance],
                        y=y_range,
                        mode="lines",
                        line_dash="dot",
                        line_color=color,
                        opacity=0.3,
                        name=f"midstance ({cycle.side})",
                        showlegend=bool((row == 0) & (i < 2)),
                        legendgroup=f"midstance ({cycle.side})",
                    ),
                )

            # plot the thresholds
            if ref == "COP":
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

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """set the gait cycle detection algorithm"""
        super().set_algorithm(algorithm)
        self._cycles = []
        if self.algorithm == "kinetics":
            self._find_cycles_kinetics()
        elif self.algorithm == "kinematics":
            self._find_cycles_kinematics()

    def _find_cycles_kinetics(self) -> None:
        """find the gait cycles using the kinetics algorithm"""
        raise NotImplementedError

    def _find_cycles_kinematics(self) -> None:
        """find the gait cycles using the kinematics algorithm"""
        raise NotImplementedError

    # * constructors

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
        grf_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
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

    @classmethod
    def from_tdf_file(
        cls,
        file: str,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        left_heel: str | None = "lHeel",
        right_heel: str | None = "rHeel",
        left_toe: str | None = "lToe",
        right_toe: str | None = "rToe",
        left_meta_head: str | None = "lMid",
        right_meta_head: str | None = "rMid",
        grf: str | None = "fRes",
        grf_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
    ):
        """
        Generate a GaitTest object directly from a .tdf file.

        Parameters
        ----------
        file: str
            the path to a ".tdf" file.

        algorithm: Literal['kinematics', 'kinetics'] = 'kinematics'
            If algorithm = 'kinematics' and markers are available an algorithm
            based just on kinematic data is adopted to detect the gait cycles.
            To run the kinematics algorithm, the following markers must be
            available:
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
        return cls(
            frame=StateFrame.from_tdf_file(file=file),
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
