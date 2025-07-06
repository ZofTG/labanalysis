"""basic gait module"""

#! IMPORTS


from os.path import exists
import warnings
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from ...io.read.btsbioengineering import read_tdf
from ... import signalprocessing as labsp
from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ..base import LabTest
from ...frames import *


#! CONSTANTS


__all__ = ["GaitCycle", "GaitTest"]


#! CLASSESS


class GaitObject:
    """
    basic gait object class.

    Parameters
    ----------
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

    left_heel: Point3D | None
        the left heel label

    right_heel: Point3D | None
        the right heel label

    left_toe: Point3D | None
        the left toe label

    right_toe: Point3D | None
        the right toe label

    left_meta_head: Point3D | None
        the left metatarsal head label

    right_meta_head: Point3D | None
        the right metatarsal head label

    grf: ForcePlatform | None
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

    *extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        additional objects not directly used by the analysis
    """

    # * class variables

    _algorithm: Literal["kinetics", "kinematics"]
    _left_heel: Point3D | None
    _right_heel: Point3D | None
    _left_toe: Point3D | None
    _right_toe: Point3D | None
    _left_meta_head: Point3D | None
    _right_meta_head: Point3D | None
    _grf: ForcePlatform | None
    _vertical_axis: Literal["X", "Y", "Z"]
    _antpos_axis: Literal["X", "Y", "Z"]
    _grf_threshold: float
    _height_threshold: float
    _extra_signals: list[Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform]

    # * constructor

    def __init__(
        self,
        algorithm: Literal["kinematics", "kinetics"],
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
        *extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):

        # set the algorithm
        self.set_algorithm(algorithm)

        # set the thresholds
        self.set_height_threshold(height_threshold)
        self.set_grf_threshold(grf_threshold)

        # set the vertical and antpos axis
        self.set_vertical_axis(vertical_axis)
        self.set_antpos_axis(antpos_axis)

        # set the labels
        self.set_left_heel(left_heel)
        self.set_right_heel(right_heel)
        self.set_left_metatarsal_head(left_meta_head)
        self.set_right_metatarsal_head(right_meta_head)
        self.set_left_toe(left_toe)
        self.set_right_toe(right_toe)
        self.set_ground_reaction_force(grf)
        self._extra_signals = []
        for signal in extra_signals:
            self.add_extra_signal(signal)

    # * attributes

    @property
    def algorithm(self):
        """the selected cycle detection algorithm"""
        return self._algorithm

    @property
    def resultant_force(self):
        """return the ground reaction force data"""
        if self._grf is None:
            return None
        return self._grf.force

    @property
    def centre_of_pressure(self):
        """return the center of pressure data"""
        if self._grf is None:
            return None
        return self._grf.origin

    @property
    def ground_reaction_force(self):
        """return the ground reaction force object"""
        return self._grf

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
    def anteroposterior_axis(self):
        """return the anterior-posterior axis"""
        return self._antpos_axis

    @property
    def lateral_axis(self):
        """return the anterior-posterior axis"""
        used = [self.vertical_axis, self.anteroposterior_axis]
        axes = [Literal['X'], Literal['Y'], Literal['Z']]
        return [i for i in axes if i not in used][0]

    @property
    def grf_threshold(self):
        """return the grf threshold"""
        return self._grf_threshold

    @property
    def height_threshold(self):
        """return the height threshold"""
        return self._height_threshold

    # * methods

    def _check_input(self, obj: Any, target: type | Iterable[type]):
        """check if the provided object is None or a Point3D object and return it"""
        if isinstance(target, Iterable):
            target = tuple(i for i in target)
        if obj is None or isinstance(obj, target):
            return obj
        raise ValueError(f"obj must be a {type} instance or None.")

    def _filter_kinetics(self, grf: np.ndarray, time: np.ndarray):
        """filter the ground reaction force signal"""
        fsamp = float(1 / np.mean(np.diff(time)))
        grff = labsp.fillna(grf.astype(float).flatten(), value=0)
        grff = labsp.butterworth_filt(
            arr=grff.astype(float).flatten(),  # type: ignore
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

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """set the gait cycle detection algorithm"""
        algorithms = ["kinematics", "kinetics"]
        if not isinstance(algorithm, str) or algorithm not in algorithms:
            msg = "'algorithm' must be any between 'kinematics' or 'kinetics'."
            raise ValueError(msg)
        algo = algorithm
        if (
            algo == "kinetics"
            and self.resultant_force is None
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
            and self.resultant_force is not None
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
        elif self.resultant_force is None and any(
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

    def set_left_heel(self, heel: Point3D | None):
        """set the left heel coordinates."""
        self._left_heel = self._check_input(heel, Point3D)

    def set_right_heel(self, heel: Point3D | None):
        """set the right heel coordinates."""
        self._right_heel = self._check_input(heel, Point3D)

    def set_left_metatarsal_head(self, meta: Point3D | None):
        """set the left metatarsal head coordinates."""
        self._left_meta_head = self._check_input(meta, Point3D)

    def set_right_metatarsal_head(self, meta: Point3D | None):
        """set the right metatarsal head coordinates."""
        self._right_meta_head = self._check_input(meta, Point3D)

    def set_left_toe(self, toe: Point3D | None):
        """set the left toe coordinates."""
        self._left_toe = self._check_input(toe, Point3D)

    def set_right_toe(self, toe: Point3D | None):
        """set the right toe coordinates."""
        self._right_toe = self._check_input(toe, Point3D)

    def set_ground_reaction_force(self, grf: ForcePlatform | None):
        """set the ground reaction force data"""
        self._grf = self._check_input(grf, ForcePlatform)

    def add_extra_signal(self, signal: Signal1D | Signal3D | Point3D | EMGSignal | ForcePlatform):
        """add extra signals to the object"""
        types = (Signal1D, Signal3D, Point3D, EMGSignal, ForcePlatform)
        self._extra_signals += [self._check_input(signal, types)]

    def remove_extra_signal(self, name:str):
        """remove the first signal having the provided name"""
        if not isinstance(name, str):
            raise ValueError("'name' must be a str instance.")
        for signal in self._extra_signals:
            if signal.name == name:
                self._extra_signals.remove(signal)

    def pop_extra_signal(self, index:int):
        """remove the extra signal at the given index"""
        if not isinstance(index, int):
            raise ValueError("'index' must be an int instance.")
        return self._extra_signals.pop(index)


class GaitCycle(GaitObject):
    """
    basic gait cycle class.

    Parameters
    ----------
    side: Literal['left', 'right']
        the side of the cycle

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

    left_heel: Point3D | None
        the left heel label

    right_heel: Point3D | None
        the right heel label

    left_toe: Point3D | None
        the left toe label

    right_toe: Point3D | None
        the right toe label

    left_meta_head: Point3D | None
        the left metatarsal head label

    right_meta_head: Point3D | None
        the right metatarsal head label

    grf: ForcePlatform | None
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

    *extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        additional objects not directly used by the analysis

    Note
    ----
    the cycle starts from the toeoff and ends at the next toeoff of the
    same foot
    """

    # * class variables

    _side: Literal["left", "right"]
    _footstrike_s: float
    _midstance_s: float
    _absolute_time_events: list[str] = [
        "footstrike_s",
        "midstance_s",
        "init_s",
        "end_s",
    ]

    # * constructor

    def __init__(
        self,
        side: Literal["left", "right"],
        algorithm: Literal["kinematics", "kinetics"],
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
        *extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
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
            *extra_signals,
        )
        self.set_side(side)

    # * attributes

    @property
    def side(self):
        """return the end time in seconds"""
        return self._side

    @property
    def init_s(self):
        """return the first toeoff time in seconds"""
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.data.index.to_numpy()[0])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.data.index.to_numpy()[0])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

    @property
    def end_s(self):
        """return the toeoff time corresponding to the end of the cycle in seconds"""
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.data.index.to_numpy()[-1])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.data.index.to_numpy()[-1])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

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
                if lbl in self._absolute_time_events:
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
        super().set_algorithm(algorithm)
        self._update_events()

    def set_side(self, side:Literal['right', 'left']):
        """set the cycle side"""
        if not isinstance(side, (Literal, str)):
            raise ValueError("'side' must be 'left' or 'right'.")
        if side not in ['left', 'right']:
            raise ValueError("'side' must be 'left' or 'right'.")
        self._side = side


class GaitTest(GaitObject, LabTest):
    """
    detect steps and strides from kinematic/kinetic data and extract biofeedback
    info

    Parameters
    ----------
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

    left_heel: Point3D | None
        the left heel label

    right_heel: Point3D | None
        the right heel label

    left_toe: Point3D | None
        the left toe label

    right_toe: Point3D | None
        the right toe label

    left_meta_head: Point3D | None
        the left metatarsal head label

    right_meta_head: Point3D | None
        the right metatarsal head label

    grf: ForcePlatform | None
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

    *extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        additional objects not directly used by the analysis
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

    def _make_results_plot(self):
        """
        generate a view with allowing to understand the detected gait cycles
        """
        return NotImplementedError

    def _find_cycles_kinetics(self) -> None:
        """find the gait cycles using the kinetics algorithm"""
        raise NotImplementedError

    def _find_cycles_kinematics(self) -> None:
        """find the gait cycles using the kinematics algorithm"""
        raise NotImplementedError

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """set the gait cycle detection algorithm"""
        super().set_algorithm(algorithm)

        # update cycles
        self._cycles = []
        if self.algorithm == "kinetics":
            self._find_cycles_kinetics()
        elif self.algorithm == "kinematics":
            self._find_cycles_kinematics()

    # * constructors

    def __init__(
        self,
        algorithm: Literal["kinematics", "kinetics"],
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
        *extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
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
            *extra_signals,
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
        if not isinstance(file, str) or not exists(file):
            raise ValueError("'file' must be the path to an existing .tdf file.")
        tdf = read_tdf(file)
        args = dict(algorithm=algorithm,
            grf_threshold=grf_threshold,
            height_threshold=height_threshold,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,)
        points = tdf['CAMERA']['TRACKED']['TRACKS']  # type: ignore
        if points is not None:
            targets = {
                'left_heel': left_heel,
                'right_heel': right_heel,
                'left_toe': left_toe,
                'right_toe': right_toe,
                'left_meta_head': left_meta_head,
                'right_meta_head': right_meta_head,
            }
            lbls = points.columns.get_level_values(0).unique().tolist()
            for target, lbl in targets.items():
                if lbl is not None and lbl in lbls:
                    obj = Point3D(
                        xarr=points[lbl]['X'].values.astype(float).flatten(),
                        yarr=points[lbl]['Y'].values.astype(float).flatten(),
                        zarr=points[lbl]['Z'].values.astype(float).flatten(),
                        time=points.index.to_numpy(),
                        name=lbl,
                        strip=True,
                        reset_index=False,
                    )
                else:
                    obj = None
                args[target] = obj

            targets = [left_heel, right_heel, left_toe, right_toe, left_meta_head, right_meta_head]
            for target in relevant:

                
        obj = cls(
            algorithm=algorithm,
            grf_threshold=grf_threshold,
            height_threshold=height_threshold,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,

        )

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
