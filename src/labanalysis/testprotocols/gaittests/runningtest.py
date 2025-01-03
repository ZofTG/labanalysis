"""kinematics module"""

#! IMPORTS


import warnings
from typing import Literal

import numpy as np
import pandas as pd

from ... import signalprocessing as labsp
from ..frames import StateFrame
from .gait import GRF_THRESHOLD_DEFAULT, HEIGHT_THRESHOLD_DEFAULT, GaitCycle, GaitTest

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
        return self.slice(self._init_s, self._footstrike_s)

    @property
    def contact_frame(self):
        """return a stateframe corresponding to the contact phase"""
        return self.slice(self._footstrike_s, self._end_s)

    @property
    def loading_response_frame(self):
        """return a stateframe corresponding to the loading response phase"""
        return self.slice(self._footstrike_s, self._midstance_s)

    @property
    def propulsion_frame(self):
        """return a stateframe corresponding to the propulsive phase"""
        return self.slice(self._midstance_s, self._end_s)

    @property
    def _flight_time_s(self):
        """return the flight time in seconds"""
        return self._footstrike_s - self._end_s

    @property
    def _loadingresponse_time_s(self):
        """return the loading response time in seconds"""
        return self._midstance_s - self._footstrike_s

    @property
    def _propulsion_time_s(self):
        """return the propulsion time in seconds"""
        return self._end_s - self._midstance_s

    @property
    def _contact_time_s(self):
        """return the contact time in seconds"""
        return self._end_s - self._footstrike_s

    # * methods

    def _footstrike_kinetics(self):
        """find the footstrike time using the kinetics algorithm"""

        # get the contact phase samples
        if self.grf is None:
            raise ValueError("no ground reaction force data available.")
        vgrf = self.grf[self.grf.columns.get_level_values(0)[0]]
        vgrf = vgrf[self.vertical_axis]
        vgrf = labsp.fillna(vgrf).values.astype(float).flatten()  # type: ignore
        time = self.grf.index.to_numpy()
        grff = self._filter_kinetics(vgrf, time)
        mask = grff >= self.grf_threshold
        contacts = labsp.continuous_batches(mask)

        # extract the first contact time
        if len(contacts) == 0:
            raise ValueError("no footstrike has been found.")

        return float(time[contacts[0][0]])

    def _footstrike_kinematics(self):
        """find the footstrike time using the kinematics algorithm"""

        # get the relevant vertical coordinates
        vcoords = {}
        heel_lbl = f"{self.side.lower()}_heel"
        heelv = getattr(self, heel_lbl)
        heelv = heelv[heelv.columns.get_level_values(0)[0]][self.vertical_axis]
        vcoords[heel_lbl] = heelv.values.astype(float).flatten()
        meta_lbl = f"{self.side.lower()}_meta_head"
        meta_dfr = getattr(self, meta_lbl)
        if meta_dfr is not None:
            meta_dfr = meta_dfr[meta_dfr.columns.get_level_values(0)[0]]
            vcoords[meta_lbl] = meta_dfr[self.vertical_axis]
            vcoords[meta_lbl] = vcoords[meta_lbl].values.astype(float).flatten()

        # filter the signals and extract the first contact time
        time = self.markers.index.to_numpy()
        fs_time = []
        for val in vcoords.values():
            mask = self._filter_kinematics(val, time) < self.height_threshold
            contacts = labsp.continuous_batches(mask)
            if len(contacts) > 0:
                fs_time += [time[contacts[0][0]]]

        # get output time
        if len(fs_time) > 0:
            return float(np.min(fs_time))
        raise ValueError("no footstrike has been found.")

    def _midstance_kinetics(self):
        """find the midstance time using the kinetics algorithm"""
        if self.grf is None:
            raise ValueError("no ground reaction force data available.")
        time = self.grf.index.to_numpy()
        vgrf = self.grf[self.grf.columns.get_level_values(0)[0]]
        vgrf = vgrf[self.vertical_axis]
        vgrf = labsp.fillna(vgrf).values.astype(float).flatten()  # type: ignore
        grff = self._filter_kinetics(vgrf, time)
        return float(time[np.argmax(grff)])

    def _midstance_kinematics(self):
        """find the midstance time using the kinematics algorithm"""
        # get the available markers
        lbls = [f"{self.side.lower()}_{i}" for i in ["heel", "toe"]]
        meta_lbl = f"{self.side.lower()}_meta_head"
        meta_dfr = getattr(self, meta_lbl)
        if meta_dfr is not None:
            lbls += [meta_lbl]

        # get the mean vertical signal
        time = self.markers.index.to_numpy()
        ref = np.zeros_like(time)
        for lbl in lbls:
            val = getattr(self, lbl)
            val = val[val.columns.get_level_values(0)[0]][self.vertical_axis]
            val = val.values.astype(float).flatten()
            ref += self._filter_kinematics(val, time)
        ref /= len(lbls)

        # return the time corresponding to the minimum value
        return float(time[np.argmin(val)])

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
        grf_threshold: float | int = GRF_THRESHOLD_DEFAULT,
        height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT,
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
        time = self.markers.index.to_numpy()
        fsamp = float(1 / np.mean(np.diff(time)))
        for side in ["l", "r"]:

            # get the vertical coordinates of the toe markers
            arr = self.markers[f"{side}_toe"][self.vertical_axis]
            arr = arr.values.astype(float).flatten()

            # get the minimum reasonable contact time for each step
            ftoe = self._filter_kinematics(arr, time)
            frq, pwr = labsp.psd(ftoe, fsamp)
            ffrq = frq[np.argmax(pwr)]
            cycle_time = 1 / ffrq
            dsamples = int(cycle_time * fsamp * 0.1)

            # get the contact phases
            mask = ftoe < self.height_threshold
            contacts = labsp.continuous_batches(mask)
            contacts = [i for i in contacts if len(i) >= dsamples]

            # get the toe-offs
            for i in contacts:
                line = pd.Series({"time": time[i[-1]], "side": side})
                tos += [pd.DataFrame(line).T]

        # wrap the events
        if len(tos) == 0:
            raise ValueError("no toe-offs have been found.")
        tos = pd.concat(tos, ignore_index=True).sort_values("time")
        tos.reset_index(inplace=True, drop=True)

        # check the alternation of the steps
        sides = tos.side.values
        if not all(s0 != s1 for s0, s1 in zip(sides[:-1], sides[1:])):
            warnings.warn("Left-Right steps alternation not guaranteed.")
        for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
            t0 = float(tos.time.values[i0])
            t1 = float(tos.time.values[i1])
            args = {
                "frame": self.slice(from_time=t0, to_time=t1),
                "side": "LEFT" if tos.side.values[i1] == "l" else "RIGHT",
                "grf_threshold": self.grf_threshold,
                "height_threshold": self.height_threshold,
                "algorithm": self.algorithm,
                "vertical_axis": self.vertical_axis,
                "antpos_axis": self.antpos_axis,
            }
            if self.left_heel is not None:
                col = self.left_heel.columns.get_level_values(0)[0]
                args["left_heel"] = col
            else:
                args["left_heel"] = None
            if self.right_heel is not None:
                col = self.right_heel.columns.get_level_values(0)[0]
                args["right_heel"] = col
            else:
                args["right_heel"] = None
            if self.left_toe is not None:
                col = self.left_toe.columns.get_level_values(0)[0]
                args["left_toe"] = col
            else:
                args["left_toe"] = None
            if self.right_toe is not None:
                col = self.right_toe.columns.get_level_values(0)[0]
                args["right_toe"] = col
            else:
                args["right_toe"] = None
            if self.left_meta_head is not None:
                col = self.left_meta_head.columns.get_level_values(0)[0]
                args["left_meta_head"] = col
            else:
                args["left_meta_head"] = None
            if self.right_meta_head is not None:
                col = self.right_meta_head.columns.get_level_values(0)[0]
                args["right_meta_head"] = col
            else:
                args["right_meta_head"] = None
            if self.grf is not None:
                args["grf"] = self.grf.columns.get_level_values(0)[0]
            else:
                args["grf"] = None
            self._cycles += [RunningStep(**args)]

    def _find_cycles_kinetics(self):
        """find the gait cycles using the kinetics algorithm"""
        if self.grf is None or self.cop is None:
            raise ValueError("no ground reaction force data available.")

        # get the grf and the latero-lateral COP
        time = self.grf.index.to_numpy()
        axs = [self.vertical_axis, self.antpos_axis]
        axs = [i for i in ["X", "Y", "Z"] if i not in axs]
        mlc = self.cop[self.cop.columns.get_level_values(0)[0]].loc[:, axs]
        mlc = mlc.values.astype(float).flatten()
        grf = self.grf[self.grf.columns.get_level_values(0)[0]]
        grf = grf[self.vertical_axis].values.astype(float).flatten()
        grff = self._filter_kinetics(grf, time)
        mlcf = self._filter_kinetics(mlc, time)

        # check if there are flying phases
        flights = labsp.continuous_batches(grff <= self.grf_threshold)
        if len(flights) > 0 and flights[0][0] == 0:
            flights = flights[1:]

        # in case of positive contacts we have a running test
        if len(flights) == 0:
            raise ValueError("No flight phases have been found on data.")

        # get the toe-offs
        tos = [i[0] for i in flights]

        # get the mean latero-lateral position of each contact
        pos = [np.mean(mlcf[i]) for i in flights]

        # get the mean value of alternated contacts and set the step sides
        # accordingly
        evens = np.mean(pos[0:-1:2])
        odds = np.mean(pos[1:-1:2])
        sides = []
        for i in np.arange(len(pos)):
            if evens > odds:
                sides += ["LEFT" if i % 2 == 0 else "RIGHT"]
            else:
                sides += ["LEFT" if i % 2 != 0 else "RIGHT"]

        for to, ed, side in zip(tos[:-1], tos[1:], sides):
            t0 = float(time[to])
            t1 = float(time[ed])
            args = {
                "frame": self.slice(from_time=t0, to_time=t1),
                "side": side,
                "grf_threshold": self.grf_threshold,
                "height_threshold": self.height_threshold,
                "algorithm": self.algorithm,
                "vertical_axis": self.vertical_axis,
                "antpos_axis": self.antpos_axis,
            }
            if self.left_heel is not None:
                col = self.left_heel.columns.get_level_values(0)[0]
                args["left_heel"] = col
            else:
                args["left_heel"] = None
            if self.right_heel is not None:
                col = self.right_heel.columns.get_level_values(0)[0]
                args["right_heel"] = col
            else:
                args["right_heel"] = None
            if self.left_toe is not None:
                col = self.left_toe.columns.get_level_values(0)[0]
                args["left_toe"] = col
            else:
                args["left_toe"] = None
            if self.right_toe is not None:
                col = self.right_toe.columns.get_level_values(0)[0]
                args["right_toe"] = col
            else:
                args["right_toe"] = None
            if self.left_meta_head is not None:
                col = self.left_meta_head.columns.get_level_values(0)[0]
                args["left_meta_head"] = col
            else:
                args["left_meta_head"] = None
            if self.right_meta_head is not None:
                col = self.right_meta_head.columns.get_level_values(0)[0]
                args["right_meta_head"] = col
            else:
                args["right_meta_head"] = None
            if self.grf is not None:
                args["grf"] = self.grf.columns.get_level_values(0)[0]
            else:
                args["grf"] = None
            self._cycles += [RunningStep(**args)]

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
        grf_threshold: float | int = GRF_THRESHOLD_DEFAULT,
        height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT,
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
