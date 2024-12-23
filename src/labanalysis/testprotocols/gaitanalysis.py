"""kinematics module"""

#! IMPORTS


from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import signalprocessing as labsp
from .base import LabTest
from .frames import StateFrame

#! CONSTANTS

GRF_THRESHOLD_DEFAULT = 100
HEIGHT_THRESHOLD_DEFAULT = 0.01


__all__ = [
    "GRF_THRESHOLD_DEFAULT",
    "HEIGHT_THRESHOLD_DEFAULT",
    "GaitCycle",
    "GaitTest",
    "RunningStep",
    "WalkingStride",
    "RunningTest",
    # "WalkingTest",
]


#! FUNCTIONS


def _filter_grf(grf: np.ndarray, time: np.ndarray):
    """filter the ground reaction force signal"""
    fsamp = float(1 / np.mean(np.diff(time)))
    grff = labsp.fillna(np.atleast_2d(grf).T, None, None)
    grff = np.array([grff]).astype(float).flatten()
    grff = labsp.butterworth_filt(
        arr=grff,
        fcut=[10, 100],
        fsamp=fsamp,
        order=4,
        ftype="bandstop",
        phase_corrected=True,
    )
    return grff.astype(float).flatten()


def _filter_vcoord(coord: np.ndarray, time: np.ndarray):
    """filter vertical coordinates from kinematic data"""
    fsamp = float(1 / np.mean(np.diff(time)))
    fcoord = labsp.fillna(np.atleast_2d(coord).T, None, None)
    fcoord = labsp.butterworth_filt(
        arr=np.array([fcoord - np.min(fcoord)]).astype(float).flatten(),
        fcut=6,
        fsamp=fsamp,
        order=4,
        ftype="lowpass",
        phase_corrected=True,
    )
    return fcoord.astype(float).flatten()


#! CLASSESS


class GaitCycle(StateFrame):
    """basic gait class to be properly implemented"""

    # * class variables

    _side: Literal["LEFT", "RIGHT"]

    # * attributes

    @property
    def init_s(self):
        """return the first toeoff time in seconds"""
        return float(self.to_dataframe().index.to_list()[0])

    @property
    def end_s(self):
        """return the toeoff time corresponding to the end of the cycle in seconds"""
        return float(self.to_dataframe().index.to_list()[-1])

    @property
    def side(self):
        """return the end time in seconds"""
        return self._side

    @property
    def time_events(self):
        """return all the time events defining the cycle"""
        evts: dict[str, float | str] = {}
        for lbl in dir(self):
            if lbl.endswith("_s") or lbl == "side":
                evts[lbl] = getattr(self, lbl)
        return evts

    # * constructor

    def __init__(
        self,
        frame: StateFrame,
        side: Literal["LEFT", "RIGHT"],
    ):

        if not isinstance(frame, StateFrame):
            raise ValueError("'frame' must be a StateFrame instance.")

        super().__init__(
            markers_raw=frame.markers,
            forceplatforms_raw=frame.forceplatforms,
            emgs_raw=frame.emgs,
        )
        self._processed = frame.is_processed()
        self._marker_processing_options = frame.marker_processing_options
        self._forceplatform_processing_options = frame.forceplatform_processing_options
        self._emg_processing_options = frame.emg_processing_options

        # check the side
        if not isinstance(side, str):
            raise ValueError("'side' must be 'LEFT' or 'RIGHT'")
        if side in ["LEFT", "RIGHT"]:
            self._side = side


class RunningStep(GaitCycle):
    """
    basic running step class.

    Parameters
    ----------
    frame: StateFrame
        a stateframe object containing all the available kinematic, kinetic
        and emg data related to the step

    side: Literal['LEFT', 'RIGHT']
        the side of the step

    grf_threshold: float | int = GRF_THRESHOLD_DEFAULT
        the minimum ground reaction force value (in N) to evaluate the impact
        of one foot on the ground.

    height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT
        the maximum vertical height of one marker (in cm) from the ground
        to be assumed in contact with it.

    Note
    ----
    the step starts from the toeoff and ends at the next toeoff of the
    contralateral foot.
    """

    # * class variables

    _grf_threshold: float
    _height_threshold: float

    # * properties

    @property
    def grf_threshold(self):
        """return the stored grf threshold in N"""
        return self._grf_threshold

    @property
    def height_threshold(self):
        """return the stored height threshold in cm"""
        return self._height_threshold

    @property
    def flight_phase(self):
        """return a stateframe corresponding to the flight phase"""
        return self.slice(self.init_s, self.footstrike_s)

    @property
    def contact_phase(self):
        """return a stateframe corresponding to the contact phase"""
        return self.slice(self.footstrike_s, self.toeoff_s)

    @property
    def loading_response_phase(self):
        """return a stateframe corresponding to the loading response phase"""
        return self.slice(self.footstrike_s, self.midstance_s)

    @property
    def propulsion_phase(self):
        """return a stateframe corresponding to the propulsive phase"""
        return self.slice(self.midstance_s, self.end_s)

    @property
    def step_time_s(self):
        """return the step time in seconds"""
        return self.end_s - self.init_s

    @property
    def cadence_spm(self):
        """return the cadence of the stride in strides per minute"""
        return 60 / self.step_time_s

    @property
    def flight_time_s(self):
        """return the flight time in seconds"""
        return self.footstrike_s - self.end_s

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

    @property
    def toeoff_s(self):
        """return the step toeoff in seconds"""
        return self.end_s

    @property
    def footstrike_s(self):
        """return the foot-strike time in seconds"""

        # if grf exists
        if self.forceplatforms.shape[1] > 0:
            grf = self.forceplatforms.fRes.FORCE[self.vertical_axis]
            grf = labsp.fillna(grf).values.astype(float).flatten()  # type: ignore
            time = self.forceplatforms.index.to_numpy()
            grff = _filter_grf(grf, time)
            mask = grff >= self.grf_threshold
        else:

            # otherwise look at the markers
            time = self.markers.index.to_numpy()
            side = self.side[0].lower()
            heel = self.markers[f"{side}Heel"][self.vertical_axis]
            heel = heel.values.astype(float).flatten()
            fheel = _filter_vcoord(heel, time)
            mask = fheel < self.height_threshold
            marker_labels = self.markers.columns.get_level_values(0).unique()
            if f"{side}Mid" in marker_labels:
                mid = self.markers[f"{side}Mid"][self.vertical_axis]
                mid = mid.values.astype(float).flatten()
                fmid = _filter_vcoord(mid, time)
                mask |= fmid < self.height_threshold

        # get the contact phases and extract the first contact time
        contacts = labsp.continuous_batches(mask)
        if len(contacts) == 0:
            raise ValueError("no footstrike has been found.")

        return float(time[contacts[0][0]])

    @property
    def midstance_s(self):
        """return the mid-stance time in seconds"""
        # if grf exists
        if self.forceplatforms.shape[1] > 0:
            time = self.forceplatforms.index.to_numpy()
            grf = self.forceplatforms.fRes.FORCE[self.vertical_axis]
            grf = grf.values.astype(float).flatten()
            grff = _filter_grf(grf, time)
            return float(time[np.argmax(grff)])

        # otherwise look at the markers
        time = self.markers.index.to_numpy()
        marker_labels = self.markers.columns.get_level_values(0).unique()
        side = self.side[0].lower()
        lbls = [f"{side}Heel", f"{side}Toe"]
        if f"{side}Mid" in marker_labels:
            lbls += [f"{side}Mid"]
        ref = np.zeros_like(time)
        for lbl in lbls:
            val = self.markers[lbl][self.vertical_axis]
            val = val.values.astype(float).flatten()
            ref += _filter_vcoord(val, time)
        val /= len(lbls)
        return float(time[np.argmin(val)])

    # * constructor

    def __init__(
        self,
        frame: StateFrame,
        side: Literal["LEFT", "RIGHT"],
        grf_threshold: float | int = GRF_THRESHOLD_DEFAULT,
        height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT,
    ):
        super().__init__(frame=frame, side=side)
        if not isinstance(grf_threshold, (int, float)):
            raise ValueError("'grf_threshold' must be a float or int")
        self._grf_threshold = float(grf_threshold)
        if not isinstance(height_threshold, (int, float)):
            raise ValueError("'height_threshold' must be a float or int")
        self._height_threshold = float(height_threshold)


class WalkingStride(GaitCycle):
    """
    basic walking stride class.

    Parameters
    ----------
    ipsilateral_toeoff_s: int | float
        the toe-off time in seconds of the contralateral foot defining the start
        of the step

    ipsilateral_footstrike_s: int | float
        the foot-strike time in seconds of the foot setting the start of the
        stance phase

    contralateral_toeoff_s: int | float
        the toeoff of the contralateral foot. This sets the end of the
        double-support phase

    ending_s: int | float
        the toeoff time in seconds of the ipsilateral foot setting the end
        of the step

    side: Literal['RIGHT', 'LEFT']
        the side of the test

    Attributes
    ----------
    stride_time_s: float
        the stride time in seconds

    stance_time_s: float
        the stance time in seconds

    swing_time_s: float
        the swing time in seconds

    single_support_time_s: float
        the single support time in seconds

    double_support_time_s: float
        the double support time in seconds

    step_time_s: float
        the time lapse between one toe-off and the one of the contralateral
        step in seconds.

    Note
    ----
    the stride starts from the toe-off of the ipsilateral foot.
    """

    _ipsilateral_footstrike_s: float
    _ipsilateral_toeoff_s: float
    _contralateral_footstrike_s: float
    _contralateral_toeoff_s: float
    _ending_s: float
    _side: Literal["RIGHT", "LEFT"] | None

    # * properties

    @property
    def ipsilateral_footstrike_s(self):
        """
        return the foot-strike time of the foot performing the actual stride
        in seconds
        """
        return self._ipsilateral_footstrike_s

    @property
    def contralateral_footstrike_s(self):
        """
        return the foot-strike time of the foot not performing the actual stride
        in seconds
        """
        return self._contralateral_footstrike_s

    @property
    def ipsilateral_toeoff_s(self):
        """
        return the toe-off time of the foot performing the actual stride
        in seconds
        """
        return self._ipsilateral_toeoff_s

    @property
    def contralateral_toeoff_s(self):
        """
        return the toe-off time of the foot not performing the actual stride
        in seconds
        """
        return self._contralateral_toeoff_s

    @property
    def ending_s(self):
        """
        return the ending toe-off (i.e. the second ipsilateral toeoff time)
        in seconds
        """
        return self._ending_s

    @property
    def side(self):
        """return the side of the stride"""
        return self._side

    @property
    def stride_time_s(self):
        """return the stride time in seconds"""
        return self.ending_s - self.ipsilateral_toeoff_s

    @property
    def step_time_s(self):
        """return the step time in seconds"""
        return self.contralateral_toeoff_s - self.ipsilateral_toeoff_s

    @property
    def stance_time_s(self):
        """return the stance time in seconds"""
        return self.ending_s - self.ipsilateral_footstrike_s

    @property
    def swing_time_s(self):
        """return the swing time in seconds"""
        return self.ipsilateral_footstrike_s - self.ipsilateral_toeoff_s

    @property
    def single_support_time_s(self):
        """return the single support time in seconds"""
        return self.contralateral_toeoff_s - self.contralateral_footstrike_s

    @property
    def double_support_time_s(self):
        """return the double support time in seconds"""
        return self.stride_time_s - self.single_support_time_s

    # * constructor

    def __init__(
        self,
        ipsilateral_toeoff_s: int | float,
        ipsilateral_footstrike_s: int | float,
        contralateral_toeoff_s: int | float,
        contralateral_footstrike_s: int | float,
        ending_s: int | float,
        side: Literal["RIGHT", "LEFT"] | None = None,
    ):
        super().__init__()
        self._ipsilateral_footstrike_s = ipsilateral_footstrike_s
        self._ipsilateral_toeoff_s = ipsilateral_toeoff_s
        self._contralateral_footstrike_s = contralateral_footstrike_s
        self._contralateral_toeoff_s = contralateral_toeoff_s
        self._ending_s = ending_s
        self._side = side


class GaitTest(StateFrame, LabTest):
    """
    detect steps and strides from kinematic/kinetic data and extract biofeedback
    info

    Parameters
    ----------
    frame: StateFrame
        a stateframe object containing all the available kinematic, kinetic
        and emg data related to the test

    grf_threshold: float | int = GRF_THRESHOLD_DEFAULT
        the minimum ground reaction force value (in N) to evaluate the impact
        of one foot on the ground.

    height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT
        the maximum vertical height of one marker (in cm) from the ground
        to be assumed in contact with it.
    """

    # * class variables

    _cycles: list[RunningStep | WalkingStride]
    _grf_threshold: float
    _height_threshold: float

    # * attributes

    @property
    def cycles(self):
        """the detected gait cycles"""
        return self._cycles

    @property
    def grf_threshold(self):
        """return the stored grf threshold in N"""
        return self._grf_threshold

    @property
    def height_threshold(self):
        """return the stored height threshold in cm"""
        return self._height_threshold

    # * methods

    def _make_summary_table(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ):
        # TODO
        raise NotImplementedError

    def _make_summary_plot(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ):
        # TODO
        raise NotImplementedError

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
        raise NotImplementedError

    def _find_cycles(self):
        """find steps via grf coordinates"""
        return NotImplementedError

    # * constructors

    def __init__(
        self,
        frame: StateFrame,
        grf_threshold: float | int = GRF_THRESHOLD_DEFAULT,
        height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT,
    ):

        if not isinstance(frame, StateFrame):
            raise ValueError("'frame' must be a StateFrame instance.")
        super().__init__(
            markers_raw=frame.markers,
            forceplatforms_raw=frame.forceplatforms,
            emgs_raw=frame.emgs,
        )
        self._processed = frame.is_processed()
        self._marker_processing_options = frame.marker_processing_options
        self._forceplatform_processing_options = frame.forceplatform_processing_options
        self._emg_processing_options = frame.emg_processing_options

        if not isinstance(grf_threshold, (int, float)):
            raise ValueError("'grf_threshold' must be a float or int")
        self._grf_threshold = float(grf_threshold)

        if not isinstance(height_threshold, (int, float)):
            raise ValueError("'height_threshold' must be a float or int")
        self._height_threshold = float(height_threshold)

        # check the input data
        if self.markers.shape[1] > 0:
            markers = self.markers.columns.get_level_values(0).tolist()
            required = ["lHeel", "lToe", "rHeel", "rToe"]
            for i in required:
                if i not in markers:
                    raise ValueError(f"markers_raw must include {i}.")
            optional = ["lMid", "rMid"]
            cols = required + [i for i in optional if i in markers]
            self._markers = self.markers[cols].copy()
        if self.forceplatforms.shape[1] > 0:
            fps = self.forceplatforms.columns.get_level_values(0).tolist()
            if "fRes" not in fps:
                raise ValueError("'forceplatforms_raw' must include 'fRes'.")
            self._forceplatforms = self.forceplatforms[["fRes"]].copy()

        # find steps and strides
        self._cycles = []
        self._find_cycles()

    @classmethod
    def from_tdf_file(
        cls,
        file: str,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
        grf_threshold: float | int = GRF_THRESHOLD_DEFAULT,
        height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT,
    ):
        """
        Generate a GaitTest object directly from a .tdf file.

        Parameters
        ----------
        file: str
            the path to a ".tdf" file.

        vertical_axis: Literal['X', 'Y', 'Z'] = 'Y', optional
            the axis defining the vertical direction.

        antpos_axis: Literal['X', 'Y', 'Z'] = 'Z', optional
            the axis defining the anterior-posterior direction.

        grf_threshold: float | int = GRF_THRESHOLD_DEFAULT
            the minimum ground reaction force value (in N) to evaluate the impact
            of one foot on the ground.

        height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT
            the maximum vertical height of one marker (in cm) from the ground
            to be assumed in contact with it.
        """
        frame = StateFrame.from_tdf_file(
            file=file,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,
        )
        frame.process(inplace=True)
        return cls(
            frame=frame,
            grf_threshold=grf_threshold,
            height_threshold=height_threshold,
        )


class RunningTest(GaitTest):
    """
    generate a RunningTest instance

    Parameters
    ----------
    frame: StateFrame
        a stateframe object containing all the available kinematic, kinetic
        and emg data related to the test

    grf_threshold: float | int = GRF_THRESHOLD_DEFAULT
        the minimum ground reaction force value (in N) to evaluate the impact
        of one foot on the ground.

    height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT
        the maximum vertical height of one marker (in cm) from the ground
        to be assumed in contact with it.
    """

    # * class variables

    _cycles: list[RunningStep]

    # * methods

    def _find_cycles(self):
        """find steps via grf coordinates"""
        if self.forceplatforms.shape[1] > 0:

            # get the grf and the latero-lateral COP
            time = self.forceplatforms.index.to_numpy()
            grf = self.forceplatforms.fRes.FORCE[self.vertical_axis]
            grf = grf.values.astype(float).flatten()
            axs = [self.vertical_axis, self.antpos_axis]
            axs = [i for i in ["X", "Y", "Z"] if i not in axs]
            mlc = self.forceplatforms.fRes.ORIGIN[axs]
            mlc = mlc.values.astype(float).flatten()
            grff = _filter_grf(grf, time)

            # check if there are flying phases
            contacts = labsp.continuous_batches(grff < self.grf_threshold)
            if len(contacts) > 0 and contacts[0][0] == 0:
                contacts = contacts[1:]
            if len(contacts) > 0 and contacts[-1][-1] == len(time) - 1:
                contacts = contacts[:-1]

            # in case of positive contacts we have a running test
            if len(contacts) < 1:
                raise ValueError("No flight phases have been found on data.")

            # get the toe-offs
            tos = [i[0] for i in contacts]

            # get the mean latero-lateral position of each contact
            pos = [np.mean(mlc[i]) for i in contacts]

            # get the mean value of alternated contacts
            evens = np.mean(pos[0:-1:2])
            odds = np.mean(pos[1:-1:2])
            sides = []
            for i in np.arange(len(pos)):
                if evens > odds:
                    sides += ["LEFT" if i % 2 == 0 else "RIGHT"]
                else:
                    sides += ["LEFT" if i % 2 != 0 else "RIGHT"]

            for to, ed, side in zip(tos[:-1], tos[1:], sides):
                step = RunningStep(
                    frame=self.slice(float(time[to]), float(time[ed])),
                    side=side,
                    grf_threshold=self.grf_threshold,
                    height_threshold=self.height_threshold,
                )
                self._cycles += [step]

        elif self.markers.shape[1] > 0:

            # get the vertical coordinates of all relevant markers
            vcoords = self.markers.columns
            vcoords = [i for i in vcoords if i[1] == self.vertical_axis]
            vcoords = self.markers[vcoords].copy()
            vcoords.columns = pd.Index([i[0] for i in vcoords.columns])
            vlc = vcoords[[i for i in vcoords.columns if i[0] == "l"]]
            vrc = vcoords[[i for i in vcoords.columns if i[0] == "r"]]

            # get the height thresholds
            time = vcoords.index.to_numpy()
            fsamp = float(1 / np.mean(np.diff(time)))
            tos = []
            for side, dfr in zip(["l", "r"], [vlc, vrc]):

                # get the local minima and maxima in the mean signal
                toe = dfr[f"{side}Toe"].values.astype(float).flatten()
                ftoe = _filter_vcoord(toe, time)

                # get the expected minimmum contact phase length
                frq, pwr = labsp.psd(ftoe, fsamp)
                ffrq = frq[np.argmax(pwr)]
                cycle_time = 1 / ffrq
                dsamples = int(cycle_time * fsamp * 0.1)

                # get the contacts
                contacts = labsp.continuous_batches(ftoe < self.height_threshold)
                contacts = [i for i in contacts if len(i) >= dsamples]

                # get the toe-offs
                for i in contacts:
                    line = pd.Series({"time": time[i[-1]], "side": side})
                    tos += [pd.DataFrame(line).T]

            # wrap the events
            tos = pd.concat(tos, ignore_index=True).sort_values("time")
            tos.reset_index(inplace=True, drop=True)
            for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
                t0 = float(tos.time.values[i0])
                t1 = float(tos.time.values[i1])
                step = RunningStep(
                    frame=self.slice(from_time=t0, to_time=t1),
                    side="LEFT" if tos.side.values[i1] == "l" else "RIGHT",
                    grf_threshold=self.grf_threshold,
                    height_threshold=self.height_threshold,
                )
                self._cycles += [step]

    def _make_results_plot(self):
        """
        generate a view with allowing to understand the detected gait cycles
        """
        # generate the output figure
        rows = []
        if self.forceplatforms.shape[1] > 0:
            rows = ["N", "mm"]
        if self.markers.shape[1] > 0:
            rows += ["mm", "mm"]
            markers = self.markers.columns.get_level_values(0).unique()
            if "lMid" in markers or "rMid" in markers:
                rows += ["mm"]
        fig = make_subplots(
            rows=len(rows),
            cols=1,
            shared_xaxes=True,
            shared_yaxes=False,
            x_title="Time (s)",
            row_titles=rows,
        )
        fig.update_layout(
            template="simple_white",
            title="RunningTest",
            height=300 * len(rows),
        )
        fig.update_xaxes(showticklabels=True)
        row = 1

        # plot force and cop data
        if self.forceplatforms.shape[1] > 0:
            time = self.forceplatforms.index.to_numpy()
            fres = self.forceplatforms.fRes
            grf = fres.FORCE[self.vertical_axis].values.astype(float).flatten()
            axes = [self.vertical_axis, self.antpos_axis]
            lateral_axis = [i for i in ["X", "Y", "Z"] if i not in axes][0]
            cop = fres.ORIGIN[lateral_axis].values.astype(float).flatten()
            cop *= 1000
            idx = np.where(grf < self.grf_threshold)[0]
            cop[idx] = np.nan
            fig.add_trace(
                row=1,
                col=1,
                trace=go.Scatter(
                    x=time,
                    y=grf,
                    name="Raw GRF",
                    opacity=0.5,
                    mode="lines",
                    line_color="navy",
                    legendgroup="Raw GRF",
                ),
            )
            fig.add_trace(
                row=1,
                col=1,
                trace=go.Scatter(
                    x=time,
                    y=_filter_grf(grf, time),
                    name="Filtered GRF",
                    opacity=0.5,
                    mode="lines",
                    line_color="red",
                    legendgroup="Filtered GRF",
                ),
            )
            fig.add_trace(
                row=2,
                col=1,
                trace=go.Scatter(
                    x=time,
                    y=cop,
                    name="Lateral COP",
                    opacity=0.5,
                    mode="lines",
                    line_color="navy",
                    legendgroup="Lateral COP",
                ),
            )

            # plot the thresholds
            fig.add_trace(
                row=1,
                col=1,
                trace=go.Scatter(
                    x=time[[0, -1]],
                    y=[self.grf_threshold, self.grf_threshold],
                    line_dash="dashdot",
                    line_color="black",
                    opacity=0.3,
                    mode="lines",
                    name="GRF Threshold",
                    legendgroup="GRF Threshold",
                ),
            )

            # update the layout and the row number
            row = 3
            fig.update_yaxes(row=1, title="GRF")
            fig.update_yaxes(row=2, title="Lateral CoP")

        # plot vertical marker coordinates
        if self.markers.shape[1] > 0:
            time = self.markers.index.to_numpy()
            labels = ["Heel", "Toe"]
            markers = self.markers.columns.get_level_values(0).unique()
            if "lMid" in markers or "rMid" in markers:
                labels += ["Mid"]
            sides = ["l", "r"]
            for l, label in enumerate(labels):
                for side in sides:
                    color = "navy" if side == "l" else "red"
                    lbl = side + label
                    val = self.markers[lbl][self.vertical_axis].values
                    val = val.astype(float).flatten() * 1000
                    val -= np.min(val)
                    fig.add_trace(
                        row=row,
                        col=1,
                        trace=go.Scatter(
                            x=time,
                            y=val,
                            mode="lines",
                            name=lbl,
                            line_color=color,
                            legendgroup=lbl,
                            opacity=0.5,
                        ),
                    )

                # plot the thresholds
                thresh = self.height_threshold * 1000
                fig.add_trace(
                    row=row,
                    col=1,
                    trace=go.Scatter(
                        x=time[[0, -1]],
                        y=[thresh, thresh],
                        line_dash="dashdot",
                        line_color="brown",
                        opacity=0.3,
                        mode="lines",
                        name="Height Threshold",
                        legendgroup="Height Threshold",
                        showlegend=bool(l == 0),
                    ),
                )

                # update the layout and row counter
                fig.update_yaxes(row=row, title=label)
                row += 1

        # add the annotations of the events and thresholds on each row
        time = self.to_dataframe().index.to_numpy()[[0, -1]]
        for row in np.arange(len(rows)):

            # plot the cycles
            axis = "y" + ("" if row == 0 else str(row + 1))
            traces = [i for i in fig.data if i.yaxis == axis]  # type: ignore
            minv = np.min([np.nanmin(i.y) for i in traces])  # type: ignore
            maxv = np.max([np.nanmax(i.y) for i in traces])  # type: ignore
            range = [minv, maxv]
            for i, cycle in enumerate(self.cycles):
                color = "orange" if cycle.side == "LEFT" else "green"
                fig.add_trace(
                    row=row + 1,
                    col=1,
                    trace=go.Scatter(
                        x=[cycle.init_s, cycle.init_s],
                        y=range,
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
                        x=[cycle.end_s, cycle.end_s],
                        y=range,
                        line_dash="solid",
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
                        x=[cycle.footstrike_s, cycle.footstrike_s],  # type: ignore
                        y=range,
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
                        x=[cycle.midstance_s, cycle.midstance_s],  # type: ignore
                        y=range,
                        mode="lines",
                        line_dash="dot",
                        line_color=color,
                        opacity=0.3,
                        name=f"midstance ({cycle.side})",
                        showlegend=bool((row == 0) & (i < 2)),
                        legendgroup=f"midstance ({cycle.side})",
                    ),
                )

        return fig

    # * constructor

    def __init__(
        self,
        frame: StateFrame,
        grf_threshold: float | int = GRF_THRESHOLD_DEFAULT,
        height_threshold: float | int = HEIGHT_THRESHOLD_DEFAULT,
    ):
        super().__init__(
            frame=frame,
            grf_threshold=grf_threshold,
            height_threshold=height_threshold,
        )
