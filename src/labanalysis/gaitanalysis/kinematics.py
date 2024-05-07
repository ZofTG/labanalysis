"""kinematics module"""

#! IMPORTS


from os.path import exists

import numpy as np
import pandas as pd

from ..signalprocessing import continuous_batches, fillna
from .events import RunningStep, RunningStride, WalkingStep, WalkingStride
from labio import read_tdf


#! CONSTANTS


__all__ = ["GaitTest"]


#! CLASSES


class GaitTest:
    """
    detect steps and strides from kinematic data and extract biofeedback
    info

    Parameters
    ----------
    lheel: pd.DataFrame,
        a dataframe with the time instant of each sample as index and
        MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting
        the left heel coordinates in meters where:
            X = latero-lateral axis (right direction is positive)
            Y = anterior-posterio axis (forward direction is positive)
            Z = vertical axis (upward direction is positive)

    ltoe: pd.DataFrame,
        a dataframe with the time instant of each sample as index and
        MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting
        the left toe coordinates in meters where:
            X = latero-lateral axis (right direction is positive)
            Y = anterior-posterio axis (forward direction is positive)
            Z = vertical axis (upward direction is positive)

    rheel: pd.DataFrame,
        a dataframe with the time instant of each sample as index and
        MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting
        the right heel coordinates in meters where:
            X = latero-lateral axis (right direction is positive)
            Y = anterior-posterio axis (forward direction is positive)
            Z = vertical axis (upward direction is positive)

    rtoe: pd.DataFrame,
        a dataframe with the time instant of each sample as index and
        MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting
        the right toe coordinates in meters where:
            X = latero-lateral axis (right direction is positive)
            Y = anterior-posterio axis (forward direction is positive)
            Z = vertical axis (upward direction is positive)

    lmidfoot: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of the fifth metatarsal head of the left foot,
        where:
            X = latero-lateral axis (right direction is positive)
            Y = anterior-posterio axis (forward direction is positive)
            Z = vertical axis (upward direction is positive)

    rmidfoot: pd.DataFrame | None = None,
        if provided, a dataframe with the time instant of each sample as index
        and MultiIndex columns ([(X, m), (Y, m), (Z, m)]) denoting the
        coordinates in meters of fifth metatarsal head of the right foot, where:
            X = latero-lateral axis (right direction is positive)
            Y = anterior-posterio axis (forward direction is positive)
            Z = vertical axis (upward direction is positive)

    preprocess: bool = True,
        if True, the provided data is assumed to be "raw" and the following
        processing is performed:
            - gap filling via linear regression of the known coordinates
            - lowpass filtering via a Butterworth, 6th order, phase-corrected
                filter with cut-off frequency of 12 Hz.

    height_thresh: float | int = 0.02,
        the minimum threshold in meters to be considered for assuming that
        one foot is in contact to the ground.
    """

    _raw_coordinates: pd.DataFrame | None
    _preprocessed: bool
    _coordinates: pd.DataFrame
    _height_threshold: int | float
    _source_file: str | None
    _steps: list[RunningStep | WalkingStep]

    @property
    def raw_coordinates(self):
        """the input data"""
        return self._raw_coordinates

    @property
    def coordinates(self):
        """has the data used for steps and strides detection"""
        return self._coordinates

    @property
    def preprocessed(self):
        """
        has the raw data being preprocessed before fininding
        steps and strides?
        """
        return self._preprocessed

    @property
    def height_threshold(self):
        """the minimum threshold in meters to be considered for assuming that
        one foot is in contact to the ground."""
        return self._height_threshold

    @property
    def source_file(self):
        "return the tdf file containing the data of the test"
        return self._source_file

    @property
    def steps(self):
        """the detected steps"""
        return self._steps

    @property
    def strides(self):
        """the detected strides"""
        strides = []
        for st1, st2 in zip(self._steps[:-1], self._steps[1:]):
            if (
                st1.landing_s == st2.foot_strike_s
                and st1.side is not None
                and st2.side is not None
                and st1.side != st2.side
            ):
                if isinstance(st1, RunningStep) and isinstance(st2, RunningStep):
                    strides += [RunningStride(st1, st2)]
                elif isinstance(st1, WalkingStep) and isinstance(st2, WalkingStep):
                    strides += [WalkingStride(st1, st2)]

        return strides

    def __init__(
        self,
        lheel: pd.DataFrame,
        ltoe: pd.DataFrame,
        rheel: pd.DataFrame,
        rtoe: pd.DataFrame,
        lmidfoot: pd.DataFrame | None = None,
        rmidfoot: pd.DataFrame | None = None,
        preprocess: bool = True,
        height_thresh: float | int = 0.02,
    ):

        # check the entries
        vlc = {"lHeel": lheel.copy(), "lToe": ltoe.copy()}
        if lmidfoot is not None:
            vlc["lMidfoot"] = lmidfoot.copy()
        vrc = {"rHeel": rheel.copy(), "rToe": rtoe.copy()}
        if rmidfoot is not None:
            vrc["rMidfoot"] = rmidfoot.copy()
        coords = {**vlc, **vrc}
        time = coords["lHeel"].index.to_numpy()
        for lbl, coord in coords.items():

            # check the presence of X, Y and Z columns
            msg = f"{lbl} must be a pandas DataFrame with ['X', 'Y', 'Z'] columns."
            if not isinstance(coord, pd.DataFrame):
                raise TypeError(msg)
            coord.columns = pd.Index(coord.columns.get_level_values(0))
            if not all(i in coord.columns.tolist() for i in ["X", "Y", "Z"]):
                raise ValueError(msg)

            # check the index
            if not coord.shape[0] == len(time):
                raise ValueError(f"{lbl} must have shape [{len(time)}, 3]")
            tarr = coord.index.to_numpy()
            if np.sum(tarr - time) != 0:
                msg = "time index is not consistent between the input dataframes."
                raise ValueError(msg)

        # check preprocess
        if not isinstance(preprocess, bool):
            raise TypeError("preprocess must be True or False")
        self._preprocessed = preprocess

        # check height_thresh
        if not isinstance(height_thresh, (int, float)):
            raise ValueError("height_thresh must be an int or float.")
        self._height_threshold = height_thresh

        # wrap
        for lbl, coord in coords.items():
            coord.columns = pd.MultiIndex.from_product([[lbl], coord.columns])
        self._raw_coordinates = pd.concat(list(coords.values()), axis=1)

        # preprocess (if required)
        if preprocess:
            time = self._raw_coordinates.index.to_numpy()
            fsamp = 1 / np.mean(np.diff(time))

            # fill missing values
            self._coordinates = pd.DataFrame(
                fillna(
                    arr=self._raw_coordinates,
                    n_regressors=6,
                )
            )

            # smooth all marker coordinates
            self._coordinates = self._coordinates.map(
                laban.butterworth_filt,  # type: ignore
                fcut=12,
                fsamp=fsamp,
                order=6,
                ftype="lowpass",
                phase_corrected=True,
                raw=True,
            )
        else:
            self._coordinates = self._raw_coordinates.copy()

        # find steps and strides
        self._find_steps()

    def _find_steps(self):

        # get the vertical coordinates of all relevant markers
        vcoords = self._coordinates[
            [i for i in self._coordinates.columns if i[1] == "Z"]
        ].copy()
        vcoords.columns = pd.Index([i[0] for i in vcoords.columns])
        vlc = vcoords[[i for i in vcoords.columns if i[0] == "l"]]
        vlc.columns = pd.Index([i[1:] for i in vlc.columns])
        vrc = vcoords[[i for i in vcoords.columns if i[0] == "r"]]
        vrc.columns = pd.Index([i[1:] for i in vrc.columns])

        # get the instants where heels and toes are on ground
        vlc -= vlc.min(axis=0)
        vrc -= vrc.min(axis=0)

        # get the mean values (they are used for mid-stance detection)
        mlc = vlc.mean(axis=1)
        mrc = vrc.mean(axis=1)

        # get the batches of time with part of the feet on ground
        glc = vlc < self._height_threshold
        grc = vrc < self._height_threshold
        blc = continuous_batches(glc.any(axis=1).values.astype(bool))
        brc = continuous_batches(grc.any(axis=1).values.astype(bool))

        # exclude those batches that start at the beginning of the data
        # acquisition or are continuning at the end of the data acquisition
        if len(blc) > 0:
            if blc[0][0] == 0:
                blc = blc[1:]
            if blc[-1][-1] >= len(vlc) - 1:
                blc = blc[:-1]
        if len(brc) > 0:
            if brc[0][0] == 0:
                brc = brc[1:]
            if brc[-1][-1] >= len(vrc) - 1:
                brc = brc[:-1]

        # get the events
        time = self._coordinates.index.to_numpy()
        evts_map = {
            "FS LEFT": [time[i[0]] for i in blc],
            "FS RIGHT": [time[i[0]] for i in brc],
            "MS LEFT": [time[np.argmin(mlc.iloc[i]) + i[0]] for i in blc],
            "MS RIGHT": [time[np.argmin(mrc.iloc[i]) + i[0]] for i in brc],
            "TO LEFT": [time[i[-1]] for i in blc],
            "TO RIGHT": [time[i[-1]] for i in brc],
        }
        evts_map = {i: np.array(j) for i, j in evts_map.items()}
        evts_val = np.concatenate(list(evts_map.values()))
        evts_lbl = [np.tile(i, len(v)) for i, v in evts_map.items()]
        evts_lbl = np.concatenate(evts_lbl)
        evts_idx = np.argsort(evts_val)
        evts_val = evts_val[evts_idx]
        evts_side = np.array([i.split(" ")[1] for i in evts_lbl[evts_idx]])
        evts_lbl = np.array([i.split(" ")[0] for i in evts_lbl[evts_idx]])

        # get the steps
        self._steps = []
        run_seq = ["FS", "MS", "TO", "LD"]
        walk_seq = ["FS", "TO", "MS", "LD"]
        for n in np.arange(0, len(evts_lbl) - 4, 3):
            idx = np.arange(4) + n
            seq = evts_lbl[idx].copy()
            seq[-1] = "LD"
            sides = evts_side[idx].copy()
            vals = evts_val[idx].copy()
            s0 = sides[0]
            if (
                all([i == v for i, v in zip(seq, run_seq)])
                & all(i == s0 for i in sides[:-1])
                & (sides[-1] != s0)
            ):  # running
                self._steps += [RunningStep(*vals, side=s0.upper())]
            elif (
                all([i == v for i, v in zip(seq, walk_seq)])
                & all(i == s0 for i in sides[2:-1])
                & (sides[1] != s0)
                & (sides[-1] != s0)
            ):  # walking
                self._steps += [WalkingStep(*vals, side=s0.upper())]

    @classmethod
    def from_file(
        cls,
        file: str,
        rheel_label: str = "rHeel",
        lheel_label: str = "lHeel",
        rtoe_label: str = "rToe",
        ltoe_label: str = "lToe",
        lmid_label: str | None = None,
        rmid_label: str | None = None,
        height_thresh: float | int = 0.02,
    ):
        """
        Generate a GaitTest object directly from a .tdf file.

        Parameters
        ----------
        file: str
            the path to a ".tdf" file.

        lheel_label: str (optional, default = "lHeel"),
            the label of the marker defining the left heel in the tdf file.

        rheel_label: str (optional, default = "lHeel"),
            the label of the marker defining the right heel in the tdf file.

        ltoe_label: str (optional, default = "lToe"),
            the label of the marker defining the left toe in the tdf file.

        rtoe_label: str (optional, default = "lToe"),
            the label of the marker defining the right toe in the tdf file.

        lmid_label: str | None (optional, default = None),
            the label of the marker defining the right toe in the tdf file.

        rmid_label: str | None (optional, default = None),
            the label of the marker defining the right toe in the tdf file.

        height_thresh: float | int = 0.02,
            the minimum threshold in meters to be considered for assuming that
            one foot is in contact to the ground.

        Returns
        -------
        test: GaitTest
            the GaitTest class object.
        """
        if not (exists(file) and isinstance(file, str)):
            raise ValueError("file must be the path to an existing .tdf file.")
        tdf = read_tdf(file)
        try:
            markers = tdf["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
        except Exception:
            msg = "the Input file does not contain valid tracked data."
            raise ValueError(msg)
        labels = np.unique(markers.columns.get_level_values(0).astype(str))
        required = {
            "lheel": lheel_label,
            "rheel": rheel_label,
            "ltoe": ltoe_label,
            "rtoe": rtoe_label,
        }
        if lmid_label is not None:
            required["lmidfoot"] = lmid_label
        if rmid_label is not None:
            required["rmidfoot"] = rmid_label
        for key, value in required.items():
            if not np.any([i == key for i in labels]):
                msg = f"{value} not found. The available labels are "
                msg += f"{labels.tolist()}"
                raise ValueError(msg)
        required = {i: markers[v] for i, v in required.items()}
        return cls(preprocess=True, height_thresh=height_thresh, **required)
