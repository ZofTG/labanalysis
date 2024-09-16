"""jumptests module"""

#! IMPORTS


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import signalprocessing as sp
from .frames import StateFrame

__all__ = ["StaticUprightStance", "SquatJump", "SquatJumpTest"]

#! CONSTANTS

G = 9.80665  # acceleration of gravity in m/s^2


#! CLASSES


class StaticUprightStance(StateFrame):
    """
    class defining a static upright stance.

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
        return float(np.median(self.forceplatforms.Y.values.astype(float))) / G

    @property
    def emg_norms(self):
        """
        the EMG signal norms, i.e. a DataFrame with:
            - column names equal to the EMG signal names with units
            - index defining the target metric:
                * mean
                * std
                * min
                * q1
                * median
                * q3
                * max
        """
        norms = self.emgs.describe().drop(["count"], axis=0)
        norms.index = pd.Index(["mean", "std", "min", "q1", "median", "q3", "max"])
        norms.loc["iqr", norms.columns] = norms.loc["q3"] - norms.loc["q1"]
        return norms

    # * methods

    def to_stateframe(self):
        """return the actual object as StateFrame"""
        return super().copy()

    def copy(self):
        """create a copy of the object"""
        return self.from_stateframe(self)

    # * constructors

    def __init__(
        self,
        markers_raw: pd.DataFrame,
        forceplatforms_raw: pd.DataFrame,
        emgs_raw: pd.DataFrame,
    ):
        """
        generate a new StateFrame object

        Parameters
        ----------
        markers_raw: pd.DataFrame
            a dataframe containing raw markers data.

        forceplatforms_raw: pd.DataFrame
            a raw dataframe containing raw forceplatforms data.

        emgs_raw: pd.DataFrame
            a raw dataframe containing raw emg data.
        """
        super().__init__(
            markers_raw=markers_raw,
            forceplatforms_raw=forceplatforms_raw,
            emgs_raw=emgs_raw,
        )

        # ensure that the 'fRes' force platform object exists
        lbls = np.unique(self.forceplatforms.columns.get_level_values(0))
        if not any([i == "fRes" for i in lbls]):
            msg = "the provided .tdf file doe not contain the 'fRes' force data"
            raise ValueError(msg)

    @classmethod
    def from_tdf_file(cls, file: str):
        """
        generate a StaticUprightStance from a .tdf file

        Parameters
        ----------
        file : str
            a valid .tdf file containing (tracked) markers, force platforms and
            (optionally) EMG data

        Returns
        -------
        frame: StaticUprightStance
            a UprightStaticStance instance of the data contained in the .tdf file.
        """
        return super().from_tdf_file(file)

    @classmethod
    def from_stateframe(cls, obj: StateFrame):
        """
        generate a StaticUprightStance from a StateFrame object

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        Returns
        -------
        frame: StaticUprightStance
            a StaticUprightStance instance.
        """
        if not isinstance(obj, StateFrame):
            raise ValueError("obj must be a StateFrame object.")
        out = cls(
            markers_raw=obj.markers,
            forceplatforms_raw=obj.forceplatforms,
            emgs_raw=obj.emgs,
        )
        out._processed = obj.is_processed()
        out._marker_processing_options = obj.marker_processing_options
        out._forceplatform_processing_options = obj.forceplatform_processing_options
        out._emg_processing_options = obj.emg_processing_options
        return out


class SquatJump(StateFrame):
    """
    class defining a single SquatJump collected by markers, forceplatforms
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

    concentric_phase
        a StateFrame representing the concentric phase of the jump

    flight_phase
        a StateFrame representing the flight phase of the jump

    rate_of_force_development
        return the rate of force development over the concentric phase of the
        jump

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
    def concentric_phase(self):
        """
        return a StateFrame representing the concentric phase of the jump

        Procedure
        ---------
            1. get the index of the peak vertical velocity of S2 marker
            2. check for the last zero velocity occuring before the peak.
            3. look for the last sample with the feet in touch to the ground.
            4. return a slice of the data containing the subset defined by
            points 2 and 3.

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the concentric
            phase of the jump
        """
        # get the vertical velocity in S2
        s2_y = self.markers.S2.Y.values.astype(float).flatten()
        s2_t = self.markers.index.to_numpy()
        s2_v = sp.winter_derivative1(s2_y, s2_t)
        s2_y = s2_y[1:-1]
        s2_t = s2_t[1:-1]

        # get the index of the peak velocity
        max_v = np.argmax(s2_v)

        # look at the zeros in the vertical velocity occurring before max_v
        zeros_s2 = sp.crossings(s2_v[max_v], 0)[0]
        if len(zeros_s2) == 0:
            raise RuntimeError("No zero values found in vertical velocity.")

        # get the time instants corresponding to ground reaction force = zero
        grf_y = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grf_t = self.forceplatforms.index.to_numpy()
        zeros_grf = sp.crossings(grf_y, 0)[0]
        msg = "No zero values found in ground reaction force"
        if len(zeros_grf) == 0:
            raise RuntimeError(msg)

        # set the time start as the last zero in the vertical velocity
        time_start = s2_t[zeros_s2][-1]

        # set the time stop as the time instant occurring immediately before
        # the first zero in grf occurring after time_start
        time_stop = grf_t[zeros_grf]
        time_stop = time_stop[time_stop > time_start]
        if len(time_stop) == 0:
            msg += " after the start of the concentric phase."
            raise RuntimeError(msg)
        time_stop = grf_t[grf_t < time_stop[0]][-1]

        # return a slice of the available data
        return self.slice(time_start, time_stop)

    @property
    def flight_phase(self):
        """
        return a StateFrame representing the flight phase of the jump

        Procedure
        ---------
            1. get the batches of samples with ground reaction force being zero.
            2. take the longed batch.
            3. take the time corresponding to the start and stop of the batch.
            4. return a slice containing only the data corresponding to the
            detected start and stop values.

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the flight
            phase of the jump
        """

        # get the indices of the largest interval in the ground reaction force
        # with zeros
        grf_y = self.forceplatforms.fRes.FORCE.Y.values.astype(float).flatten()
        grf_t = self.forceplatforms.index.to_numpy()
        zeros_batches = sp.continuous_batches(grf_y == 0)
        msg = "No zero values found in ground reaction force"
        if len(zeros_batches) == 0:
            raise RuntimeError(msg)

        # take the time corresponding to the start and stop of the batch
        zeros_batch = zeros_batches[np.argmax([len(i) for i in zeros_batches])]
        if len(zeros_batch) < 2:
            raise RuntimeError("no flight phase detected")
        time_start, time_stop = grf_t[zeros_batch][[0, -1]]

        # return a slice of the available data
        return self.slice(time_start, time_stop)

    @property
    def rate_of_force_development(self):
        """
        return the rate of force development over the concentric phase of the
        jump in N/s
        """
        # get the vertical force data
        confp = self.concentric_phase.forceplatforms.fRes.FORCE
        grf = confp.Y.values.astype(float).flatten()
        time = confp.index.to_numpy()

        # get the rfd value
        return float(np.mean(sp.winter_derivative1(grf, time)))

    @property
    def velocity_at_toeoff(self):
        """return the vertical velocity at the toeoff in m/s"""

        # get the vertical velocity
        pos = self.markers.S2.Y.values.astype(float).flatten()
        time = self.markers.index.to_numpy()
        vel = sp.winter_derivative1(pos, time)

        # remove the first and last sample from time to be aligned with vel
        time = time[1:-1]

        # get the velocity at the first time instant in the flight phase
        loc = np.where(time <= self.flight_phase.markers.index[0])[0][0]
        return float(vel[loc])

    @property
    def jump_height(self):
        """return the jump height in m"""

        # get the vertical position of S2 at the flight phase
        pos = self.flight_phase.markers.S2.Y.values.astype(float).flatten()

        # get the difference between the first and highest sample
        maxh = np.max(pos)
        toeoffh = pos[0]
        return float(maxh - toeoffh)

    @property
    def concentric_power(self):
        """return the mean power in W generated during the concentric phase"""
        # get the concentric phase grf and vertical velocity
        con = self.concentric_phase
        s2y = con.markers.S2.Y.values.astype(float).flatten()
        s2t = con.markers.index.to_numpy()
        s2v = sp.winter_derivative1(s2y, s2t)
        s2t = s2t[1:-1]
        grf = con.forceplatforms.loc[s2t].fRes.FORCE.Y.values.astype(float).flatten()

        # return the mean power output
        return float(np.mean(grf * s2v))

    @property
    def muscle_activation(self):
        """
        return the mean muscle activation amplitude during the concentric phase
        """
        # get the concentric phase grf and vertical velocity
        con = self.concentric_phase
        return con.emgs.mean(axis=0)

    # * methods

    def to_stateframe(self):
        """return the actual object as StateFrame"""
        return super().copy()

    def copy(self):
        """create a copy of the object"""
        return self.from_stateframe(self)

    # * constructors

    def __init__(
        self,
        markers_raw: pd.DataFrame,
        forceplatforms_raw: pd.DataFrame,
        emgs_raw: pd.DataFrame,
    ):
        """
        generate a new StateFrame object

        Parameters
        ----------
        markers_raw: pd.DataFrame
            a dataframe containing raw markers data.

        forceplatforms_raw: pd.DataFrame
            a raw dataframe containing raw forceplatforms data.

        emgs_raw: pd.DataFrame
            a raw dataframe containing raw emg data.
        """
        super().__init__(
            markers_raw=markers_raw,
            forceplatforms_raw=forceplatforms_raw,
            emgs_raw=emgs_raw,
        )

        # ensure that the 'fRes', 'rFoot' and 'lFoot' force platform objects exist
        lbls = np.unique(self.forceplatforms.columns.get_level_values(0))
        required_fp = ["fRes", "lFoot", "rFoot"]
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

    @classmethod
    def from_tdf_file(cls, file: str):
        """
        generate a StaticUprightStance from a .tdf file

        Parameters
        ----------
        file : str
            a valid .tdf file containing (tracked) markers, force platforms and
            (optionally) EMG data

        Returns
        -------
        frame: StaticUprightStance
            a UprightStaticStance instance of the data contained in the .tdf file.
        """
        return super().from_tdf_file(file)

    @classmethod
    def from_stateframe(cls, obj: StateFrame):
        """
        generate a StaticUprightStance from a StateFrame object

        Parameters
        ----------
        obj: StateFrame
            a StateFrame instance

        Returns
        -------
        frame: StaticUprightStance
            a StaticUprightStance instance.
        """
        if not isinstance(obj, StateFrame):
            raise ValueError("obj must be a StateFrame object.")
        out = cls(
            markers_raw=obj.markers,
            forceplatforms_raw=obj.forceplatforms,
            emgs_raw=obj.emgs,
        )
        out._processed = obj.is_processed()
        out._marker_processing_options = obj.marker_processing_options
        out._forceplatform_processing_options = obj.forceplatform_processing_options
        out._emg_processing_options = obj.emg_processing_options
        return out


class SquatJumpTest:
    """
    Class Squat_Jump_test is used to analyze the processed file of the single squat jump

    Parameters
    ----------

    """


class _CMJ:
    """
    Counter movement jump class is used internally for processing the single counter movement jump

    Parameters
    ----------

    """


class Counter_Movement_Jump_test:
    """
    Class Counter_Movement_Jump_test is used to analyze the processed file of the single counter movement jump

    Parameters
    ----------

    """
