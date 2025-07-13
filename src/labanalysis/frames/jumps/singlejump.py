"""singlejump module"""

#! IMPORTS


import numpy as np
import pandas as pd

from ...constants import G
from ...signalprocessing import *
from ..processingpipeline import ProcessingPipeline
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from ..timeseriesrecords.timeseriesrecord import TimeseriesRecord

__all__ = ["SingleJump"]


#! CLASSES


class SingleJump(TimeseriesRecord):
    """
    Represents a single jump trial, providing methods and properties to analyze
    phases, forces, and performance metrics of the jump.

    Parameters
    ----------
    bodymass_kg : float
        The subject's body mass in kilograms.
    left_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the left foot.
    right_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the right foot.
    vertical_axis : str, optional
        Name of the vertical axis in the force data (default "Y").
    anteroposterior_axis : str, optional
        Name of the anteroposterior axis in the force data (default "X").
    **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        Additional signals to include in the record.

    Attributes
    ----------
    _bodymass_kg : float
        The subject's body mass in kilograms.
    _vertical_axis : str
        Name of the vertical axis.
    _antpos_axis : str
        Name of the anteroposterior axis.

    Properties
    ----------
    vertical_axis : str
        The vertical axis label.
    anteroposterior_axis : str
        The anteroposterior axis label.
    lateral_axis : str
        The lateral axis label.
    vertical_force : np.ndarray
        The mean vertical ground reaction force across both feet.
    side : str
        "bilateral", "left", or "right" depending on available force data.
    bodymass_kg : float
        The subject's body mass in kilograms.
    eccentric_phase : TimeseriesRecord
        Data for the eccentric phase of the jump.
    concentric_phase : TimeseriesRecord
        Data for the concentric phase of the jump.
    flight_phase : TimeseriesRecord
        Data for the flight phase of the jump.
    contact_time_s : float
        Duration of the contact phase (s).
    flight_time_s : float
        Duration of the flight phase (s).
    takeoff_velocity_ms : float
        Takeoff velocity at the end of the concentric phase (m/s).
    elevation_cm : float
        Jump elevation (cm) calculated from flight time.
    muscle_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from EMG signals (if available).
    force_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from force signals.
    output_metrics : pd.DataFrame
        Summary metrics for the jump.

    Methods
    -------
    __init__(...)
        Initialize a Jump object.
    from_tdf(...)
        Create a Jump object from a TDF file.
    """

    _bodymass_kg: float

    def _get_coordination_and_asymmetry(
        self, left: np.ndarray, right: np.ndarray, unit: str
    ):
        line = {
            f"left_{unit}": np.mean(left),
            f"right_{unit}": np.mean(right),
        }
        line["coordination_%"] = np.corrcoef(left, right)[0][1] * 100
        den = line["right_avg"] + line["left_avg"]
        balance = line["right_avg"] / den * 100 - 50
        line["balance_%"] = balance
        return line

    @property
    def processing_pipeline(self):
        """
        Get the default processing pipeline for this test.

        Returns
        -------
        ProcessingPipeline
        """

        # emg
        def process_emg(channel: EMGSignal):
            channel -= channel.mean()
            fsamp = 1 / np.mean(np.diff(channel.index))
            channel.apply(
                butterworth_filt,
                fcut=[20, 450],
                fsamp=fsamp,
                order=4,
                ftype="bandpass",
                phase_corrected=True,
                inplace=True,
                axis=1,
            )
            channel.apply(
                rms_filt,
                order=int(0.2 * fsamp),
                pad_style="reflect",
                offset=0.5,
                inplace=True,
                axis=1,
            )
            return channel

        # points3d
        def process_point3d(point: Point3D):
            point.fillna(inplace=True)
            fsamp = 1 / np.mean(np.diff(point.index))
            point = point.apply(
                butterworth_filt,
                fcut=6,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
            )
            return point

        # forceplatforms
        def process_forceplatforms(fp: ForcePlatform):

            def process_signal3d(signal: Signal3D):
                signal.fillna(inplace=True, value=0)
                fsamp = 1 / np.mean(np.diff(signal.index))
                signal = signal.apply(
                    butterworth_filt,
                    fcut=[10, 100],
                    fsamp=fsamp,
                    order=4,
                    ftype="bandstop",
                    phase_corrected=True,
                )
                return signal

            force_platforms_processing_pipeline = ProcessingPipeline(
                point3d_funcs=[process_point3d],
                signal3d_funcs=[process_signal3d],
            )

            fp.apply(force_platforms_processing_pipeline, inplace=True)
            return fp

        return ProcessingPipeline(
            emgsignal_funcs=[process_emg],
            point3d_funcs=[process_point3d],
            forceplatform_funcs=[process_forceplatforms],
        )

    @property
    def side(self):
        """
        Returns which side(s) have force data.

        Returns
        -------
        str
            "bilateral", "left", or "right".
        """
        left_foot = self.get("ground_reaction_force_left_foot")
        right_foot = self.get("ground_reaction_force_left_foot")
        if left_foot is not None and right_foot is not None:
            return "bilateral"
        if left_foot is not None:
            return "left"
        if right_foot is not None:
            return "right"
        raise ValueError("both left_foot and right_foot are None")

    @property
    def bodymass_kg(self):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        return self._bodymass_kg

    @property
    def eccentric_phase(self):
        """
        Returns the eccentric phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the eccentric phase.
        """
        # get the time instant corresponding to the start of the concentric
        # phase
        t_end = self.index
        t_end = t_end[t_end < self.concentric_phase.index[0]]
        t_end = t_end[-1]

        # look at the last positive vertical speed value occuring before t_end
        vgrf = self.vertical_force
        time = self.index
        der1 = winter_derivative1(vgrf, time)
        t_start = np.where((der1 < 0) & (time[1:-1] < t_end))[0]

        # return a slice of the available data
        sliced = self[t_start:t_end]
        out = TimeseriesRecord()
        for key, value in sliced.items():
            out[key] = value
        return out

    @property
    def concentric_phase(self):
        """
        Returns the concentric phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the concentric phase.

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
        flight_start = self.flight_phase.index[0]
        time = self.index
        con_end = time[time < flight_start][-1]

        # take the highest peak in the vertical ground reaction force
        # before the flight
        grf = self.vertical_force
        pk = find_peaks(grf)
        pk = pk[time[pk] < flight_start]
        pk = pk[np.argmax(grf[pk])]

        # get the last local minima in the vertical ground reaction force
        # occurring before the peak
        mn = find_peaks(-grf)
        mn = mn[mn < pk]
        con_start = time[0 if len(mn) == 0 else mn[-1]]

        # return a slice of the available data
        sliced = self[con_start:con_end]
        out = TimeseriesRecord()
        for key, value in sliced.items():
            out[key] = value
        return out

    @property
    def flight_phase(self):
        """
        Returns the flight phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the flight phase.

        Procedure
        ---------
            1. get the longest batch with grf lower than 30N.
            2. define 'flight_start' as the first local minima occurring after
            the start of the detected batch.
            3. define 'flight_end' as the last local minima occurring before the
            end of the detected batch.
        """

        # get the longest batch with grf lower than 30N
        grfy = self.vertical_force
        grft = self.time
        batches = continuous_batches(grfy <= 30)
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
        sliced = self[time_start:time_stop]
        out = TimeseriesRecord()
        for key, value in sliced.items():
            out[key] = value
        return out

    @property
    def contact_time_s(self):
        """
        Returns the duration of the contact phase (eccentric + concentric).

        Returns
        -------
        float
            Contact time in seconds.
        """
        ecc = self.eccentric_phase
        con = self.concentric_phase
        if ecc.shape[0] == 0:
            start = con.index[0]
        end = con.index[-1]
        return end - start

    @property
    def flight_time_s(self):
        """
        Returns the duration of the flight phase.

        Returns
        -------
        float
            Flight time in seconds.
        """
        time = self.flight_phase.index
        return time[-1] - time[0]

    @property
    def takeoff_velocity_ms(self):
        """
        Returns the takeoff velocity at the end of the concentric phase.

        Returns
        -------
        float
            Takeoff velocity in m/s.
        """

        # get the ground reaction force during the concentric phase
        time = self.index
        con = self.concentric_phase
        con_idx = (time >= con.index[0]) & (time <= con.index[-1])
        con_grf = self.vertical_force[con_idx]
        con_time = time[con_idx]

        # get the output velocity
        weight = self.bodymass_kg * G
        net_grf = con_grf - weight
        return float(np.trapezoid(net_grf, con_time) / weight * G)

    @property
    def elevation_cm(self):
        """
        Returns the jump elevation in centimeters, calculated from flight time.

        Returns
        -------
        float
            Jump elevation in cm.
        """
        flight_time = self.flight_phase.index
        flight_time = flight_time[-1] - flight_time[0]
        return (flight_time**2) * G / 8 * 100

    @property
    def muscle_coordination_and_asymmetry(self):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        # check if a bilateral jump was performed
        # (otherwise it makes no sense to test balance)
        if self.side != "bilateral":
            return pd.DataFrame()

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = self.concentric_phase.emgsignals
        if len(emgs) == 0:
            return pd.DataFrame()

        # check the presence of left and right muscles
        muscles = {}
        for emg in emgs.values():
            name = emg.muscle_name
            side = emg.side
            if side not in ["left", "right"]:
                continue
            if name not in list(muscles.keys()):
                muscles[name] = {}

            # get the area under the curve of the muscle activation
            muscles[name][side] = np.asarray(emg.data, float).flatten()

        # remove those muscles not having both sides
        muscles = {i: v for i, v in muscles.items() if len(v) == 2}

        # calculate coordination and imbalance between left and right side
        out = {}
        for muscle, sides in muscles.items():
            params = self._get_coordination_and_asymmetry(
                **sides,
                unit=emgs[muscle].unit,
            )
            out.update(**{f"{muscle}_{i}": v for i, v in params.items()})

        return pd.DataFrame(pd.Series(out)).T

    @property
    def force_coordination_and_asymmetry(self):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics, or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        if left_foot is None or right_foot is None:
            return pd.DataFrame()
        left_foot = np.asarray(left_foot["force"][self.vertical_axis], float).flatten()
        right_foot = np.asarray(
            right_foot["force"][self.vertical_axis], float
        ).flatten()

        # get the pairs to be tested
        pairs = {"lower_limbs": {"left_foot": left_foot, "right_foot": right_foot}}

        # calculate balance and coordination
        out = []
        unit = self.vertical_force.unit
        for region, pair in pairs.items():
            left, right = list(pair.values())
            fit = self._get_coordination_and_balance(
                left / G / self.bodymass_kg,
                right / G / self.bodymass_kg,
                unit,
            )
            line = {f"force_{i}": v for i, v in fit.items()}
            line = pd.DataFrame(pd.Series(line)).T
            line.insert(0, "region", region)
            out += [line]

        return pd.concat(out, ignore_index=True)

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = {
            "type": self.__class__.__name__,
            "side": self.side,
            "elevation_cm": self.elevation_cm,
            "takeoff_velocity_m/s": self.takeoff_velocity_ms,
            "contact_time_s": self.contact_time_s,
            "flight_time_s": self.flight_time_s,
        }
        new = pd.DataFrame(pd.Series(new)).T
        return pd.concat(
            [
                new,
                self.force_coordination_and_asymmetry,
                self.muscle_coordination_and_asymmetry,
            ],
            axis=1,
        )

    def __init__(
        self,
        bodymass_kg: float,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
        strip: bool = True,
        reset_time: bool = True,
        process_inputs: bool = True,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a Jump object.

        Parameters
        ----------
        bodymass_kg : float
            The subject's body mass in kilograms.
        left_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the left foot.
        right_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the right foot.
        vertical_axis : str, optional
            Name of the vertical axis in the force data (default "Y").
        anteroposterior_axis : str, optional
            Name of the anteroposterior axis in the force data (default "X").
        **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
            Additional signals to include in the record.

        Raises
        ------
        TypeError
            If left_foot or right_foot is not a ForcePlatform.
        ValueError
            If axes are not valid or bodymass_kg is not a float or int.
        """

        # check the inputs
        forces = {}
        if left_foot_ground_reaction_force is not None:
            if not isinstance(left_foot_ground_reaction_force, ForcePlatform):
                raise ValueError(
                    "left_foot_ground_reaction_force must be a ForcePlatform"
                    + " instance or None."
                )
            forces["left_foot_ground_reaction_force"] = left_foot_ground_reaction_force
        if right_foot_ground_reaction_force is not None:
            if not isinstance(right_foot_ground_reaction_force, ForcePlatform):
                raise ValueError(
                    "right_foot_ground_reaction_force must be a ForcePlatform"
                    + " instance or None."
                )
            forces["right_foot_ground_reaction_force"] = (
                right_foot_ground_reaction_force
            )
        if len(forces) == 0:
            raise ValueError(
                "at least one of 'left_foot_ground_reaction_force' or"
                + "'right_foot_ground_reaction_force' must be ForcePlatform"
                + " instances."
            )

        # build
        super().__init__(
            vertical_axis=vertical_axis,
            anteroposterior_axis=anteroposterior_axis,
            strip=strip,
            reset_time=reset_time,
            **signals,
            **forces,
        )

        # check the inputs
        try:
            self._bodymass_kg = float(bodymass_kg)
        except Exception as exc:
            raise ValueError("bodymass_kg must be a float or int")

        # evaluate processing data
        if not isinstance(process_inputs, bool):
            raise ValueError("process_inputs must be True or False")
        if process_inputs:
            self.apply(self.processing_pipeline, inplace=True)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        bodymass_kg: float | int,
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
        strip: bool = True,
        reset_time: bool = True,
        process_inputs: bool = True,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
    ):
        """
        Create a Jump object from a TDF file.

        Parameters
        ----------
        file : str
            Path to the TDF file.
        bodymass_kg : float or int
            The subject's body mass in kilograms.
        vertical_axis : str, optional
            Name of the vertical axis in the force data.
        anteroposterior_axis : str, optional
            Name of the anteroposterior axis in the force data.
        left_foot_ground_reaction_force : str or None, optional
            Key for left foot force data.
        right_foot_ground_reaction_force : str or None, optional
            Key for right foot force data.

        Returns
        -------
        Jump
            A Jump object created from the TDF file.
        """

        record = super().from_tdf(file)
        left_foot = record.get(left_foot_ground_reaction_force)
        right_foot = record.get(right_foot_ground_reaction_force)
        extra_signals = {
            **record.signals1d,
            **record.signals3d,
            **record.points3d,
            **record.emgsignals,
            **{
                i: v
                for i, v in record.forceplatforms
                if i
                not in [
                    left_foot_ground_reaction_force,
                    right_foot_ground_reaction_force,
                ]
            },
        }
        return cls(
            bodymass_kg=bodymass_kg,
            vertical_axis=vertical_axis,
            anteroposterior_axis=anteroposterior_axis,
            left_foot_ground_reaction_force=left_foot,
            right_foot_ground_reaction_force=right_foot,
            strip=strip,
            reset_time=reset_time,
            process_inputs=process_inputs,
            **extra_signals,
        )
