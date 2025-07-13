"""repeatedjump module"""

#! IMPORTS


import numpy as np

from ...signalprocessing import butterworth_filt, continuous_batches, rms_filt
from ..processingpipeline import ProcessingPipeline
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from ..timeseriesrecords.timeseriesrecord import TimeseriesRecord
from .singlejump import SingleJump

__all__ = ["RepeatedJump"]


#! CLASSES


class RepeatedJump(TimeseriesRecord):
    """
    Represents a repeated jump trial, providing methods and properties to analyze
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
    def jumps(self):
        vgrf = self.vertical_force
        time = self.index

        # get the batches with grf lower than 30N (i.e flight phases)
        batches = continuous_batches(vgrf.data.flatten() <= 30)

        # remove those batches resulting in too short flight phases
        # a jump is assumed valid if the elevation is higher than 5 cm
        # (i.e. ~0.2s flight time)
        fsamp = 1 / np.mean(np.diff(time))
        min_samples = int(round(0.2 * fsamp))
        batches = [i for i in batches if len(i) >= min_samples]
        msg = "No jumps have been found."
        if len(batches) == 0:
            raise RuntimeError(msg)

        # ensure that the first jump does not start with a flight
        start_index = 0
        if batches[0][0] == 0:
            start_index = batches[0][-1]
            batches = batches[1:]

        # separate each jump
        jumps: list[SingleJump] = []
        for batch in batches:
            start_time = time[start_index]
            stop_time = time[batch[-1]]
            sliced = self[start_time:stop_time]
            new_jump = SingleJump(
                bodymass_kg=self.bodymass_kg,
                left_foot_ground_reaction_force=sliced.get(
                    "left_foot_ground_reaction_force"
                ),
                right_foot_ground_reaction_force=sliced.get(
                    "right_foot_ground_reaction_force"
                ),
                vertical_axis=self.vertical_axis,
                anteroposterior_axis=self.anteroposterior_axis,
                strip=False,
                reset_time=False,
                process_inputs=False,
                **{
                    i: v
                    for i, v in sliced.items()
                    if i
                    not in [
                        "left_foot_ground_reaction_force",
                        "right_foot_ground_reaction_force",
                    ]
                },
            )
            jumps += [new_jump]

        return jumps

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

        # build the object
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
