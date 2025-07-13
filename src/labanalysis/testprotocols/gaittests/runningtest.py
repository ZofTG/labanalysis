"""Running test module"""

#! IMPORTS


from typing import Literal

import pandas as pd

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...frames.gaits.runningexercise import RunningExercise
from ...frames.timeseries.emgsignal import EMGSignal
from ...frames.timeseries.point3d import Point3D
from ...frames.timeseries.signal1d import Signal1D
from ...frames.timeseries.signal3d import Signal3D
from ...frames.timeseriesrecords.forceplatform import ForcePlatform
from ..protocols import Participant, TestProtocol

__all__ = ["RunningTest"]


#! CLASSESS


class RunningTest(RunningExercise, TestProtocol):
    def __init__(
        self,
        participant: Participant,
        normative_data_path: str = "",
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_metatarsal_head: Point3D | None = None,
        right_metatarsal_head: Point3D | None = None,
        ground_reaction_force: ForcePlatform | None = None,
        ground_reaction_force_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        vertical_axis: Literal["X", "Y", "Z"] = "Y",
        antpos_axis: Literal["X", "Y", "Z"] = "Z",
        strip: bool = True,
        reset_time: bool = True,
        process_inputs: bool = True,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        super().__init__(
            algorithm=algorithm,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_metatarsal_head=left_metatarsal_head,
            right_metatarsal_head=right_metatarsal_head,
            ground_reaction_force=ground_reaction_force,
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
            vertical_axis=vertical_axis,
            antpos_axis=antpos_axis,
            strip=strip,
            reset_time=reset_time,
            process_inputs=process_inputs,
            **extra_signals,
        )
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)

    @property
    def results(self):
        cop = []
        grf = []
        metrics = []
        horizontal_axes = [self.anteroposterior_axis, self.lateral_axis]
        for i, cycle in enumerate(self.cycles):

            # add cop
            cop_cycle = cycle.centre_of_pressure.reset_time()
            cop_cycle = pd.DataFrame(cop_cycle.to_dataframe())
            cop_cycle.insert(0, "Time", cop_cycle.index)
            cop_cycle.insert(0, "Cycle", i + 1)
            cop += [cop_cycle[horizontal_axes]]

            # add grf
            grf_cycle = cycle.vertical_force.reset_time()
            grf_cycle = pd.DataFrame(grf_cycle.to_dataframe())
            grf_cycle.insert(0, "Time", grf_cycle.index)
            grf_cycle.insert(0, "Cycle", i + 1)
            grf += [grf_cycle]

            # add summary metrics
            metrics_cycle = cycle.output_metrics
            metrics_cycle.insert(0, "Cycle", i + 1)
            metrics += [metrics_cycle]

        # outcomes
        out = {
            "summary": pd.concat(metrics, ignore_index=True),
            "analytics": {
                "cop": pd.concat(cop, ignore_index=True),
                "grf": pd.concat(grf, ignore_index=True),
            },
        }
        return out
