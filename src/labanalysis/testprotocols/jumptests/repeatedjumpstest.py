"""repeated jumps test module"""

#! IMPORTS


__all__ = ["RepeatedJumpsTest"]


#! CLASSES


import pandas as pd

from ...frames.timeseries.signal1d import Signal1D
from ...frames.timeseries.signal3d import Signal3D
from ...frames.timeseries.emgsignal import EMGSignal
from ...frames.timeseries.point3d import Point3D
from ...frames.timeseriesrecords.forceplatform import ForcePlatform
from ...frames.jumps.repeatedjumps import RepeatedJump
from ..protocols import Participant, TestProtocol


class RepeatedJumpsTest(RepeatedJump, TestProtocol):

    def __init__(
        self,
        participant: Participant,
        normative_data_path: str = "",
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
        strip: bool = True,
        reset_time: bool = True,
        process_inputs: bool = True,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        super().__init__(
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            vertical_axis=vertical_axis,
            anteroposterior_axis=anteroposterior_axis,
            strip=strip,
            reset_time=reset_time,
            process_inputs=process_inputs,
            **signals,
        )
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)

    @property
    def results(self):
        grf = []
        metrics = []
        for i, jump in enumerate(self.jumps):

            # add grf
            grf_cycle = jump.vertical_force
            start_time = max(0, jump.concentric_phase.index[0] - 0.5)
            end_time = min(jump.flight_phase.index[-1] + 0.5, jump.index[-1])
            grf_cycle = grf_cycle[start_time:end_time].reset_time()
            grf_cycle = pd.DataFrame(grf_cycle.to_dataframe())
            grf_cycle.insert(0, "Time", grf_cycle.index)
            grf_cycle.insert(0, "Jump", i + 1)
            grf_cycle.insert(0, "Type", "Squat Jump")
            grf += [grf_cycle]

            # add summary metrics
            metrics_cycle = jump.output_metrics
            metrics_cycle.insert(0, "Jump", i + 1)
            metrics_cycle.insert(0, "Type", "Squat Jump")
            metrics += [metrics_cycle]

        # outcomes
        out = {
            "summary": pd.concat(metrics, ignore_index=True),
            "analytics": {
                "grf": pd.concat(grf, ignore_index=True),
            },
        }
        return out
