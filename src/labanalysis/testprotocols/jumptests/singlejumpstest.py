"""singlejumps test module"""

#! IMPORTS


__all__ = ["SingleJumpsTest"]


#! CLASSES


import pandas as pd

from ...frames.jumps.singlejump import SingleJump
from ..protocols import Participant, TestProtocol


class SingleJumpsTest(TestProtocol):

    _sj_list: list[SingleJump]
    _cmj_list: list[SingleJump]
    _dj_list: list[SingleJump]

    def __init__(
        self,
        participant: Participant,
        normative_data_path: str = "",
        squat_jumps: list[SingleJump] = [],
        counter_movement_jumps: list[SingleJump] = [],
        drop_jumps: list[SingleJump] = [],
    ):
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)

        # check jumps
        for jump in squat_jumps + counter_movement_jumps + drop_jumps:
            if not isinstance(jump, SingleJump):
                raise ValueError("provided jumps must all be SingleJump instances.")
        self._sj_list = squat_jumps
        self._cmj_list = counter_movement_jumps
        self._dj_list = drop_jumps

    @property
    def squat_jumps(self):
        return self._sj_list

    @property
    def counter_movement_jumps(self):
        return self._cmj_list

    @property
    def drop_jumps(self):
        return self._dj_list

    @property
    def results(self):
        grf = []
        metrics = []
        for i, jump in enumerate(self.squat_jumps):

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
