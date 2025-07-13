"""plank balance test module"""

#! IMPORTS


__all__ = ["PlankBalanceTest"]


#! CLASSES


import pandas as pd
from ...frames.timeseries.point3d import Point3D
from ...frames.timeseries.emgsignal import EMGSignal
from ...frames.timeseries.signal3d import Signal3D
from ...frames.timeseries.signal1d import Signal1D
from ...frames.timeseriesrecords.forceplatform import ForcePlatform
from ..protocols import Participant, TestProtocol
from ...frames.stances.pronestance import ProneStance


class PlankBalanceTest(ProneStance, TestProtocol):

    def __init__(
        self,
        participant: Participant,
        left_foot_ground_reaction_force: ForcePlatform,
        right_foot_ground_reaction_force: ForcePlatform,
        left_hand_ground_reaction_force: ForcePlatform,
        right_hand_ground_reaction_force: ForcePlatform,
        normative_data_path: str = "",
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
        strip: bool = True,
        reset_time: bool = True,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):

        super().__init__(
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            vertical_axis=vertical_axis,
            anteroposterior_axis=anteroposterior_axis,
            strip=strip,
            reset_time=reset_time,
            **signals,
        )
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)

    @property
    def results(self):
        cop = self.centre_of_pressure.to_dataframe()
        horizontal_axes = [self.anteroposterior_axis, self.lateral_axis]
        cop = cop[[i for i in cop.columns if i[0] in horizontal_axes]]
        return {
            "summary": self.output_metrics,
            "analytics": {"centre_of_pressure": pd.DataFrame(cop)},
        }
