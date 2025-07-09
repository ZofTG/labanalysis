"""Counter Movement Jump Test module"""

#! IMPORTS


from typing import Iterable, Union

import numpy as np
import pandas as pd

from ...frames import EMGSignal, ForcePlatform, Point3D, Signal1D, Signal3D

from ... import signalprocessing as sp
from .squat_jump import SquatJump, SquatJumpTest

__all__ = ["CounterMovementJump", "CounterMovementJumpTest"]


#! CLASSES


class CounterMovementJump(SquatJump):

    @property
    def eccentric_phase(self):
        """
        return a StateFrame denoting the eccentric phase of the jump

        Returns
        -------
        phase: StateFrame
            a StateFrame containing the data corresponding to the concentric
            phase of the jump

        Procedure
        ---------
            1. define 'time_end' as the time instant corresponding to the start
            of the concentric phase. Please looka the 'concentric_phase'
            documentation to have a detailed description of the procedure used
            to extract this phase.
            2. look at the last positive speed value in the vertical S2 signal
            occurring before 'time_end'.
            3. define 'time_end' as the last peak in the grf occurring before
            the time defined in 2.
        """
        # get the time instant corresponding to the start of the concentric
        # phase
        t_end = self.index.to_numpy()
        t_end = t_end[t_end < self.concentric_phase.index[0]]
        t_end = float(round(t_end[-1], 3))

        # get the last peak in vertical position before the concentric phase
        s2 = self["s2"].dropna()
        s2y = s2[self.vertical_axis].values.astype(float).flatten()
        s2t = s2.index.to_numpy()

        # look at the last positive vertical speed value occuring before t_end
        s2v = sp.winter_derivative1(s2y)
        s2t = s2t[1:-1]
        batches = sp.continuous_batches(s2v[s2t < t_end] <= 0)
        if len(batches) == 0:
            raise RuntimeError("No eccentric phase has been found.")
        s2y_0 = float(round(s2t[batches[-1][0]], 3))  # type: ignore

        # take the last peak in vertical grf occurring before s2y_0
        grfy = self.grf.values.astype(float).flatten()
        grft = self.grf.index.to_numpy()
        idx = np.where(grft < s2y_0)[0]
        grf_pks = sp.find_peaks(grfy[idx])
        if len(grf_pks) == 0:
            t_start = float(round(grft[0], 3))
        else:
            t_start = float(round(grft[grf_pks[-1]], 3))  # type: ignore

        # get the time corresponding phase
        return self.slice(t_start, t_end)

    def __init__(
        self,
        s2: Point3D,
        left_foot: ForcePlatform,
        right_foot: ForcePlatform,
        **signals: Union[Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform],
    ):

        # check the inputs
        if (
            not isinstance(s2, Point3D)
            or not isinstance(left_foot, ForcePlatform)
            or not isinstance(right_foot, ForcePlatform)
        ):
            msg = "s2 must be a Point3D object, while left_foot and right_foot "
            msg += "have to be ForcePlatform objects."
            raise TypeError(msg)

        # build the object
        super().__init__(
            s2=s2,
            left_foot=left_foot,
            right_foot=right_foot,
            **signals,
        )


class CounterMovementJumpTest(SquatJumpTest):

    def __init__(
        self,
        jumps: list[CounterMovementJump],
    ):
        super().__init__(baseline, jumps)  # type: ignore
