"""
isokinetic test module
"""

#! IMPORTS

from typing import Literal

import numpy as np
import pandas as pd
from ...frames.resistanceexercises.isokineticexercise import IsokineticExercise
from ...io.read.biostrength import BiostrengthProduct
from ..protocols import Participant, TestProtocol

#! CONSTANTS


__all__ = ["Isokinetic1RMTest"]

#! CLASSES


class Isokinetic1RMTest(IsokineticExercise, TestProtocol):

    def __init__(
        self,
        participant: Participant,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        normative_data_path: str = "",
    ):
        super().__init__(product=product, side=side)
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)

    @property
    def results(self):
        grf = []
        metrics = []
        for i, rep in enumerate(self.repetitions):

            # add grf
            grf_cycle = pd.DataFrame(rep.to_dataframe())
            grf_cycle.insert(0, "Time", grf_cycle.index)
            grf_cycle.insert(0, "Repetition", i + 1)
            grf_cycle.insert(0, "Side", rep.side)
            grf += [grf_cycle]

            # add summary metrics
            metrics_cycle = rep.output_metrics
            metrics_cycle.insert(0, "Repetition", i + 1)
            metrics_cycle.insert(0, "Side", rep.side)
            metrics += [metrics_cycle]

        # outcomes
        out = {
            "summary": pd.concat(metrics, ignore_index=True),
            "analytics": {"grf": pd.concat(grf, ignore_index=True)},
        }
        return out
