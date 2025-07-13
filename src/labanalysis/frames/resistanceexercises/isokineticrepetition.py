"""
isokinetic exercise module
"""

#! IMPORTS

from typing import Literal

import numpy as np
import pandas as pd

from ...constants import G
from ...io.read.biostrength import BiostrengthProduct
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from ..timeseriesrecords.timeseriesrecord import TimeseriesRecord

#! CONSTANTS


__all__ = ["IsokineticRepetition"]

#! CLASSES


class IsokineticRepetition(TimeseriesRecord):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    # * class variables

    _product: BiostrengthProduct
    _side: Literal["bilateral", "left", "right"]

    # * attributes

    @property
    def side(self):
        """get the side of the test"""
        return self._side

    @property
    def product(self):
        """return the product on which the test has been performed"""
        return self._product

    @property
    def peak_load(self):
        """return the ending position of the repetitions"""
        return float(np.max(self.product.load_lever_kgf))

    @property
    def estimated_1rm(self):
        """return the predicted 1RM"""
        b1, b0 = self.product.rm1_coefs
        return self.peak_load * b1 + b0

    @property
    def muscle_activations(self):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = self.emgsignals
        if len(emgs) == 0:
            return pd.DataFrame()

        # check the presence of left and right muscles
        muscles = {}
        for emg in emgs.values():
            name = emg.muscle_name
            side = emg.side
            if side != self.side:
                continue
            muscles[f"{side}_{name}"] = np.asarray(emg.data, float).mean()

        return pd.DataFrame(pd.Series(muscles)).T

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
            f"peak_load_{self['force'].unit}": self.peak_load,
            "estimated_1rm": self.estimated_1rm,
        }
        new = pd.DataFrame(pd.Series(new)).T
        return pd.concat([new, self.muscle_activations], axis=1)

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):

        # check the input
        if not issubclass(product.__class__, BiostrengthProduct):
            raise ValueError("'product' must be a valid Biostrength Product.")
        if not side in ["bilateral", "left", "right"]:
            raise ValueError("'side' must be any of 'bilateral', 'left', 'right'")

        # get the required data
        time_s = product.time_s.tolist()
        force = Signal1D(
            product.load_lever_kgf * G,
            time_s,
            "N",
            "force",
        )
        position = Signal1D(
            product.position_lever_m,
            time_s,
            "m",
            "position",
        )
        super().__init__(
            force=force,
            position=position,
            strip=True,
            reset_time=True,
            **signals,
        )

        # get the raw data
        self._product = product  # type: ignore
        self._side = side
