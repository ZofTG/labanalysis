"""
isokinetic exercise module
"""

#! IMPORTS

from typing import Literal

import numpy as np

from ...constants import G
from ...io.read.biostrength import BiostrengthProduct
from ...signalprocessing import butterworth_filt, continuous_batches, winter_derivative1
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from ..timeseriesrecords.timeseriesrecord import TimeseriesRecord
from .isokineticrepetition import IsokineticRepetition

#! CONSTANTS


__all__ = ["IsokineticExercise"]

#! CLASSES


class IsokineticExercise(TimeseriesRecord):
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

    _side: Literal["bilateral", "left", "right"]
    _product: BiostrengthProduct

    # * attributes

    @property
    def side(self):
        """get the side of the test"""
        return self._side

    @property
    def repetitions(self):
        """return the tracked repetitions data"""
        tarr = np.asarray(self.index, float).flatten()
        farr = np.asarray(self.force.data, float).flatten()
        parr = np.asarray(self.position.data, float).flatten()
        if abs(np.min(parr)) > abs(np.max(parr)):
            parr *= -1
        parr -= parr[0]
        varr = winter_derivative1(parr, tarr)
        farr = farr[1:-1]
        parr = farr * varr
        fsamp = float(1 / np.mean(np.diff(tarr)))
        parr = butterworth_filt(
            arr=parr,
            fcut=1,
            fsamp=fsamp,
            order=6,
            ftype="lowpass",
            phase_corrected=True,
        )
        start_batches = continuous_batches(parr > 5)
        if len(start_batches) == 0:
            raise RuntimeError("No repetitions have been found")
        samples = np.argsort([np.max(parr[i]) for i in start_batches])[::-1][:3]
        starts = [start_batches[i][0] for i in np.sort(samples)]
        stop_batches = continuous_batches(parr < -5)
        repetitions: list[IsokineticRepetition] = []
        for start in starts:
            stops = [
                i[-1]
                for i in stop_batches
                if i[0] > start and tarr[i[-1]] - tarr[i[0]] > 0.5
            ]
            if len(stops) > 0:
                stop = int(np.min(stops) + 1)
                sub = self[tarr[start] : tarr[stop]]
                repetitions += [
                    IsokineticRepetition(
                        self.product.slice(tarr[start], tarr[stop]),
                        self.side,
                        **{i: v for i, v in sub.items()},
                    )
                ]
        return repetitions

    @property
    def product(self):
        """return the product on which the test has been performed"""
        return self._product

    @property
    def peak_load_kgf(self):
        """return the ending position of the repetitions"""
        return float(np.max([i.peak_load for i in self.repetitions]))

    @property
    def estimated_1rm(self):
        """return the predicted 1RM"""
        b1, b0 = self.product.rm1_coefs
        return self.peak_load_kgf * b1 + b0

    # * constructors

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

    @classmethod
    def from_file(
        cls,
        filename: str,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
    ):
        return cls(product=product.from_txt_file(filename), side=side)
