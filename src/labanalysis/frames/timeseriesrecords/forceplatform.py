"""forceplatform module"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pint

from ..timeseries import *
from .timeseriesrecord import *

ureg = pint.UnitRegistry()


__all__ = ["ForcePlatform"]


class ForcePlatform(TimeseriesRecord):
    """
    Represents a force platform measurement system.

    Parameters
    ----------
    origin : Point3D
        The center of pressure (CoP) location over time.
    force : Signal3D
        The 3D ground reaction force vector over time.
    torque : Signal3D
        The 3D torque vector over time.
    vertical_axis : str, optional
        The label for the vertical axis (default "Y").
    anteroposterior_axis : str, optional
        The label for the anteroposterior axis (default "Z").
    strip : bool, optional
        If True, remove leading/trailing rows or columns that are all NaN from all contained objects (default True).
    reset_time : bool, optional
        If True, reset the time index to start at zero for all contained objects (default True).

    Methods
    -------
    copy()
        Return a deep copy of the ForcePlatform.
    """

    def __init__(
        self,
        origin: Point3D,
        force: Signal3D,
        torque: Signal3D,
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
        strip: bool = True,
        reset_time: bool = True,
    ):
        """
        Initialize a ForcePlatform.

        Parameters
        ----------
        origin : Point3D
        force : Signal3D
        torque : Signal3D

        Raises
        ------
        TypeError
            If any argument is not of the correct type.
        """
        if not isinstance(origin, Point3D):
            raise TypeError("origin must be an instance of Point3D")
        if not isinstance(force, Signal3D):
            raise TypeError("force must be an instance of Signal3D")
        if not isinstance(torque, Signal3D):
            raise TypeError("torque must be an instance of Signal3D")

        super().__init__(
            origin=origin,
            force=force,
            torque=torque,
            vertical_axis=vertical_axis,
            anteroposterior_axis=anteroposterior_axis,
            strip=strip,
            reset_time=reset_time,
        )

    @property
    def vertical_force(self):
        return Signal1D(
            np.asarray(self["force"][self.vertical_axis], float),
            self.index.tolist(),
            self["force"].unit,
            self.vertical_axis,
        )

    @property
    def centre_of_pressure(self):
        cop = np.cross(self["force"].data, self["torque"].data) / np.dot(
            self["force"].data, self["torque"].data
        )
        return Point3D(
            cop,
            self.index.tolist(),
            self["origin"].unit,
            self["origin"].columns,
        )

    def copy(self):
        """
        Return a deep copy of the ForcePlatform.

        Returns
        -------
        ForcePlatform
            A new ForcePlatform object with the same data.
        """
        return ForcePlatform(
            origin=self.origin.copy(),
            force=self.force.copy(),
            torque=self.torque.copy(),
            vertical_axis=self.vertical_axis,
            anteroposterior_axis=self.anteroposterior_axis,
            strip=False,
            reset_time=False,
        )
