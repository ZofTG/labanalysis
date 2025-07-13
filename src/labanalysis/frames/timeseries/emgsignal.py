"""
emgsignal module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


from typing import Literal
import pint

from .frames import Signal1D

ureg = pint.UnitRegistry()


__all__ = ["EMGSignal"]


class EMGSignal(Signal1D):
    """
    A 1D EMG signal, automatically converted to microvolts (uV).
    """

    _muscle_name: str
    _side: Literal["left", "right", "bilateral"]

    def __init__(
        self,
        data,
        index,
        muscle_name: str,
        side: Literal["left", "right", "bilateral"],
        unit: str | pint.Quantity = "uV",
    ):
        """
        Initialize an EMGSignal.

        Parameters
        ----------
        data : array-like
            2D data array with one column.
        index : list of float
            Time values.
        muscle_name : str
            Name of the muscle.
        side : {'left', 'right', 'bilateral'}
            Side of the body.
        unit : str or pint.Quantity, optional
            Unit of measurement for the data, by default "uV".

        Raises
        ------
        ValueError
            If unit is not valid.
        """
        # check the unit and convert to uV if required
        if not self._unit.check("[voltage]"):
            raise ValueError("unit must represent voltage.")
        microvolts = pint.Quantity("uV")
        magnitude = self._unit.to(microvolts).magnitude
        self *= magnitude
        self._unit = microvolts

        # check the side
        valid_sides = ["left", "right", "bilateral"]
        if (
            not isinstance(side, (str, Literal["left", "right", "bilateral"]))
            or side not in valid_sides
        ):
            raise ValueError(f"side must be any of: {valid_sides}")

        # check the muscle name
        if not isinstance(muscle_name, str):
            raise ValueError("muscle_name must be a str.")

        # build the object
        column = f"{side}_{muscle_name}"
        super().__init__(data, index, unit, column=column)
        self._side = side
        self._muscle_name = muscle_name

    @property
    def side(self):
        """
        Get the side of the body.

        Returns
        -------
        {'left', 'right', 'bilateral'}
            The side of the body.
        """
        return self._side

    @property
    def muscle_name(self):
        """
        Get the name of the muscle.

        Returns
        -------
        str
            The name of the muscle.
        """
        return self._muscle_name

    def copy(self):
        """
        Return a deep copy of the EMGSignal.

        Returns
        -------
        EMGSignal
            A new EMGSignal object with the same data, index, muscle_name, side, and unit.
        """
        return EMGSignal(
            data=self._data.copy(),
            index=self.index,
            muscle_name=self.muscle_name,
            side=self.side,
            unit=self.unit,
        )
