"""products module"""

#! IMPORTS


import copy
from os.path import exists
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from labanalysis import constants


#! CLASSES


class BiostrengthProduct:
    """Product class object"""

    # * class variables

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 0
    _camme_ratio: float = 1
    _lever_number: int = 1
    _lever_radius_m: float = 0.054
    _rom_correction_coefs: list[float] = [0, 0, 0]
    _rm1_coefs: list[float] = [1, 0]
    _torque_load_coefs: list[float] = [1, 0]

    _position_motor_rad: NDArray[np.floating[Any]]
    _load_motor_nm: NDArray[np.floating[Any]]
    _time_s: NDArray[np.floating[Any]]

    # * attributes

    @property
    def time_s(self):
        """return the time of each sample"""
        return self._time_s[1:-1].astype(float)

    @property
    def position_motor_rad(self):
        """return the raw postition in radians"""
        return self._position_motor_rad[1:-1].astype(float)

    @property
    def pulley_radius_m(self):
        """pulley radius coefficient in m for each time sample"""
        return np.tile(self._pulley_radius_m, len(self.time_s))

    @property
    def lever_weight_kgf(self):
        """lever weight coefficient in kgf for each time sample"""
        return np.tile(self._lever_weight_kgf, len(self.time_s))

    @property
    def camme_ratio(self):
        """camme ratio coefficient for each time sample"""
        return np.tile(self._camme_ratio, len(self.time_s))

    @property
    def spring_correction(self):
        """spring correction coefficient for each time sample"""
        return np.tile(self._spring_correction, len(self.time_s))

    @property
    def load_motor_nm(self):
        """return the motor load in Nm"""
        return self._load_motor_nm[1:-1].astype(float)

    @property
    def lever_number(self):
        """number of levers"""
        return np.tile(self._lever_number, len(self.time_s))

    @property
    def rom_correction_coefs(self):
        """rom correction coefficients with higher order first"""
        return self._rom_correction_coefs

    @property
    def position_lever_deg(self):
        """return the calculated position of the lever in degrees"""
        out = self.position_lever_m / self.lever_radius_m * 180 / np.pi
        return out.astype(float)

    @property
    def lever_radius_m(self):
        """radius of the lever(s) in m for each sample"""
        return np.tile(self._lever_radius_m, len(self.time_s)).astype(float)

    @property
    def position_lever_m(self):
        """return the calculated position of the lever in meters"""
        return (self.position_motor_rad * self.lever_radius_m).astype(float)

    @property
    def load_lever_kgf(self):
        """return the calculated lever weight"""
        return (
            self.load_motor_nm
            / constants.G
            / self._pulley_radius_m
            * self.camme_ratio
            / self.spring_correction
            + self.lever_weight_kgf
        )

    @property
    def speed_motor_rads(self):
        """
        return the calculated speed at the motor level in rad for each sample
        """
        num = self._position_motor_rad[:-2] - self._position_motor_rad[2:]
        den = self._time_s[:-2] - self._time_s[2:]
        return (num / den).astype(float)

    @property
    def speed_lever_degs(self):
        """
        return the calculated speed at the lever level in deg/s for each sample
        """
        degs = self.speed_lever_ms / self._lever_radius_m * 180 / np.pi
        return degs.astype(float)

    @property
    def speed_lever_ms(self):
        """
        return the calculated speed at the lever level in m/s for each sample
        """
        num = self._position_motor_rad * self._lever_radius_m
        num = num[2:] - num[:-2]
        den = self._time_s[2:] - self._time_s[:-2]
        return (num / den).astype(float)

    @property
    def power_w(self):
        """return the calculated power"""
        return self.load_motor_nm * self.speed_motor_rads

    @property
    def rm1_coefs(self):
        """1RM coefficients with higher order first"""
        return self._rm1_coefs

    @property
    def correction_coefs(self):
        """
        return the correction coefficients that extract the
        load in kgf from the input Torque.
        """
        return self._torque_load_coefs

    @property
    def name(self):
        """the name of the product"""
        return type(self).__name__

    # * methods

    def copy(self):
        """make a copy of the object"""
        return copy.deepcopy(self)

    def as_dataframe(self):
        """return a summary table containing the resulting data"""
        out = {
            ("Time", "s"): self.time_s,
            ("Load", "kgf"): self.load_lever_kgf,
            # ("Motor Load", "Nm"): self.load_motor_nm,
            ("Position", "m"): self.position_lever_m,
            ("Position", "deg"): self.position_lever_deg,
            # ("Motor Position", "rad"): self.position_motor_rad,
            ("Speed", "m/s"): self.speed_lever_ms,
            ("Speed", "deg/s"): self.speed_lever_degs,
            # ("Motor Speed", "rad/s"): self.speed_motor_rads,
            ("Power", "W"): self.power_w,
        }
        return pd.DataFrame(out)

    def slice(self, start_time: float, stop_time: float):
        """
        slice _summary_

        _extended_summary_

        Parameters
        ----------
        start_time : float
            the start time of the slice

        stop_time : float
            the end time of the slice

        Returns
        -------
        sliced: Product
            a sliced version of the object.
        """
        idx = (self._time_s >= start_time) & (self._time_s <= stop_time)
        idx = np.where(idx)[0]

        obj = self.copy()
        obj._time_s = self._time_s[idx]
        obj._position_motor_rad = self._position_motor_rad[idx]
        obj._load_motor_nm = self._load_motor_nm[idx]

        return obj

    # * constructors

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        load_kgf: NDArray[np.floating],
    ):
        # check the entries
        try:
            self._time_s = np.array([time_s]).astype(float).flatten()
        except Exception as exc:
            raise ValueError(
                "time must be castable to a numpy array of floats"
            ) from exc
        try:
            self._position_motor_rad = (
                np.array([motor_position_rad]).astype(float).flatten()
            )
        except Exception as exc:
            raise ValueError(
                "motor_position_rad must be castable to a numpy array of floats"
            ) from exc
        try:
            self._load_motor_nm = (
                (load_kgf - self._lever_weight_kgf)
                * constants.G
                * self._pulley_radius_m
                * self._spring_correction
                / self._camme_ratio
            )

        except Exception as exc:
            raise ValueError(
                "motor_load_nm must be castable to a numpy array of floats"
            ) from exc

        # check the length of each element
        if (
            not len(self.time_s)
            == len(self.position_motor_rad)
            == len(self.load_motor_nm)
        ):
            msg = "time_s, motor_position_rad and motor_load_nm must all have "
            msg += "the same number of samples."
            raise ValueError(msg)

    @classmethod
    def from_txt_file(cls, file: str):
        """
        read raw data from file

        Parameters
        ----------
        file : str
            the path to the file
        """

        # check the inputs
        msg = "incorrect file."
        assert isinstance(file, str), msg
        assert exists(file), msg
        assert file.endswith(".txt") or file.endswith(".csv"), msg

        # get the data
        obj = pd.read_csv(file, sep="|")
        col = obj.columns[[0, 2, 5]]
        obj = obj[col].astype(str).map(lambda x: x.replace(",", "."))
        time, load, pos = obj.astype(float).values.T
        load = cls._torque_load_coefs[0] * load + cls._torque_load_coefs[1]
        load += cls._lever_weight_kgf

        # return
        return cls(time, pos, load)  # type: ignore


class ChestPress(BiostrengthProduct):
    """Chest Press class object"""

    _spring_correction: float = 1.15  # vecchia 1.35
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = -4.0
    _camme_ratio: float = 0.74
    _lever_number: int = 1
    _lever_radius_m: float = 0.054
    _rom_correction_coefs: list[float] = [
        -0.0000970270993668,
        0.0284363503605837,
        -0.1454105176656738,
    ]
    _rm1_coefs: list[float] = [0.963351, 2.845189]
    _torque_load_coefs: list[float] = [1, 0]  # torque correction coefficients

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class ShoulderPress(BiostrengthProduct):
    """Shoulder Press class object"""

    _spring_correction: float = 1  # vecchia 1.35
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = -1.2
    _camme_ratio: float = 0.794
    _lever_number: int = 1
    _lever_radius_m: float = 0.054
    _rom_correction_coefs: list[float] = [
        -0.0001308668672017,
        0.0242885910602534,
        -0.0911406828188467,
    ]
    _rm1_coefs: list[float] = [0.862141, 1.419287]
    _torque_load_coefs: list[float] = [1, 0]  # torque correction coefficients

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LowRow(BiostrengthProduct):
    """Low Row class object"""

    _spring_correction: float = 1  # vecchia 1.0
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 5.0
    _camme_ratio: float = 0.64
    _lever_number: int = 1
    _lever_radius_m: float = 0.054
    _rom_correction_coefs: list[float] = [
        0.0009021430405893,
        -0.0236810740758083,
        0.1162621888946583,
    ]
    _rm1_coefs: list[float] = [0.695723, 3.142428]
    _torque_load_coefs: list[float] = [1, 0]  # torque correction coefficients

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegPress(BiostrengthProduct):
    """Leg Press class object"""

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.08175
    _lever_weight_kgf: float = 9.0 + 0.17 * 85
    _camme_ratio: float = 1
    _lever_number: int = 1
    _lever_radius_m: float = 0.08175
    _rom_correction_coefs: list[float] = [
        -0.0000594298355666,
        0.0155680740573513,
        -0.0022758912872085,
    ]
    _rm1_coefs: list[float] = [0.65705, 9.17845]
    _torque_load_coefs: list[float] = [1.246048, -23.594744]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegExtension(BiostrengthProduct):
    """Leg Extension class object"""

    _spring_correction: float = 0.79
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 1  # ? TO BE CHECKED
    _camme_ratio: float = 0.738
    _lever_number: int = 1
    _lever_radius_m: float = 0.21  # ? TO BE CHECKED
    _rom_correction_coefs: list[float] = [
        0.1237962826137063,
        -0.0053627811034270,
        0.0003232899485875,
    ]
    _rm1_coefs: list[float] = [0.7351, 6]
    _torque_load_coefs: list[float] = [1.042277, 0.072454]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class AdjustablePulleyREV(BiostrengthProduct):
    """Adjustable Pulley REV class object"""

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 0.01
    _camme_ratio: float = 0.25
    _lever_number: int = 2
    _lever_radius_m: float = 0.054
    _rom_correction_coefs: list[float] = [0, 0, 0]
    _rm1_coefs: list[float] = [1, 0]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegPressREV(LegPress):
    """Leg Press REV class object"""

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegExtensionREV(LegExtension):
    """Leg Extension REV class object"""

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 0.875
    _camme_ratio: float = 0.689
    _lever_number: int = 1
    _lever_radius_m: float = 0.21
    _rom_correction_coefs: list[float] = [
        0.000201694,
        -0.030051020,
        0.03197279,
    ]
    _rm1_coefs: list[float] = [0.7351, 6]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


__all__ = [
    "PRODUCTS",
    "ChestPress",
    "ShoulderPress",
    "LowRow",
    "BiostrengthProduct",
    "LegPress",
    "LegExtension",
    "AdjustablePulleyREV",
    "LegPressREV",
    "LegExtensionREV",
]

PRODUCTS = {
    "CHEST PRESS": ChestPress,
    "SHOULDER PRESS": ShoulderPress,
    "LOW ROW": LowRow,
    "LEG PRESS": LegPress,
    "LEG EXTENSION": LegExtension,
    "ADJUSTABLE PULLEY REV": AdjustablePulleyREV,
    "LEG PRESS REV": LegPressREV,
    "LEG EXTENSION REV": LegExtensionREV,
}
