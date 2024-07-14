"""metabolic module containing VO2 predicting equations"""

#! IMPORTS


from math import pi as PI

from scipy.optimize import least_squares


#! CLASSES


class CardioProduct:
    """base class for cardio products."""

    def _validate_input(self, obj: object, lbl: str):
        """private method used to validate the inputs"""
        if not isinstance(obj, (int, float)):
            raise ValueError(f"{lbl} must be an int or float.")
        if obj < 0:
            raise ValueError(f"{lbl} must be an >= 0.")

    def calculate_torque(self, power: float | int, cadence: float | int):
        """
        calculate the torque output from power and cadence

        Parameters
        ----------
        power: float | int
            the power output in W

        cadence: float | int
            the cadence in rpm

        Returns
        -------
        torque: float
            return the calculated torque in Nm.
        """
        self._validate_input(power, "power")
        self._validate_input(cadence, "cadence")
        return power / (cadence / 60 * 2 * PI)

    def calculate_cadence(self, power: float | int, torque: float | int):
        """
        calculate the cadence output from power and cadence

        Parameters
        ----------
        power: float | int
            the power output in W

        torque: float | int
            the torque in Nm

        Returns
        -------
        torque: float
            return the calculated cadence in rpm.
        """
        self._validate_input(power, "power")
        self._validate_input(torque, "torque")
        return torque / power / (2 * PI) * 60

    def calculate_power(self, torque: float | int, cadence: float | int):
        """
        calculate the power output from torque and cadence

        Parameters
        ----------
        torque: float | int
            the torque in Nm

        cadence: float | int
            the cadence in rpm

        Returns
        -------
        power: float
            return the calculated power in W.
        """
        self._validate_input(torque, "torque")
        self._validate_input(cadence, "cadence")
        return torque * (cadence / 60 * 2 * PI)


class BikeP(CardioProduct):
    """Bike P"""

    def __init__(self):
        super().__init__()

    def predict_vo2(self, power: float | int, bodyweight: float | int):
        """
        predict the exercise VO2 in ml/kg/min.

        Parameters
        ----------
        power: int | float
            the power output in W

        bodyweight: float | int
            the user bodyweight in kg

        Returns
        -------
        vo2: float
            return the predicted VO2 in ml/kg/min.
        """
        self._validate_input(power, "power")
        self._validate_input(bodyweight, "bodyweight")
        return 11.016 * power / bodyweight + 7

    def predict_bodyweight(self, vo2: float | int, power: float | int):
        """
        predict the bodyweight from vo2 and power

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        power: float | int
            the power output in W

        Returns
        -------
        bodyweight: float
            return the predicted bodyweight in kg.
        """
        self._validate_input(power, "power")
        self._validate_input(vo2, "vo2")
        return 11.016 * power / (vo2 - 7)

    def predict_power(self, vo2: float | int, bodyweight: float | int):
        """
        predict the power output from vo2 and bodyweight

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        bodyweight: float | int
            the user bodyweight in kg

        Returns
        -------
        bodyweight: float
            return the predicted bodyweight in kg.
        """
        self._validate_input(vo2, "vo2")
        self._validate_input(bodyqeight, "bodyweight")
        return (vo2 - 7) / 11.016 * bodyweight

    def torque_levels(self):
        """return the torque range allowed by the product in Nm"""
        return [
            5.0,
            7.0,
            9.0,
            11.0,
            13.0,
            15.0,
            17.0,
            19.0,
            21.0,
            23.0,
            25.0,
            27.0,
            29.0,
            31.0,
            33.0,
            35.0,
            37.0,
            39.0,
            41.0,
            42.0,
            43.0,
            44.0,
            45.0,
            46.0,
            47.0,
        ]


class BikeHP(BikeP):
    """BikeHP"""

    def __init__(self):
        super().__init__()

    def torque_levels(self):
        """return the torque range allowed by the product in Nm"""
        return [
            6.0,
            7.5,
            9.0,
            11.0,
            13.0,
            15.0,
            17.0,
            19.0,
            21.0,
            23.0,
            25.0,
            27.0,
            29.0,
            31.0,
            33.0,
            35.0,
            37.0,
            39.0,
            41.0,
            42.0,
            43.0,
            44.0,
            45.0,
            46.0,
            47.0,
        ]


class TechnogymBike(BikeP):
    """TechnogymBike"""

    def __init__(self):
        super().__init__()


class SynchroP(CardioProduct):
    """SynchroP"""

    def __init__(self):
        super().__init__()

    def estimate_incline(self, grade_level: int):
        """estimate the incline percentage according to the input level"""
        self._validate_input(grade_level, "grade_level")
        return 1.3 * grade_level + 5

    def estimate_grade_level(self, incline_perc: float):
        """
        estimate the grade level according to the provided
        incline percentage provided in the 0-100 range
        """
        self._validate_input(incline_perc, "incline_perc")
        return int(round((incline_perc - 5) / 1.3))

    def predict_vo2(
        self,
        power: float | int,
        bodyweight: float | int,
        grade_level: int = 8,
    ):
        """
        predict the exercise VO2 in ml/kg/min.

        Parameters
        ----------
        power: int | float
            the power output in W

        bodyweight: float | int
            the user bodyweight in kg

        grade_level: int (optional, default = 8)
            the grade level of the machine.

        Returns
        -------
        vo2: float
            return the predicted VO2 in ml/kg/min.
        """
        self._validate_input(power, "power")
        self._validate_input(bodyweight, "bodyweight")
        incline = self.estimate_incline(grade_level)
        return (
            1.13
            + 0.13 * power
            + 1090.28 / bodyweight
            - 0.2821 * incline
            + 0.0085 * incline**2
        )

    def predict_bodyweight(
        self,
        vo2: float | int,
        power: float | int,
        grade_level: int = 8,
    ):
        """
        predict the bodyweight from vo2 and power

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        power: float | int
            the power output in W

        grade_level: int (optional, default = 8)
            the grade level of the machine.

        Returns
        -------
        bodyweight: float
            return the predicted bodyweight in kg.
        """
        self._validate_input(power, "power")
        self._validate_input(vo2, "vo2")
        incline = self.estimate_incline(grade_level)
        return 1090.28 / (
            vo2 - 1.13 - 0.13 * power + 0.2821 * incline - 0.0085 * incline**2
        )

    def predict_power(
        self,
        vo2: float | int,
        bodyweight: float | int,
        grade_level: int = 8,
    ):
        """
        predict the power output from vo2 and bodyweight

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        bodyweight: float | int
            the user bodyweight in kg

        grade_level: int (optional, default = 8)
            the grade level of the machine.

        Returns
        -------
        bodyweight: float
            return the predicted bodyweight in kg.
        """
        self._validate_input(vo2, "vo2")
        self._validate_input(bodyweight, "bodyweight")
        incline = self.estimate_incline(grade_level)
        return (
            vo2 - 1.13 - 1090.28 / bodyweight + 0.2821 * incline - 0.0085 * incline**2
        ) / 0.13

    def predict_grade_level(
        self,
        vo2: float | int,
        bodyweight: float | int,
        power: int | float,
    ):
        """
        predict the power output from vo2 and bodyweight

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        bodyweight: float | int
            the user bodyweight in kg

        power: int | float
            the power output in W

        Returns
        -------
        power: float
            return the predicted power in W.
        """
        self._validate_input(power, "power")
        self._validate_input(bodyweight, "bodyweight")
        self._validate_input(vo2, "vo2")
        a = -0.0085
        b = 0.2821
        c = vo2 - 1.13 - 1090.28 / bodyweight
        d = b**2 - 4 * a * c
        if d < 0:
            d = 0
        x1 = (-b - d**2) / (2 * a)
        x2 = (-b + d**2) / (2 * a)
        return self.estimate_grade_level(max(x1, x2))

    def torque_levels(self):
        """return the torque range allowed by the product in Nm"""
        return [
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            24.0,
            26.0,
            28.0,
            30.0,
            33.0,
            36.0,
            39.0,
            42.0,
            45.0,
            48.0,
            51.0,
            54.0,
            57.0,
            60.0,
            63.0,
            66.0,
        ]


class SynchroHP(SynchroP):
    """SynchroHP"""

    def __init__(self):
        super().__init__()

    def torque_levels(self):
        """return the torque range allowed by the product in Nm"""
        return [
            6.0,
            7.5,
            9.0,
            11.0,
            13.0,
            15.0,
            17.0,
            19.0,
            21.0,
            23.0,
            25.0,
            27.0,
            29.0,
            31.0,
            33.0,
            35.0,
            37.0,
            39.0,
            41.0,
            43.0,
            45.0,
            47.0,
            49.0,
            51.0,
            53.0,
        ]


class TechnogymElliptical(CardioProduct):
    """Technogym Elliptical"""

    def __init__(self):
        super().__init__()

    def predict_vo2(
        self,
        power: float | int,
        bodyweight: float | int,
        cadence: float | int,
    ):
        """
        predict the exercise VO2 in ml/kg/min.

        Parameters
        ----------
        power: int | float
            the power output in W

        bodyweight: float | int
            the user bodyweight in kg

        cadence: float | int
            the cadence in spm

        Returns
        -------
        vo2: float
            return the predicted VO2 in ml/kg/min.
        """
        self._validate_input(power, "power")
        self._validate_input(bodyweight, "bodyweight")
        self._validate_input(cadence, "cadence")
        return (
            19.517629
            - 0.005871 * bodyweight
            + 3.358767 * power / bodyweight
            + 1.562194 * (power / bodyweight) ** 2
            - 23697.453729 / cadence**2
        )

    def predict_bodyweight(
        self,
        vo2: float | int,
        power: float | int,
        cadence: float | int,
    ):
        """
        predict the bodyweight from vo2 and power

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        power: float | int
            the power output in W

        cadence: float | int
            the cadence in spm

        Returns
        -------
        bodyweight: float
            return the predicted bodyweight in kg.
        """
        self._validate_input(power, "power")
        self._validate_input(vo2, "vo2")
        self._validate_input(cadence, "cadence")

        def fun(x):
            return (
                19.517629
                - 0.005871 * x
                + 3.358767 * power / x
                + 1.562194 * (power / x) ** 2
                - 23697.453729 / cadence**2
                - vo2
            )

        return float(least_squares(fun, 70))

    def predict_power(
        self,
        vo2: float | int,
        bodyweight: float | int,
        cadence: float | int,
    ):
        """
        predict the power output from vo2 and bodyweight

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        bodyweight: float | int
            the user bodyweight in kg

        cadence: float | int
            the cadence in spm

        Returns
        -------
        bodyweight: float
            return the predicted bodyweight in kg.
        """
        self._validate_input(vo2, "vo2")
        self._validate_input(bodyweight, "bodyweight")
        self._validate_input(cadence, "cadence")
        a = 1.562194
        b = 3.358767
        c = 19.517629 - 0.005871 * bodyweight - 23697.453729 / cadence**2 - vo2
        d = b**2 - 4 * a * c
        if d < 0:
            d = 0
        x1 = (-b - d**0.5) / (2 * a)
        x2 = (-b + d**0.5) / (2 * a)
        return float(bodyweight * max(x1, x2))

    def predict_cadence(
        self,
        vo2: float | int,
        bodyweight: float | int,
        power: int | float,
    ):
        """
        predict the cadence in spm from vo2 and bodyweight and power

        Parameters
        ----------
        vo2: float | int
            the vo2 in ml/kg/min

        bodyweight: float | int
            the user bodyweight in kg

        power: int | float
            the power output in W

        Returns
        -------
        cadence: float
            return the predicted cadence in spm.
        """
        self._validate_input(power, "power")
        self._validate_input(bodyweight, "bodyweight")
        self._validate_input(vo2, "vo2")
        den = vo2 - (
            19.517629
            - 0.005871 * bodyweight
            + 3.358767 * power / bodyweight
            + 1.562194 * (power / bodyweight) ** 2
        )
        num = -23697.453729
        eps = 1e-15
        return float(((num + eps) / (den + eps)) ** 0.5)

    def torque_levels(self):
        """return the torque range allowed by the product in Nm"""
        return [
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            24.0,
            26.0,
            28.0,
            30.0,
            33.0,
            36.0,
            39.0,
            42.0,
            45.0,
            48.0,
            51.0,
            54.0,
            57.0,
            60.0,
            63.0,
            66.0,
        ]
