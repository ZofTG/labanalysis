"""
base test module containing classes and functions used to perform lab tests.

Classes
-------
Participant
    an instance defining the general parameters of one subject during a test.

LabTest
    an abstract class containing the attributes:
        - results_table
        - summary_table
        _ summary_plot
"""

#! IMPORTS


from abc import abstractmethod
from datetime import datetime, date
import pandas as pd

__all__ = ["LabTest", "Participant"]


#! CLASSES


class Participant:
    """
    class containing all the data relevant to a participant.

    Parameters
    ----------
    surname: str | None = None
        the participant surname

    name: str | None = None
        the participant name

    gender: str | None = None
        the participant gender

    height: int | float | None = None
        the participant height

    weight: int | float | None = None
        the participant weight

    age: int | float | None = None
        the participant age

    birthdate: date | None = None
        the participant birth data

    recordingdate: date | None = None
        the the test recording date
    """

    # class variables
    _name = None
    _surname = None
    _gender = None
    _height = None
    _weight = None
    _birthdate = None
    _recordingdate = date  # type:ignore
    _units = {
        "fullname": "",
        "surname": "",
        "name": "",
        "gender": "",
        "height": "m",
        "weight": "kg",
        "bmi": "kg/m^2",
        "birthdate": "",
        "age": "years",
        "hrmax": "bpm",
        "recordingdate": "",
    }

    def __init__(
        self,
        surname: str | None = None,
        name: str | None = None,
        gender: str | None = None,
        height: int | float | None = None,
        weight: int | float | None = None,
        age: int | float | None = None,
        birthdate: date | None = None,
        recordingdate: date = datetime.now().date,  # type: ignore
    ):
        self.set_surname(surname)
        self.set_name(name)
        self.set_gender(gender)
        self.set_height((height / 100 if height is not None else height))
        self.set_weight(weight)
        self.set_age(age)
        self.set_birthdate(birthdate)
        self.set_recordingdate(recordingdate)

    def set_recordingdate(
        self,
        recordingdate: date | None,
    ):
        """
        set the test recording date.

        Parameters
        ----------
        recording_date: datetime.date | None
            the test recording date.
        """
        if recordingdate is not None:
            txt = "'recordingdate' must be a datetime.date or datetime.datetime."
            assert isinstance(recordingdate, (datetime, date)), txt
            if isinstance(recordingdate, datetime):
                self._recordingdate = recordingdate.date()
            else:
                self._recordingdate = recordingdate
        else:
            self._recordingdate = recordingdate

    def set_surname(
        self,
        surname: str | None,
    ):
        """
        set the participant surname.

        Parameters
        ----------
        surname: str | None,
            the surname of the participant.
        """
        if surname is not None:
            assert isinstance(surname, str), "'surname' must be a string."
        self._surname = surname

    def set_name(
        self,
        name: str | None,
    ):
        """
        set the participant name.

        Parameters
        ----------
        name: str | None
            the name of the participant.
        """
        if name is not None:
            assert isinstance(name, str), "'name' must be a string."
        self._name = name

    def set_gender(
        self,
        gender: str | None,
    ):
        """
        set the participant gender.

        Parameters
        ----------
        gender: str | None
            the gender of the participant.
        """
        if gender is not None:
            assert isinstance(gender, str), "'gender' must be a string."
        self._gender = gender

    def set_height(
        self,
        height: int | float | None,
    ):
        """
        set the participant height in meters.

        Parameters
        ----------
        height: int | float | None
            the height of the participant.
        """
        if height is not None:
            txt = "'height' must be a float or int."
            assert isinstance(height, (int, float)), txt
        self._height = height

    def set_weight(
        self,
        weight: int | float | None,
    ):
        """
        set the participant weight in kg.

        Parameters
        ----------
        weight: int | float | None
            the weight of the participant.
        """
        if weight is not None:
            txt = "'weight' must be a float or int."
            assert isinstance(weight, (int, float)), txt
        self._weight = weight

    def set_age(
        self,
        age: int | float | None,
    ):
        """
        set the participant age in years.


        Parameters
        ----------
        age: int | float | None,
            the age of the participant.
        """
        if age is not None:
            txt = "'age' must be a float or int."
            assert isinstance(age, (int, float)), txt
        self._age = age

    def set_birthdate(
        self,
        birthdate: date | None,
    ):
        """
        set the participant birth_date.

        Parameters
        ----------
        birth_date: datetime.date | None
            the birth date of the participant.
        """
        if birthdate is not None:
            txt = "'birth_date' must be a datetime.date or datetime.datetime."
            assert isinstance(birthdate, (datetime, date)), txt
            if isinstance(birthdate, datetime):
                self._birthdate = birthdate.date()
            else:
                self._birthdate = birthdate
        else:
            self._birthdate = birthdate

    @property
    def surname(self):
        """get the participant surname"""
        return self._surname

    @property
    def name(self):
        """get the participant name"""
        return self._name

    @property
    def gender(self):
        """get the participant gender"""
        return self._gender

    @property
    def height(self):
        """get the participant height in meter"""
        return self._height

    @property
    def weight(self):
        """get the participant weight in kg"""
        return self._weight

    @property
    def birthdate(self):
        """get the participant birth date"""
        return self._birthdate

    @property
    def recordingdate(self):
        """get the test recording date"""
        return self._recordingdate

    @property
    def bmi(self):
        """get the participant BMI in kg/m^2"""
        if self.height is None or self.weight is None:
            return None
        return self.weight / (self.height**2)

    @property
    def fullname(self):
        """
        get the participant full name.
        """
        return f"{self.surname} {self.name}"

    @property
    def age(self):
        """
        get the age of the participant in years
        """
        if self._age is not None:
            return self._age
        if self._birthdate is not None:
            return int((self._recordingdate - self._birthdate).days // 365)  # type: ignore
        return None

    @property
    def hrmax(self):
        """
        get the maximum theoretical heart rate according to Gellish.

        References
        ----------
        Gellish RL, Goslin BR, Olson RE, McDonald A, Russi GD, Moudgil VK.
            Longitudinal modeling of the relationship between age and maximal
            heart rate.
            Med Sci Sports Exerc. 2007;39(5):822-9.
            doi: 10.1097/mss.0b013e31803349c6.
        """
        if self.age is None:
            return None
        return 207 - 0.7 * self.age

    @property
    def units(self):
        """return the unit of measurement of the stored data."""
        return self._units

    def copy(self):
        """return a copy of the object."""
        return Participant(**{i: getattr(self, i) for i in self.units.keys()})

    @property
    def dict(self):
        """return a dict representation of self"""
        out = {}
        for i, v in self.units.items():
            out[i + ((" [" + v + "]") if v != "" else "")] = getattr(self, i)
        return out

    @property
    def series(self):
        """return a pandas.Series representation of self"""
        vals = [(i, v) for i, v in self.units.items()]
        vals = pd.MultiIndex.from_tuples(vals)
        return pd.Series(list(self.dict.values()), index=vals)

    @property
    def dataframe(self):
        """return a pandas.DataFrame representation of self"""
        return pd.DataFrame(self.series).T

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.dataframe.__str__()


class LabTest:
    """
    abstract class defining the general methods expected from a test

    Attributes
    -------
    summary_plot
        return a matplotlib figure highlighting the test' results.

    summary_table
        return a pandas dataframe containing the summary data.
    """

    @property
    @abstractmethod
    def summary_plot(self):
        return NotImplementedError

    @property
    @abstractmethod
    def summary_table(self):
        return NotImplementedError

    @property
    @abstractmethod
    def results_table(self):
        return NotImplementedError
