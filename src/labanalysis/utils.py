"""
utils

module containing several utilities that can be used for multiple purposes.

Classes
-------
Participant
    an instance defining the general parameters of one subject during a test.

Functions
---------
magnitude
    get the order of magnitude of a numeric scalar value according to the
    specified base.

get_files
    get the full path of the files contained within the provided folder
    (and optionally subfolders) having the provided extension.

split_data
    get the indices randomly separating the input data into subsets according
    to the given proportions.
"""


#! IMPORTS


import os
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "Participant",
    "magnitude",
    "get_files",
    "split_data",
]


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


#! FUNCTIONS


def magnitude(
    value: int | float,
    base: int | float = 10,
):
    """
    return the order in the given base of the value

    Parameters
    ----------
        value: int | float
            the value to be checked

        base:int | float=10
            the base to be used to define the order of the number

    Returns
    -------
        mag float
            the number required to elevate the base to get the value
    """
    if value == 0 or base == 0:
        return int(0)
    else:
        val = np.log(abs(value)) / np.log(base)
        if val < 0:
            return -int(np.ceil(-val))
        return int(np.ceil(val))


def get_files(
    path: str,
    extension: str = "",
    check_subfolders: bool = False,
):
    """
    list all the files having the required extension in the provided folder
    and its subfolders (if required).

    Parameters
    ----------
        path: str
            a directory where to look for the files.

        extension: str
            a str object defining the ending of the files that have to be
            listed.

        check_subfolders: bool
            if True, also the subfolders found in path are searched,
            otherwise only path is checked.

    Returns
    -------
        files: list
            a list containing the full_path to all the files corresponding
            to the input criteria.
    """

    # output storer
    out = []

    # surf the path by the os. walk function
    for root, _, files in os.walk(path):
        for obj in files:
            if obj[-len(extension) :] == extension:
                out += [os.path.join(root, obj)]

        # handle the subfolders
        if not check_subfolders:
            break

    # return the output
    return out


def split_data(
    data: np.ndarray[Any, np.dtype[np.float_]],
    proportion: dict[str, float],
    groups: int,
):
    """
    get the indices randomly separating the input data into subsets according
    to the given proportions.

    Note
    ----
    the input array is firstly divided into quantiles according to the groups
    argument. Then the indices are randomly drawn from each subset according
    to the entered proportions. This ensures that the resulting groups
    will mimic the same distribution of the input data.

    Parameters
    ----------
    data : np.ndarray[Any, np.dtype[np.float_]]
        a 1D input array

    proportion : dict[str, float]
        a dict where each key contains the proportion of the total samples
        to be given. The proportion must be a value within the (0, 1] range
        and the sum of all entered proportions must be 1.

    groups : int
        the number of quantilic groups to be used.

    Returns
    -------
    splits: dict[str, np.ndarray[Any, np.dtype[np.int_]]]
        a dict with the same keys of proportion, which contains the
        corresponding indices.
    """

    # get the grouped data by quantiles
    nsamp = len(data)
    if groups <= 1:
        grps = [np.arange(nsamp)]
    else:
        qnts = np.quantile(data, np.linspace(0, 1, groups + 1)[1:])
        grps = np.digitize(data, qnts, right=True)
        idxs = np.arange(nsamp)
        grps = [idxs[grps == i] for i in np.arange(groups)]

    # split each group
    dss = {i: [] for i in proportion.keys()}
    for grp in grps:
        arr = np.random.permutation(grp)
        nsamp = len(arr)
        for i, k in enumerate(list(dss.keys())):
            if i < len(proportion) - 1:
                n = int(np.round(nsamp * proportion[k]))
            else:
                n = len(arr)
            dss[k] += [arr[:n]]
            arr = arr[n:]

    # aggregate
    return {i: np.concatenate(v) for i, v in dss.items()}
