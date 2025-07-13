"""
base test module containing classes and functions used to perform lab tests.
"""

#! IMPORTS

from warnings import warn
import pickle
from datetime import date, datetime
from os.path import exists
from typing import Protocol, runtime_checkable

import pandas as pd
from scipy.stats.distributions import norm as normal_distribution

from ..messages import askyesnocancel

__all__ = ["TestProtocol", "TestBattery", "Participant"]


#! CLASSES


class Participant:
    """
    Represents a participant in a lab test.

    Stores participant data such as name, gender, height, weight, age, birthdate, and test recording date.
    Provides methods and properties to access and compute participant metrics.

    Parameters
    ----------
    surname : str, optional
        The participant's surname.
    name : str, optional
        The participant's name.
    gender : str, optional
        The participant's gender.
    height : int or float, optional
        The participant's height in centimeters. Will be converted to meters.
    weight : int or float, optional
        The participant's weight in kilograms.
    age : int or float, optional
        The participant's age in years.
    birthdate : datetime.date, optional
        The participant's birth date.
    recordingdate : datetime.date, optional
        The date when the test was recorded. Defaults to the current date.

    Attributes
    ----------
    units : dict
        Units of measurement for each attribute.

    Methods
    -------
    set_recordingdate(recordingdate)
        Set the test recording date.
    set_surname(surname)
        Set the participant's surname.
    set_name(name)
        Set the participant's name.
    set_gender(gender)
        Set the participant's gender.
    set_height(height)
        Set the participant's height in meters.
    set_weight(weight)
        Set the participant's weight in kilograms.
    set_age(age)
        Set the participant's age in years.
    set_birthdate(birthdate)
        Set the participant's birth date.
    copy()
        Return a copy of the object.

    Properties
    ----------
    surname : str or None
        The participant's surname.
    name : str or None
        The participant's name.
    gender : str or None
        The participant's gender.
    height : float or None
        The participant's height in meters.
    weight : float or None
        The participant's weight in kilograms.
    birthdate : datetime.date or None
        The participant's birth date.
    recordingdate : datetime.date or None
        The test recording date.
    bmi : float or None
        The participant's BMI in kg/m^2.
    fullname : str
        The participant's full name.
    age : int or None
        The participant's age in years.
    hrmax : float or None
        The maximum theoretical heart rate according to Gellish.
    units : dict
        Units of measurement for each attribute.
    dict : dict
        Dictionary representation of the participant's data.
    series : pandas.Series
        pandas.Series representation of the participant's data.
    dataframe : pandas.DataFrame
        pandas.DataFrame representation of the participant's data.
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
        """
        Initializes a Participant object.
        """
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
        Sets the test recording date.

        Parameters
        ----------
        recordingdate : datetime.date or datetime.datetime or None
            The test recording date.

        Returns
        -------
        None
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
        Sets the participant's surname.

        Parameters
        ----------
        surname : str or None
            The surname of the participant.

        Returns
        -------
        None
        """
        if surname is not None:
            assert isinstance(surname, str), "'surname' must be a string."
        self._surname = surname

    def set_name(
        self,
        name: str | None,
    ):
        """
        Sets the participant's name.

        Parameters
        ----------
        name : str or None
            The name of the participant.

        Returns
        -------
        None
        """
        if name is not None:
            assert isinstance(name, str), "'name' must be a string."
        self._name = name

    def set_gender(
        self,
        gender: str | None,
    ):
        """
        Sets the participant's gender.

        Parameters
        ----------
        gender : str or None
            The gender of the participant.

        Returns
        -------
        None
        """
        if gender is not None:
            assert isinstance(gender, str), "'gender' must be a string."
        self._gender = gender

    def set_height(
        self,
        height: int | float | None,
    ):
        """
        Sets the participant's height in meters.

        Parameters
        ----------
        height : int or float or None
            The height of the participant.

        Returns
        -------
        None
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
        Sets the participant's weight in kg.

        Parameters
        ----------
        weight : int or float or None
            The weight of the participant.

        Returns
        -------
        None
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
        Sets the participant's age in years.

        Parameters
        ----------
        age : int or float or None
            The age of the participant.

        Returns
        -------
        None
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
        Sets the participant's birth date.

        Parameters
        ----------
        birthdate : datetime.date or None
            The birth date of the participant.

        Returns
        -------
        None
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
        """
        Gets the participant surname.

        Returns
        -------
        str or None
            The participant's surname.
        """
        return self._surname

    @property
    def name(self):
        """
        Gets the participant name.

        Returns
        -------
        str or None
            The participant's name.
        """
        return self._name

    @property
    def gender(self):
        """
        Gets the participant gender.

        Returns
        -------
        str or None
            The participant's gender.
        """
        return self._gender

    @property
    def height(self):
        """
        Gets the participant height in meters.

        Returns
        -------
        float or None
            The participant's height in meters.
        """
        return self._height

    @property
    def weight(self):
        """
        Gets the participant weight in kg.

        Returns
        -------
        float or None
            The participant's weight in kg.
        """
        return self._weight

    @property
    def birthdate(self):
        """
        Gets the participant birth date.

        Returns
        -------
        datetime.date or None
            The participant's birth date.
        """
        return self._birthdate

    @property
    def recordingdate(self):
        """
        Gets the test recording date.

        Returns
        -------
        datetime.date or None
            The test recording date.
        """
        return self._recordingdate

    @property
    def bmi(self):
        """
        Gets the participant BMI in kg/m^2.

        Returns
        -------
        float or None
            The participant's BMI in kg/m^2. Returns None if height or weight is None.
        """
        if self.height is None or self.weight is None:
            return None
        return self.weight / (self.height**2)

    @property
    def fullname(self):
        """
        Gets the participant full name.

        Returns
        -------
        str
            The participant's full name.
        """
        return f"{self.surname} {self.name}"

    @property
    def age(self):
        """
        Gets the age of the participant in years.

        Returns
        -------
        int or None
            The age of the participant in years. Returns None if age or birthdate is None.
        """
        if self._age is not None:
            return self._age
        if self._birthdate is not None:
            return int((self._recordingdate - self._birthdate).days // 365)  # type: ignore
        return None

    @property
    def hrmax(self):
        """
        Gets the maximum theoretical heart rate according to Gellish.

        Returns
        -------
        float or None
            The maximum theoretical heart rate. Returns None if age is None.

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
        """
        Returns the unit of measurement of the stored data.

        Returns
        -------
        dict
            A dictionary containing the units of measurement for each attribute.
        """
        return self._units

    def copy(self):
        """
        Returns a copy of the object.

        Returns
        -------
        Participant
            A copy of the Participant object.
        """
        return Participant(**{i: getattr(self, i) for i in self.units.keys()})

    @property
    def dict(self):
        """
        Returns a dict representation of self

        Returns
        -------
        dict
            A dictionary representation of the Participant object.
        """
        out = {}
        for i, v in self.units.items():
            out[i + ((" [" + v + "]") if v != "" else "")] = getattr(self, i)
        return out

    @property
    def series(self):
        """
        Returns a pandas.Series representation of self

        Returns
        -------
        pandas.Series
            A pandas.Series representation of the Participant object.
        """
        vals = [(i, v) for i, v in self.units.items()]
        vals = pd.MultiIndex.from_tuples(vals)
        return pd.Series(list(self.dict.values()), index=vals)

    @property
    def dataframe(self):
        """
        Returns a pandas.DataFrame representation of self

        Returns
        -------
        pandas.DataFrame
            A pandas.DataFrame representation of the Participant object.
        """
        return pd.DataFrame(self.series).T

    def __repr__(self):
        """
        Returns a string representation of the Participant object.
        """
        return self.__str__()

    def __str__(self):
        """
        Returns a string representation of the Participant object.
        """
        return self.dataframe.__str__()


@runtime_checkable
class TestProtocol:
    """
    Protocol for lab test classes.

    Defines the required interface for test protocol implementations, including participant data,
    normative data, and methods for saving/loading and summarizing results.

    Parameters
    ----------
    participant : Participant
        The participant associated with the test.
    normative_data_path : str, optional
        Path to a CSV file containing normative data.

    Attributes
    ----------
    _normative_data_path : str
        Path to the CSV file with normative data.
    _participant : Participant
        The participant object.

    Properties
    ----------
    participant : Participant
        The participant associated with the test.
    name : str
        The name of the test (class name).
    normative_data_path : str
        Path to the CSV file with normative data.
    normative_values : pandas.DataFrame
        Normative values loaded from the CSV file.

    Methods
    -------
    save(file_path: str)
        Save the test object to a file.
    load(file_path: str)
        Load a test object from a file.
    result_tables() -> dict[str, pd.DataFrame]
        Abstract method. Return a summary of the test results as a dictionary of pandas DataFrames.
    processing_pipeline
        Return the default processing pipeline for this test.
    raw_data_table()
        Return a table containing the raw data (optional, may raise NotImplementedError).
    raw_data_figure()
        Return a figure displaying the raw data (optional, may raise NotImplementedError).
    """

    _normative_data_path: str
    _participant: Participant

    def set_normative_data_path(self, path: str):
        if not isinstance(path, str) or not exists(path) or not path.endswith(".csv"):
            warn(
                "'normative_data_path' is not valid."
                + " If provided, it should be a "
                + " .csv table containing normative data."
                + " Not having valid normative references might affect"
                + " the implementation of specific test reports."
            )
            self._normative_data_path = ""
        self._normative_data_path = path

    def set_participant(self, participant: Participant):
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant instance.")
        self._participant = participant

    @property
    def participant(self):
        return self._participant

    @property
    def name(self):
        """
        Returns the test name.

        Returns
        -------
        str
            The name of the test.
        """
        return type(self).__name__

    @property
    def normative_data_path(self):
        """
        Returns the path to the CSV file containing normative data.

        Returns
        -------
        str
            The path to the CSV file containing normative data.
        """
        return self._normative_data_path

    @property
    def normative_values(self):
        """
        Returns the normative values loaded from the CSV file.

        Returns
        -------
        pandas.DataFrame
            The normative values loaded from the CSV file.
        """
        if self.normative_data_path == "":
            return pd.DataFrame()

        data = pd.read_csv(self.normative_data_path)
        for row, line in data.iterrows():
            row = int(row)  # type: ignore
            p01, p20, p40, p60, p80, p99 = normal_distribution.ppf(
                [0.01, 0.2, 0.4, 0.6, 0.8, 0.99],
                loc=line.mean,
                scale=line.std,
            )
            data.loc[row, "p1"] = p01
            data.loc[row, "p20"] = p20
            data.loc[row, "p40"] = p40
            data.loc[row, "p60"] = p60
            data.loc[row, "p80"] = p80
            data.loc[row, "p99"] = p99
        return data

    def save(self, file_path: str):
        """
        Save the test object to a file.

        Parameters
        ----------
        file_path : str
            Path where to save the file. The file extension should match the test name.
            If not, the appropriate extension is appended.

        Returns
        -------
        None
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + self.__class__.__name__.lower()
        if not file_path.endswith(extension):
            file_path += extension
        overwrite = False
        while exists(file_path) and not overwrite:
            overwrite = askyesnocancel(
                title="File already exists",
                message="the provided file_path already exist. Overwrite?",
            )
            if not overwrite:
                file_path = file_path[: len(extension)] + "_" + extension
        if not exists(file_path) or overwrite:
            with open(file_path, "wb") as buf:
                pickle.dump(self, buf)

    @classmethod
    def load(cls, file_path: str):
        """
        Load a test object from a file.

        Parameters
        ----------
        file_path : str
            Path to the file to load. The file extension must match the test name.

        Returns
        -------
        TestProtocol
            The loaded test object.

        Raises
        ------
        ValueError
            If file_path is not a string or does not have the correct extension.
        RuntimeError
            If loading fails.
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + cls.__name__.lower()
        if not file_path.endswith(extension):
            raise ValueError(f"'file_path' must have {extension}.")
        try:
            with open(file_path, "rb") as buf:
                return pickle.load(buf)
        except Exception:
            raise RuntimeError(f"an error occurred importing {file_path}.")

    #! MANDATORY METHODS TO BE IMPLEMENTED

    def results(self) -> dict[str, pd.DataFrame]:
        """
        Abstract method to return a summary of the test results.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Dictionary of summary tables for the test results.
        """
        ...


@runtime_checkable
class TestBattery(Protocol):
    """
    Protocol for a collection of lab tests.

    Allows grouping multiple tests and performing operations on them as a single entity.

    Parameters
    ----------
    *tests : TestProtocol
        TestProtocol objects to include in the battery.

    Attributes
    ----------
    _tests : list[TestProtocol]
        List of tests in the battery.

    Properties
    ----------
    name : str
        The name of the test battery (class name).
    tests : list[TestProtocol]
        List of tests in the battery.

    Methods
    -------
    save(file_path: str)
        Save the test battery to a file.
    load(file_path: str)
        Load a test battery from a file.
    result_table() -> dict[str, pd.DataFrame]
        Abstract method. Return a summary of the test results as a dictionary of pandas DataFrames.
    """

    _tests: list[TestProtocol]

    def __init__(self, *tests: TestProtocol):
        """
        Initializes a TestBattery object.

        Parameters
        ----------
        *tests : TestProtocol
            The tests to be included in the battery.

        Raises
        ------
        ValueError
            If any of the input objects is not a TestProtocol subclass.

        Returns
        -------
        None
        """
        # check the inputs
        msg = "'tests' must be LabTest subclassed objects."
        for test in tests:
            if not isinstance(test, TestProtocol):
                raise ValueError(msg)

        # store the tests
        self._tests = list(tests)

    @property
    def name(self):
        """
        Returns the test name.

        Returns
        -------
        str
            The name of the test.
        """
        return type(self).__name__

    @property
    def tests(self):
        """
        Returns the list of tests being part of the test battery.

        Returns
        -------
        list[TestProtocol]
            The list of tests being part of the test battery.
        """
        return self._tests

    def save(self, file_path: str) -> None:
        """
        Save the test battery to a file.

        Parameters
        ----------
        file_path : str
            Path where to save the file. The file extension should match the test battery name.
            If not, the appropriate extension is appended.

        Returns
        -------
        None
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + self.__class__.__name__.lower()
        if not file_path.endswith(extension):
            file_path += extension
        overwrite = False
        while exists(file_path) and not overwrite:
            overwrite = askyesnocancel(
                title="File already exists",
                message="the provided file_path already exist. Overwrite?",
            )
            if not overwrite:
                file_path = file_path[: len(extension)] + "_" + extension
        if not exists(file_path) or overwrite:
            with open(file_path, "wb") as buf:
                pickle.dump(self, buf)

    @classmethod
    def load(cls, file_path: str):
        """
        Load a test battery from a file.

        Parameters
        ----------
        file_path : str
            Path to the file to load. The file extension must match the test battery name.

        Returns
        -------
        TestBattery
            The loaded test battery object.

        Raises
        ------
        ValueError
            If file_path is not a string or does not have the correct extension.
        RuntimeError
            If loading fails.
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + cls.__name__.lower()
        if not file_path.endswith(extension):
            raise ValueError(f"'file_path' must have {extension}.")
        try:
            with open(file_path, "rb") as buf:
                return pickle.load(buf)
        except Exception:
            raise RuntimeError(f"an error occurred importing {file_path}.")

    #! MANDATORY METHODS TO BE IMPLEMENTED

    def result_table(self) -> dict[str, pd.DataFrame]:
        """
        Abstract method to return a summary of the test results as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A summary of the test results.
        """
        ...
