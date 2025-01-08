"""
base test module containing classes and functions used to perform lab tests.

Classes
-------
Participant
    an instance defining the general parameters of one subject during a test.

LabTest
    abstract class defining the general methods expected from a test

TestBattery
    class allowing the analysis of a set of tests
"""

#! IMPORTS


import pickle
from datetime import date, datetime
from os.path import exists
from typing import Protocol, runtime_checkable

import pandas as pd
import plotly.graph_objects as go

from .. import messages

__all__ = ["LabTest", "TestBattery", "Participant"]


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


@runtime_checkable
class LabTest(Protocol):
    """
    abstract class defining the general methods expected from a test

    Properties
    ----------
    name: str
        the name of the test

    Methods
    -------
    summary
        return a summary of the test results both as dictionary of plotly
        FigureWidget objects and as pandas dataframe

    results
        return the "raw" test results both as plotly FigureWidget and
        pandas dataframe

    save
        a method allowing the saving of the data in an appropriate format.

    load
        a class method to load a LabTest object saved in its own format.
    """

    @property
    def name(self):
        """return the test name"""
        return type(self).__name__

    def _make_summary_table(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ) -> pd.DataFrame: ...

    def _make_summary_plot(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ) -> dict[str, go.FigureWidget]: ...

    def get_intervals(
        self,
        norms_table: pd.DataFrame,
        param: str,
        value: float | int,
    ):
        """
        return the upper and lower band interval plus its rendering color
        for the given parameter

        Parameters
        ----------
        norms_table: pd.DataFrame
            the dataframe containing all the normative data.
            The dataframe must have the following columns:

                Parameter: str
                    the name of the parameter

                Rank: str
                    the label defining the interpretation of the value

                Lower: int | float
                    the lower bound of the interval.

                Upper: int | float
                    the upper bound of the interval.

                Color: str
                    code that can be interpreted as a color.

        param: str
            the parameter of interest

        Return
        ------
        intervals: list[tuple[float, float, str, str]]
            a list of tuples containing the upper and lower interval plus the
            corresponding color for the given parameter. An empty list is
            returned in case no intervals are found for the given parameter.

        Raise
        -----
        Warning in case the given parameter is not found.
        """
        # check the inputs
        self.check_intervals(norms_table)
        if not isinstance(param, str):
            raise ValueError("'param' must be a str object.")
        if not isinstance(value, (float, int)):
            raise ValueError("'value' must be a float or int.")

        # get the values
        out: list[tuple[float, float, str, str]] = []
        cols = ["Parameter", "Rank", "Lower", "Upper", "Color"]
        if norms_table.shape[0] > 0:
            norms = norms_table.loc[norms_table.Parameter == str(param)]
            for row in range(norms.shape[0]):
                vals = norms[cols[1:]].iloc[row]
                rnk, low, upp, clr = vals.values.astype(float).flatten()
                if value >= low and value <= upp:
                    out += [(float(low), float(upp), rnk, clr)]

        return out

    def _make_results_table(self) -> pd.DataFrame: ...

    def _make_results_plot(self) -> go.FigureWidget: ...

    def check_intervals(self, normative_intervals: object):
        """check the normative intervals architecture"""
        columns = ["Parameter", "Rank", "Lower", "Upper", "Color"]
        msg = "normative_intervals must be a pandas.DataFrame containing the "
        msg += "following columns: " + str(columns)
        msg2 = "Lower and Upper columns must contain only int or float-like values."
        if not isinstance(normative_intervals, pd.DataFrame):
            raise ValueError(msg)
        if normative_intervals.shape[0] > 0:
            for col in columns:
                if col not in normative_intervals.columns.tolist():
                    raise ValueError(msg)
                if col in ["Lower", "Upper"]:
                    try:
                        _ = normative_intervals[col].astype(float)
                    except Exception:
                        raise ValueError(msg2)

    def results(self):
        """
        return a plotly figurewidget highlighting the resulting data
        and a table with the resulting outcomes as pandas DataFrame.
        """
        raw = self._make_results_table()
        fig = self._make_results_plot()
        return fig, raw

    def summary(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ):
        """
        return a dictionary of plotly FigureWidget objects highlighting the
        test summary and a table reporting the summary data.

        Parameters
        ----------
        normative_intervals: pd.DataFrame, optional
            all the normative intervals. The dataframe must have the following
            columns:

                Parameter: str
                    the name of the parameter

                Rank: str
                    the label defining the interpretation of the value

                Lower: int | float
                    the lower bound of the interval.

                Upper: int | float
                    the upper bound of the interval.

                Color: str
                    code that can be interpreted as a color.

        Returns
        -------
        figures: dict[str, FigureWidget]
            return a dictionary of plotly FigureWidget objects summarizing the
            results of the test.

        tab: pandas DataFrame
            return a pandas dataframe with a summary of the test results.
        """
        self.check_intervals(normative_intervals)
        res = self._make_summary_table(normative_intervals)
        fig = self._make_summary_plot(normative_intervals)
        return fig, res

    def save(self, file_path: str):
        """
        save the test to the input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the test name. If this is not the case, the appropriate extension
            is appended.
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + self.__class__.__name__.lower()
        if not file_path.endswith(extension):
            file_path += extension
        overwrite = False
        while exists(file_path) and not overwrite:
            overwrite = messages.askyesnocancel(
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
        load the test data from an input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the test name. If this is not the case, the appropriate extension
            is appended.

        Returns
        -------
        obj: Self
            the test object
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


@runtime_checkable
class TestBattery(Protocol):
    """
    class allowing to deal with multiple lab tests

    Attributes
    ----------
    tests
        the list of tests being part of the battery.

    Methods
    -------
    summary
        return a summary of the test results both as dictionary of plotly
        FigureWidget objects and as pandas dataframe

    save
        a method allowing the saving of the data in an appropriate format.

    load
        a class method to load a LabTest object saved in its own format.
    """

    # * class variables

    _tests: list[LabTest]

    # * attributes

    @property
    def tests(self):
        """return the list of tests being part of the test battery"""
        return self._tests

    def _make_summary_table(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ) -> pd.DataFrame: ...

    def _make_summary_plot(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ) -> dict[str, go.FigureWidget]: ...

    def _check_norms(self, normative_intervals: object):
        """check the normative intervals architecture"""
        columns = ["Test", "Parameter", "Rank", "Lower", "Upper", "Color"]
        msg = "normative_intervals must be a pandas.DataFrame containing the "
        msg += "following columns: " + str(columns)
        msg2 = "Lower and Upper columns must contain only int or float-like values."
        if not isinstance(normative_intervals, pd.DataFrame):
            raise ValueError(msg)
        for col in columns:
            if col not in normative_intervals.columns.tolist():
                raise ValueError(msg)
            if col in ["Lower", "Upper"]:
                try:
                    _ = normative_intervals[col].astype(float)
                except Exception:
                    raise ValueError(msg2)

    def summary(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ):
        """
        return a dictionary of plotly FigureWidget objects highlighting the
        test summary and a table reporting the summary data.

        Parameters
        ----------
        normative_intervals: pd.DataFrame, optional
            all the normative intervals. The dataframe must have the following
            columns:

                Test: str
                    the name of the target test

                Parameter: str
                    the name of the parameter

                Rank: str
                    the label defining the interpretation of the value

                Lower: int | float
                    the lower bound of the interval.

                Upper: int | float
                    the upper bound of the interval.

                Color: str
                    code that can be interpreted as a color.

        Returns
        -------
        figures: dict[str, FigureWidget]
            return a dictionary of plotly FigureWidget objects summarizing the
            results of the test.

        tab: pandas DataFrame
            return a pandas dataframe with a summary of the test results.
        """
        self._check_norms(normative_intervals)
        res = self._make_summary_table(normative_intervals)
        fig = self._make_summary_plot(normative_intervals)
        return fig, res

    def save(self, file_path: str) -> None:
        """
        save the test battery to the input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the test battery name. If this is not the case, the appropriate
            extension is appended.
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + self.__class__.__name__.lower()
        if not file_path.endswith(extension):
            file_path += extension
        overwrite = False
        while exists(file_path) and not overwrite:
            overwrite = messages.askyesnocancel(
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
        load the test data from an input file

        Parameters
        ----------
        file_path: str
            the path where to save the file. The file extension should mimic
            the test name. If this is not the case, the appropriate extension
            is appended.

        Returns
        -------
        obj: Self
            the test object
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

    # * constructor

    def __init__(self, *tests: LabTest):
        # check the inputs
        msg = "'tests' must be LabTest subclassed objects."
        for test in tests:
            if not isinstance(test, LabTest):
                raise ValueError(msg)

        # store the tests
        self._tests = list(tests)
