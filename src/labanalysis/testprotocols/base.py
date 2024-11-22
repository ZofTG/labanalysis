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

    Methods
    -------
    summary
        return a summary of the test results both as:
            * plotly FigureWidget
            * pandas dataframe

    results
        return the "raw" test results both as:
            * plotly FigureWidget
            * pandas dataframe

    save
        a method allowing the saving of the data in an appropriate format.

    load
        a class method to load a LabTest object saved in its own format.
    """

    def _make_summary_table(
        self,
        normative_intervals: dict[
            str, dict[str, tuple[list[tuple[int | float]] | tuple[int | float], str]]
        ] = {},
    ):
        """
        make the table defining the summary results.

        Parameters
        ----------
        normative_intervals: dict[str, dict[str, tuple[list[tuple[int | float]] | tuple[int | float], str]]]
            the parameters on which the normative intervals have to be
            represented.
        """
        # get the summary results in long format
        out = self._get_jump_features()

        # add the normative bands
        for (jump, param), dfr in out.groupby(["Jump", "Parameter"]):

            # set the normative band
            if str(param) in list(normative_intervals.keys()):
                norms = normative_intervals[str(param)]
                val = dfr.Value.values[0]
                for lvl, norm in norms.items():
                    vals = norm[0] if isinstance(norm[0], list) else [norm[0]]
                    for low, upp in vals:  # type: ignore
                        if val >= low and val <= upp:
                            out.loc[dfr.index, "Interpretation"] = lvl
                            out.loc[dfr.index, "Color"] = norm[-1]
                            break

        return out

    def _make_summary_plot(
        self,
        data_frame: pd.DataFrame,
        param_col: str,
        value_col: str,
        xaxis_col: str | None = None,
        normative_intervals: dict[
            str, dict[str, tuple[list[tuple[int | float]] | tuple[int | float], str]]
        ] = {},
    ):
        """
        make the plot defining the summary results of the test

        Parameters
        ----------
        data_frame : pd.DataFrame
            the summary dataframe

        param_col : str
            the label of the column in data_frame referring to the parameters
            to be plotted

        value_col : str
            the label of the column in data_frame referring to the values
            corresponding to the height of each bar

        xaxis_col : str | None, optional
            the label of the column in data_frame referring to the bars to be
            plotted. If None, this parameter is ignored.

        normative_intervals:dict[str, dict[str, tuple[list[tuple[int | float]] | tuple[int | float], str]]], optional
            the parameters on which the normative intervals have to be
            represented.

        Returns
        -------
        fig: FigureWidget
            the output figure
        """

        def check_col(data_frame: pd.DataFrame, col: object, lbl: str):
            if not isinstance(data_frame, pd.DataFrame):
                raise ValueError("data_frame must be a pandas DataFrame")
            msg = f"{lbl} must be a string defining one column in data_frame."
            if not isinstance(col, str):
                raise ValueError(msg)
            if not any([i == col for i in data_frame.columns]):
                raise ValueError(msg)

        # check the inputs
        check_col(data_frame, param_col, "param_col")
        check_col(data_frame, value_col, "value_col")
        if xaxis_col is not None:
            check_col(data_frame, xaxis_col, "xaxis_col")

        # build the output figure
        parameters = data_frame[param_col].unique()
        fig = make_subplots(
            rows=1,
            cols=len(parameters),
            subplot_titles=parameters,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.1,
            row_titles=None,
            column_titles=parameters.tolist(),
            x_title=None,
            y_title=None,
        )

        # populate the figure
        for i, parameter in enumerate(parameters):

            # get the data and the normative bands
            dfr = data_frame.loc[data_frame[param_col] == parameter]
            if any([i == parameter for i in normative_intervals.keys()]):
                norms = normative_intervals[parameter]
            else:
                norms = {}

            # get a bar plot with optional normative bands
            xval = "Jump" if xaxis_col is None else xaxis_col
            fig0 = bars_with_normative_bands(
                data_frame=dfr,
                yarr=xval if parameter.endswith("Imbalance") else value_col,
                xarr=value_col if parameter.endswith("Imbalance") else xval,
                orientation="h" if parameter.endswith("Imbalance") else "v",
                unit=dfr.Unit.values[0],
                intervals=norms,  # type: ignore
            )[0]

            # add the figure data and annotations to the proper figure
            for trace in fig0.data:
                fig.add_trace(row=1, col=i + 1, trace=trace)
            for shape in fig0.layout["shapes"]:  # type: ignore
                showlegend = [
                    i["name"] == shape["name"]  # type: ignore
                    for i in fig.layout["shapes"]  # type: ignore
                ]
                showlegend = not any(showlegend)
                shape.update(  # type: ignore
                    legendgroup=shape["name"],  # type: ignore
                    showlegend=showlegend,
                )
            for shape in fig0.layout.shapes:  # type: ignore
                fig.add_shape(shape, row=1, col=i + 1)
            if parameter.endswith("Imbalance"):
                fig.update_xaxes(
                    row=1,
                    col=i + 1,
                    range=fig0.layout["xaxis"].range,  # type: ignore
                )
            else:
                fig.update_yaxes(
                    row=1,
                    col=i + 1,
                    range=fig0.layout["yaxis"].range,  # type: ignore
                )
        return go.FigureWidget(fig)

    def _check_norms(self, normative_intervals: object):
        """check the normative intervals architecture"""

        if not isinstance(normative_intervals, dict):
            raise ValueError("normative_intervals must be a dict")

        for key, norms in normative_intervals.items():
            if not isinstance(key, str):
                raise ValueError(f"{key} must be a str.")
            if not isinstance(norms, dict):
                raise ValueError(f"the value of {key} must be a dict object.")
            for lvl, vals in norms.items():
                if not isinstance(lvl, str):
                    raise ValueError(f"{lvl} must be a str.")
                if not isinstance(vals, tuple):
                    msg = f"the value of {key}-{lvl} must be a tuple"
                    raise ValueError(msg)
                msg = "the first and second values of each normative set "
                msg += "must be a float, int or a list of float/int"
                refs = vals[0] if isinstance(vals[0], list) else [vals[0]]
                for val in refs:
                    if not all([isinstance(i, (float, int)) for i in val]):
                        raise ValueError(msg)
                if (
                    isinstance(vals[0], (float, int))
                    != isinstance(vals[1], (float, int))
                ) or (
                    isinstance(vals[0], list)
                    and isinstance(vals[1], list)
                    and len(vals[0]) != len(vals[1])
                ):
                    msg = f"the first two elements of the {key}-{lvl} pair "
                    msg += "must have the same number of elements."
                    raise ValueError(msg)
                if not isinstance(vals[1], str):
                    msg = f"the third value of {key}-{lvl} "
                    msg += "must be a string defining a valid color."
                    raise ValueError(msg)

    def results(self):
        """
        return a plotly figurewidget highlighting the resulting data
        and a table with the resulting outcomes as pandas DataFrame.
        """
        raw = self._make_results_table()
        fig = self._make_results_plot(raw)
        return fig, raw

    def summary(
        self,
        normative_intervals: dict[
            str, dict[str, tuple[list[tuple[int | float]] | tuple[int | float], str]]
        ] = {},
    ):
        """
        return a plotly bar plot highlighting the test summary and a table
        reporting the summary data.

        Parameters
        ----------
        normative_intervals: dict[str, dict[str, tuple[list[tuple[int | float]] | tuple[int | float], str]]],
            one or more key-valued dictionaries defining the properties
            returned by the test. The keys should be:
                "Elevation"
                "Takeoff velocity"
                "<muscle> Imbalance"
            Where <muscle> denotes an (optional) investigated muscle.

            For each key, a dict shall be provided as value having structure:
                band_name: (lower_bound, upper_bound, color)

            Here the upper and lower bounds should be considered as inclusive
            of the provided values, and the color should be a string object
            that can be interpreted as color.

        Returns
        -------
        fig: plotly FigureWidget
            return a plotly FigureWidget object summarizing the results of the
            test.

        tab: pandas DataFrame
            return a pandas dataframe with a summary of the test results.
        """
        self._check_norms(normative_intervals)
        res = self._make_summary_table(normative_intervals)
        fig = self._make_summary_plot(
            data_frame=res,
            param_col="Parameter",
            value_col="Value",
            xaxis_col=None,
            normative_intervals=normative_intervals,
        )
        return fig, res

    def save(self, file_path: str) -> None:
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
        return a summary of the test battery results both as:
            * dict with output parameters name as key and plotly FigureWidget
            as values
            * pandas dataframe.

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

    def summary(
        self, **normative_intervals
    ) -> tuple[dict[str, go.FigureWidget], pd.DataFrame]: ...

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
