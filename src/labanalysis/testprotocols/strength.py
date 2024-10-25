"""
strength testing module

CLASSES

Isokinetic1RMTest
    a class to handle Isokinetic 1RM tests
"""

#! IMPORTS


from os.path import exists
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from labio.read.biostrength import Product as BiostrengthProduct
from plotly.subplots import make_subplots

from .. import signalprocessing as sp
from .base import LabTest

#! CONSTANTS


__all__ = ["Isokinetic1RMTest"]

#! CLASSES


class Isokinetic1RMTest(LabTest):
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

    coefs_1rm: dict[str, float]
        the 1RM conversion coefficients

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

    _repetitions: list[pd.DataFrame]
    _1rm_coefs: tuple[int | float, int | float]
    _raw: pd.DataFrame

    # * attributes

    @property
    def raw(self):
        """return the raw data"""
        return self._raw

    @property
    def repetitions(self):
        """return the tracked repetitions data"""
        return self._repetitions

    @property
    def coefs_1rm(self):
        """return the 1RM coefficients related to the isokinetic test"""
        return self._1rm_coefs

    @property
    def peak_load(self):
        """return the ending position of the repetitions"""
        return float(self.summary_table["max"].max())

    @property
    def estimated_1rm(self):
        """return the predicted 1RM"""
        b1, b0 = self._1rm_coefs
        return self.peak_load * b1 + b0

    @property
    def rom1(self):
        """return the ending position of the repetitions"""
        return float(np.mean([dfr.position.max() for dfr in self.repetitions]))

    @property
    def rom0(self):
        """return the starting position of the repetitions"""
        return float(np.mean([dfr.position.min() for dfr in self.repetitions]))

    @property
    def rom(self):
        """return the Range of Movement"""
        return self.rom1 - self.rom0

    @property
    def summary_plot(self):
        """return a plotly figurewidget summarizing the test outputs"""

        # generate the figure and the subplot grid
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["ISOKINETIC FORCE", "ESTIMATED<br>ISOTONIC 1RM"],
            specs=[[{"colspan": 2}, None, {}]],
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.15,
            vertical_spacing=0.15,
            row_titles=None,
            column_titles=None,
            x_title=None,
            y_title=None,
        )

        # get the isokinetic vs rom data for each repetition
        raw = self.results_table.dropna()[["Load", "REPETITION"]]
        raw.columns = pd.Index(["Load", "Repetition"])

        # plot the isokinetic vs rom data
        for rep, dfr in raw.groupby("Repetition"):
            nrep = int(rep)  # type: ignore
            fig.add_trace(
                row=1,
                col=1,
                trace=go.Scatter(
                    x=dfr.index - dfr.index[0],
                    y=dfr.Load,
                    name=f"Rep {nrep}",
                    legendgroup="REPETTION",
                    mode="lines",
                    fill="tozeroy",
                    opacity=0.4,
                    line_width=6,
                    line_color=px.colors.qualitative.Plotly[nrep - 1],
                ),
            )

        # get the 1RM
        loads = self.summary_table.T["1RM"].T.MEAN.values.astype(float)
        base = np.min(loads) * 0.9
        df_1rm = pd.DataFrame(
            {
                "1RM": loads - base,
                "TEXT": [f"{i:0.1f}" for i in loads],
                "REPETITION": [f"REP {i + 1}" for i in np.arange(len(loads))],
                "BASE": np.tile(base, len(loads)),
            }
        )
        df_1rm.sort_values("REPETITION", inplace=True)

        # plot the 1RM
        fig1 = px.bar(
            data_frame=df_1rm,
            x="REPETITION",
            y="1RM",
            base="BASE",
            text="TEXT",
            barmode="group",
        )
        fig1.update_traces(
            showlegend=False,
            marker_color=px.colors.qualitative.Plotly,
        )
        for i, trace in enumerate(fig1.data):
            fig.add_trace(row=1, col=3, trace=trace)

        # update the layout and return
        fig.update_yaxes(title="kg")
        fig.update_yaxes(col=1, range=[raw.Load.min() * 0.9, raw.Load.max() * 1.1])
        fig.update_xaxes(row=1, col=1, title="Repetition time (s)")
        fig.update_layout(
            template="simple_white",
            height=400,
            width=800,
        )

        return go.FigureWidget(fig)

    @property
    def results_plot(self):
        """return a plotly figurewidget highlighting the test results"""

        # prepare the data
        tab = self.results_table.copy()
        tab.insert(0, ("Time", "s"), tab.index.to_numpy())
        tab.reset_index(inplace=True, drop=True)
        tab.columns = pd.Index([i[0] for i in tab.columns])

        # generate the figure and the subplot grid
        fig = px.line(
            data_frame=tab,
            x="Time",
            y="Load",
            color="REPETITION",
            line_dash="REPETITION",
            template="simple_white",
            width=1200,
            height=600,
        )

        return go.FigureWidget(fig)

    @property
    def summary_table(self):
        """return a table containing summary statistics about the test"""
        des = self.results_table.dropna().melt(
            id_vars=[("REPETITION", "#")],
            var_name="PARAMETER",
            value_name="VALUE",
        )
        des.columns = pd.Index(["REPETITION", "PARAMETER", "VALUE"])
        des.insert(
            0,
            "UNIT",
            des.PARAMETER.map(
                lambda x: (
                    "m"
                    if x == "Position"
                    else ("kgf" if x == "Load" else ("m/s" if x == "Velocity" else "W"))
                )
            ),
        )
        b1, b0 = self.coefs_1rm
        out = [des]
        for grp, dfr in des.loc[des.PARAMETER == "Load"].groupby(["REPETITION"]):
            line = {
                "UNIT": "kgf",
                "REPETITION": grp[0],
                "PARAMETER": "1RM",
                "VALUE": b0 + b1 * dfr.VALUE.max(),
            }
            out += [pd.DataFrame(pd.Series(line)).T]
        des = pd.concat(out, ignore_index=True)
        des = des.groupby(["PARAMETER", "UNIT", "REPETITION"])
        out = {
            "MEAN": des.mean(),
            "STD": des.std(),
            "MIN": des.min(),
            "MEDIAN": des.median(),
            "MAX": des.max(),
        }
        des = pd.concat(list(out.values()), axis=1)
        des.columns = pd.Index(list(out.keys()))

        return des

    @property
    def results_table(self):
        """
        return a table containing the whole data
        """
        out = self.raw
        for i, rep in enumerate(self.repetitions):
            out.loc[rep.index, ("REPETITION", "#")] = i + 1
        return out

    # * methods

    def _check_array(self, obj: object):
        """
        private method used evaluate if obj is an iterable of float or int

        Parameters
        ----------
        obj: object
            the object to be checked

        Returns
        -------
        arr: ArrayLike
            a 1D array of float obtained from obj.
        """
        try:
            return np.array([obj]).astype(float).flatten()
        except Exception as exc:
            msg = "obj must be an Iterable of float or int."
            raise ValueError(msg) from exc

    # * constructors

    def __init__(
        self,
        time: Iterable[float | int],
        position: Iterable[float | int],
        load: Iterable[float | int],
        coefs_1rm: tuple[float | int, float | int] = (0, 1),
    ):
        # check the inputs
        tarr = self._check_array(time)
        parr = self._check_array(position)
        larr = self._check_array(load)
        msg = "'coefs_1rm' must be a tuple with 2 floats."
        if not isinstance(coefs_1rm, (list, tuple)):
            raise ValueError(msg)
        for i in coefs_1rm:
            if not isinstance(i, (float, int)):
                raise ValueError(msg)
        self._1rm_coefs = coefs_1rm

        # get the raw data
        self._raw = pd.DataFrame(
            data=[parr, larr],
            index=pd.MultiIndex.from_tuples([("Position", "m"), ("Load", "kgf")]),
            columns=pd.Index(tarr, name="Time (s)"),
        ).T

        # get the repetitions
        parr, farr = self._raw.values.T.astype(float)
        if abs(np.min(parr)) > abs(np.max(parr)):
            parr *= -1
        parr -= parr[0]
        varr = sp.winter_derivative1(parr, tarr)
        farr = farr[1:-1]
        parr = farr * varr
        fsamp = float(1 / np.mean(np.diff(tarr)))
        parr = sp.butterworth_filt(
            arr=parr,
            fcut=1,
            fsamp=fsamp,
            order=6,
            ftype="lowpass",
            phase_corrected=True,
        )
        batches = sp.continuous_batches(parr > 0.02 * np.max(parr))
        if len(batches) == 0:
            raise RuntimeError("No repetitions have been found")
        samples = [len(i) for i in batches]
        batches = [batches[i] for i in np.sort(np.argsort(samples)[::-1][:3])]
        self._repetitions = [pd.DataFrame(self.raw.iloc[b, :]) for b in batches]

    @classmethod
    def from_biostrength_file(cls, file: str, product: BiostrengthProduct):
        """
        generate directly from raw file

        Parameters
        ----------
        file : str
            the path to the file returned from a Biostrength device

        product: BiostrengthProduct
            the product instance defining the product from which the file has
            been generated.
        """
        # check the inputs
        msg = "'file' must be the path to a valid .txt file"
        if not isinstance(file, str) or not exists(file):
            raise ValueError(msg)
        if not issubclass(product, BiostrengthProduct):  # type: ignore
            raise ValueError("'product' must be a valid Biostrength Product")

        # read the data
        try:
            bio = product.from_file(file)
        except Exception as exc:
            msg = "An error occurred reading the provided file."
            raise RuntimeError(msg) from exc

        # generate the object instance
        return cls(
            time=bio.time_s,
            position=bio.position_lever_m,
            load=bio.load_lever_kgf,
            coefs_1rm=(bio.rm1_coefs[0], bio.rm1_coefs[1]),
        )
