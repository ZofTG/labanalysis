"""
strength testing module

CLASSES

Isokinetic1RMTest
    a class to handle Isokinetic 1RM tests
"""

#! IMPORTS

from typing import Literal
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...io.read.biostrength import BiostrengthProduct
from ... import signalprocessing as sp
from ..base import LabTest

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

    product: BiostrengthProduct
        the product on which the test has been performed

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

    _repetitions: list[BiostrengthProduct]
    _product: BiostrengthProduct
    _side: Literal["Bilateral", "Left", "Right"]

    # * attributes

    @property
    def side(self):
        """get the side of the test"""
        return self._side

    @property
    def repetitions(self):
        """return the tracked repetitions data"""
        return self._repetitions

    @property
    def product(self):
        """return the product on which the test has been performed"""
        return self._product

    @property
    def peak_load(self):
        """return the ending position of the repetitions"""
        return float(self.summary_table["max"].max())

    @property
    def estimated_1rm(self):
        """return the predicted 1RM"""
        b1, b0 = self.product.rm1_coefs
        return self.peak_load * b1 + b0

    @property
    def summary(self):
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
        tab.columns = pd.Index([i[0] for i in tab.columns])

        # generate the figure and the subplot grid
        fig = px.line(
            data_frame=tab,
            x="Time",
            y="Load",
            color="Repetition",
            line_dash="Repetition",
            template="simple_white",
            width=1200,
            height=600,
        )

        return go.FigureWidget(fig)

    @property
    def summary_table(self):
        """return a table containing summary statistics about the test"""
        des = (
            self.results_table.dropna()
            .drop(("Time", "s"), axis=1)
            .melt(
                id_vars=[("Side", "-"), ("Product", "-"), ("Repetition", "#")],
                var_name="Parameter",
                value_name="Value",
            )
        )
        des.columns = pd.Index(["Side", "Product", "Repetition", "Parameter", "Value"])

        def get_unit(param: str):
            if param == "Position":
                return "mm"
            elif param == "Load":
                return "kgf"
            elif param == "Speed":
                return "m/s"
            else:
                return "W"

        des.insert(0, "Unit", des.Parameter.map(get_unit))

        # add the 1RM
        out = [des]
        b1, b0 = self.product.rm1_coefs
        for i, rep in enumerate(self.repetitions):
            line = {
                "Side": self.side,
                "Product": self.product.name,
                "Unit": "kgf",
                "Repetition": i + 1,
                "Parameter": "1RM",
                "Value": rep.load_lever_kgf.max() * b1 + b0,
            }
            out += [pd.DataFrame(pd.Series(line)).T]
        des = pd.concat(out, ignore_index=True)
        des = des.groupby(["Side", "Product", "Parameter", "Unit", "Repetition"])
        out = {
            "Mean": des.mean(),
            "Std": des.std(),
            "Min": des.min(),
            "Median": des.median(),
            "Max": des.max(),
        }
        des = pd.concat(list(out.values()), axis=1)
        des.columns = pd.Index(list(out.keys()))
        des = pd.concat([des.index.to_frame(), des], axis=1).reset_index(drop=True)

        return des

    @property
    def results_table(self):
        """
        return a table containing the whole data
        """
        out = []
        for i, rep in enumerate(self.repetitions):
            new = rep.as_dataframe()
            new.insert(0, ("Repetition", "#"), np.tile(i + 1, new.shape[0]))
            new.insert(0, ("Side", "-"), np.tile(self.side, new.shape[0]))
            new.insert(0, ("Product", "-"), np.tile(self.product.name, new.shape[0]))
            out += [new]
        return pd.concat(out, ignore_index=True)

    # * constructors

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["Bilateral", "Left", "Right"],
    ):

        # check the input
        if not issubclass(product.__class__, BiostrengthProduct):
            raise ValueError("'product' must be a valid Biostrength Product.")
        if not side in ["Bilateral", "Left", "Right"]:
            raise ValueError("'side' must be any of 'Bilateral', 'Left', 'Right'")

        # get the raw data
        self._product = product  # type: ignore
        self._side = side
        raw = product.as_dataframe()

        # get the repetitions
        tarr, parr, farr = raw.values[:, :3].T
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
        start_batches = sp.continuous_batches(parr > 5)
        if len(start_batches) == 0:
            raise RuntimeError("No repetitions have been found")
        samples = np.argsort([np.max(parr[i]) for i in start_batches])[::-1][:3]
        starts = [start_batches[i][0] for i in np.sort(samples)]
        stop_batches = sp.continuous_batches(parr < -5)
        self._repetitions = []
        for start in starts:
            stops = [
                i[-1]
                for i in stop_batches
                if i[0] > start and tarr[i[-1]] - tarr[i[0]] > 0.5
            ]
            if len(stops) > 0:
                stop = np.min(stops) + 1
                self._repetitions += [self.product.slice(tarr[start], tarr[stop])]
        if len(self._repetitions) == 0:
            raise RuntimeError("No repetitions have been found.")
        self.product.slice(
            start_time=self.repetitions[0].time_s[0],
            stop_time=self.repetitions[-1].time_s[-1],
        )
