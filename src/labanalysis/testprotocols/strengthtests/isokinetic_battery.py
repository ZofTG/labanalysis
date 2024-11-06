"""jumps battery module"""

#! IMPORTS


import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import TestBattery
from .isokinetic import Isokinetic1RMTest

__all__ = ["Isokinetic1RMTestBattery"]


#! CLASSES


class Isokinetic1RMTestBattery(TestBattery):
    """
    generate a test battery from Isokinetic 1RM tests

    Parameters
    *tests: Isokinetic1RMTest
        the list of isokinetic 1RM tests to be analysed
    """

    # constructor

    def __init__(self, *tests: Isokinetic1RMTest):
        for test in tests:
            if not isinstance(test, Isokinetic1RMTest):
                raise ValueError("tests must be an Isokinetic1RMTest instances.")
        super().__init__(*tests)

    @property
    def summary_plots(self):
        """return a set of plotly FigureWidget for each relevant metric"""

        # get 1RM data
        tab = self.summary_table
        tab_cols = ["Side", "Product", "Parameter", "Unit", "Repetition", "Max"]
        tab = tab[tab_cols].copy()
        tab = tab.loc[tab.Parameter == "1RM"]
        rm1 = tab.groupby(["Product", "Side", "Unit"]).max()[["Max"]]
        rm1 = pd.concat([rm1.index.to_frame(), rm1], axis=1).reset_index(drop=True)
        rm1.loc[rm1.index, "Text"] = rm1.Max.map(lambda x: str(x)[:5])
        for grp, dfr in rm1.groupby(["Product", "Side", "Unit"]):
            prod, side, unit = grp
            sub = tab.loc[tab.Product == prod]
            sub = sub.loc[sub.Side == side]
            rep = sub.Repetition.values[np.argmax(sub.Max)]
            idx = (rm1.Product == prod) & (rm1.Side == side)
            rm1.loc[idx, "Repetition"] = rep
        rm1.loc[rm1.index, "Label"] = [
            f"{p} {s}" for p, s in zip(rm1.Product, rm1.Side)
        ]

        # get track data
        res = self.results_table
        tracks = []
        for (prod, side, rep), dfr in rm1.groupby(["Product", "Side", "Repetition"]):
            idx = (
                (res.Product.values == prod)
                & (res.Side.values == side)
                & (res.Repetition.values == rep)
            )
            dfp = res.loc[idx]
            dfp.loc[dfp.index, ("Time", "s")] -= dfp.Time.values[0]
            tracks += [dfp]
        tracks = pd.concat(tracks, ignore_index=True)
        tracks.columns = pd.Index([i[0] for i in tracks.columns])

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

        # plot the isokinetic vs rom data
        grouped_tracks = tracks.groupby(["Product", "Side"])
        for i, ((prod, side), dfr) in enumerate(grouped_tracks):
            fig.add_trace(
                row=1,
                col=1,
                trace=go.Scatter(
                    x=dfr.Time,
                    y=dfr.Load,
                    name=f"{prod} {side}",
                    legendgroup=f"{prod} {side}",
                    mode="lines",
                    fill="tozeroy",
                    opacity=0.4,
                    line_width=6,
                    line_color=px.colors.qualitative.Plotly[i],
                ),
            )

        # plot the 1RM
        fig1 = px.bar(
            data_frame=rm1,
            x="Label",
            y="Max",
            text="Text",
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
        fig.update_yaxes(
            col=1, range=[tracks.Load.min() * 0.9, tracks.Load.max() * 1.1]
        )
        fig.update_yaxes(col=3, range=[rm1.Max.min() * 0.9, rm1.Max.max() * 1.1])
        fig.update_xaxes(row=1, col=1, title="Repetition time (s)")
        fig.update_layout(
            template="simple_white",
            height=400,
            width=800,
        )

        return go.FigureWidget(fig)
