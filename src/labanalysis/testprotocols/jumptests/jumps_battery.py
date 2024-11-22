"""jumps battery module"""

#! IMPORTS

from os.path import dirname, join

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import TestBattery
from .counter_movement_jump import CounterMovementJumpTest
from .side_jump import SideJumpTest
from .single_leg_jump import SingleLegJumpTest
from .squat_jump import SquatJumpTest

__all__ = ["JumpTestBattery"]


#! CLASSES


class JumpTestBattery(TestBattery):
    """
    generate a test battery from jump tests

    Parameters
    *tests: SquatJumpTest | CounterMovementJumpTest | SideJumpTest | SingleLegJumpTest
        the list of jump tests to be analysed
    """

    # constructor

    def __init__(
        self,
        *tests: SquatJumpTest
        | CounterMovementJumpTest
        | SideJumpTest
        | SingleLegJumpTest,
    ):
        types = (
            SquatJumpTest,
            CounterMovementJumpTest,
            SideJumpTest,
            SingleLegJumpTest,
        )
        msg = f"tests must be an {types} instances."
        for test in tests:
            if not isinstance(test, types):
                raise ValueError(msg)
        super().__init__(*tests)

    @property
    def summary(self):
        """
        return a set of plotly FigureWidget for each relevant metric
        and a table with the summary metrics
        """

        # get the data
        tab = self.summary_table
        if not any(i == "Side" for i in tab.columns):
            tab.insert(0, "Side", np.tile("Bilateral", tab.shape[0]))
        else:
            tab.loc[tab.Side.isna(), "Side"] = "Bilateral"
        tab.loc[tab.index, "Test"] = tab.Test.map(
            lambda x: x.replace("JumpTest", "") + " Jump"
        )
        tab = tab.drop(["Mean", "Std"], axis=1)
        tab.loc[tab.index, "Text"] = tab[["Best", "Unit"]].apply(
            lambda x: str(abs(x[0]))[:5] + " " + x[1],
            axis=1,
            raw=True,
        )
        colors = px.colors.qualitative.Plotly
        sides = np.sort(tab.Side.unique())
        tab.loc[tab.index, "Color"] = tab.Side.map(
            lambda x: colors[np.where(sides == x)[0][0]]
        )
        tab.sort_values("Side", inplace=True)

        # get the normative values
        normative_values = pd.read_excel(
            io=join(dirname(dirname(__file__)), "normative_values.xlsx"),
            sheet_name="Jumps",
        )
        normative_values.loc[normative_values.index, "Test"] = (
            normative_values.Test.map(lambda x: x.replace(" ", ""))
        )
        colors = {
            "Poor": px.colors.qualitative.Plotly[1],
            "Normal": px.colors.qualitative.Plotly[2],
            "Good": px.colors.qualitative.Plotly[3],
            "Left": px.colors.qualitative.Plotly[0],
            "Right": px.colors.qualitative.Plotly[0],
            "Bilateral": px.colors.qualitative.Plotly[0],
        }
        patterns = {"Left": ".", "Right": "x", "Bilateral": "/"}

        # generate one figure for each parameter
        out: dict[str, go.FigureWidget] = {}
        for parameter, dfr in tab.groupby("Parameter"):

            # generate the output figure frame and get the ranges
            vmax = dfr.Best.max() * 1.1
            vmin = dfr.Best.min() * 0.9
            tests = dfr.Test.unique().flatten().tolist()
            fig = make_subplots(
                rows=1,
                cols=len(tests),
                subplot_titles=tests,
                horizontal_spacing=0.1,
            )
            for i, test in enumerate(tests):
                dft = dfr.loc[dfr.Test == test]

                # get the normative values
                idx = normative_values.Test == test.replace(" ", "") + "Test"
                norms = normative_values.loc[idx]
                norms = norms.loc[norms.Parameter == parameter]
                avg, std = norms[["mean", "std"]].values.astype(float).flatten()
                if str(parameter).endswith("Imbalance"):
                    vmax = max(vmax, avg + 3 * std)
                    vmin = min(vmin, avg - 3 * std)
                else:
                    vmax = max(vmax, avg + 2 * std)
                    vmin = min(vmin, avg - 2 * std)

                # prepare the data
                for side, dfs in dft.groupby("Side"):
                    rnk = "Normal"
                    if str(parameter).endswith("Imbalance"):
                        xarr = dfs.Best.values.astype(float)
                        yarr = dfs.Side.values.astype(str)
                        orientation = "h"
                        if xarr < -std or xarr > +std:
                            rnk = "Poor"
                    else:
                        yarr = dfs.Best.values.astype(float)
                        xarr = dfs.Side.values.astype(str)
                        orientation = "v"
                        if yarr < avg - std:
                            rnk = "Poor"
                        elif yarr > avg + std:
                            rnk = "Good"
                    tab.loc[dfs.index, "Rank"] = rnk
                    fig.add_trace(
                        row=1,
                        col=i + 1,
                        trace=go.Bar(
                            x=xarr,
                            y=yarr,
                            text=dfs.Text.values,
                            marker_color=colors[rnk],
                            marker_pattern_shape=patterns[str(side)],
                            marker_cornerradius="30%",
                            marker_line_color=colors[rnk],
                            marker_line_width=3,
                            name=side,
                            showlegend=False,
                            opacity=1,
                            orientation=orientation,
                            textfont_size=16,
                        ),
                    )

                # plot the normative areas
                if str(parameter).endswith("Imbalance"):

                    # this is the case of muscle symmetry
                    fig.add_vrect(
                        x0=-std,
                        x1=+std,
                        name="Normal",
                        showlegend=bool(i == 0),
                        fillcolor=colors["Normal"],
                        line_width=0,
                        opacity=0.1,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_vrect(
                        x0=-100,
                        x1=max(-100, -std),
                        name="Poor",
                        showlegend=bool(i == 0),
                        fillcolor=colors["Poor"],
                        line_width=0,
                        opacity=0.1,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_vrect(
                        x0=min(100, +std),
                        x1=100,
                        name="Poor",
                        showlegend=False,
                        fillcolor=colors["Poor"],
                        line_width=0,
                        opacity=0.1,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_vline(
                        x=0,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color="black",
                        opacity=0.5,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_vline(
                        x=-std,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color=colors["Normal"],
                        opacity=0.5,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_vline(
                        x=std,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color=colors["Normal"],
                        opacity=0.5,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                else:

                    # this is any other case
                    fig.add_hrect(
                        y0=0,
                        y1=max(0, avg - std),
                        name="Poor",
                        showlegend=bool(i == 0),
                        fillcolor=colors["Poor"],
                        line_width=0,
                        opacity=0.1,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_hline(
                        y=avg - std,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color=colors["Poor"],
                        opacity=0.5,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_hrect(
                        y0=avg - std,
                        y1=avg + std,
                        name="Normal",
                        showlegend=bool(i == 0),
                        fillcolor=colors["Normal"],
                        line_width=0,
                        opacity=0.1,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_hline(
                        y=avg,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color=colors["Normal"],
                        opacity=0.5,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_hrect(
                        y0=avg + std,
                        y1=1e15,
                        name="Good",
                        showlegend=bool(i == 0),
                        fillcolor=colors["Good"],
                        line_width=0,
                        opacity=0.1,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )
                    fig.add_hline(
                        y=avg + std,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color=colors["Good"],
                        opacity=0.5,
                        row=1,  # type: ignore
                        col=i + 1,  # type: ignore
                    )

            # update the layout
            fig.update_traces(
                zorder=0,
                marker_pattern_fillmode="replace",
                textposition="outside",
            )
            fig.update_layout(
                title=parameter,
                template="simple_white",
                legend=dict(traceorder="normal"),
            )
            fig.update_xaxes(title="", matches=None)
            fig.update_yaxes(title="")
            if str(parameter).endswith("Imbalance"):
                for col in np.arange(2) + 1:
                    fig.add_annotation(
                        text="Left",
                        x=-45,
                        y=0.5,
                        xref="x",
                        yref="y",
                        align="left",
                        valign="bottom",
                        font_size=20,
                        showarrow=False,
                        row=1,
                        col=col,
                    )
                    fig.add_annotation(
                        text="Right",
                        x=+45,
                        y=0.5,
                        xref="x",
                        yref="y",
                        align="right",
                        valign="bottom",
                        font_size=20,
                        showarrow=False,
                        row=1,
                        col=col,
                    )
                fig.update_xaxes(range=[-100, 100], title="Asymmetry")
                fig.update_yaxes(visible=False)
            else:
                fig.update_yaxes(range=[vmin, vmax], title=dft.Unit.values[0])

            # store
            out[str(parameter)] = go.FigureWidget(fig)

        # adjust the output table
        tab = tab[["Test", "Parameter", "Side", "Unit", "Best", "Rank"]]
        tab.columns = pd.Index([i.replace("Best", "Value") for i in tab.columns])

        return out, tab
