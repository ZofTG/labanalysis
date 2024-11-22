"""jumps battery module"""

#! IMPORTS

from os.path import dirname, join

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
    def summary(self):
        """return a set of plotly FigureWidget for each relevant metric"""

        # get 1RM data
        tab = self.summary_table
        tab_cols = ["Side", "Product", "Parameter", "Unit", "Repetition", "Max"]
        tab = tab[tab_cols].copy()
        rm1 = tab.loc[tab.Parameter == "1RM"]
        rm1 = rm1.groupby(["Product", "Side", "Unit"]).max()[["Max"]]
        rm1 = pd.concat([rm1.index.to_frame(), rm1], axis=1).reset_index(drop=True)
        rm1.loc[rm1.index, "Text"] = rm1.Max.map(lambda x: str(x)[:5] + " kg")
        for grp, dfr in rm1.groupby(["Product", "Side", "Unit"]):
            prod, side, unit = grp
            sub = tab.loc[tab.Product == prod]
            sub = sub.loc[sub.Side == side]
            rep = sub.Repetition.values[np.argmax(sub.Max)]
            idx = (rm1.Product == prod) & (rm1.Side == side)
            rm1.loc[idx, "Repetition"] = rep

        # get track data
        res = self.results_table
        res.columns = pd.Index([i[0] for i in res.columns])
        tracks = []
        for (prod, side, rep), dfr in res.groupby(["Product", "Side", "Repetition"]):
            time_offset = dfr.Time.values.astype(float).flatten()[0]
            dfr.loc[dfr.index, "Time"] -= time_offset
            tracks += [dfr]
        tracks = pd.concat(tracks, ignore_index=True)

        # get the normative values
        colors = {
            "Poor": px.colors.qualitative.Plotly[1],
            "Normal": px.colors.qualitative.Plotly[2],
            "Good": px.colors.qualitative.Plotly[3],
            1: px.colors.qualitative.Plotly[0],
            2: px.colors.qualitative.Plotly[4],
            3: px.colors.qualitative.Plotly[5],
        }
        norm_file = join(dirname(dirname(__file__)), "normative_values.xlsx")
        norms = pd.read_excel(io=norm_file, sheet_name="Isokinetic1RMTest")
        out_fig: dict[str, go.FigureWidget] = {}
        patterns = {"Bilateral": "/", "Right": "-", "Left": "+"}
        sides = sorted([i for i in patterns.keys() if i in rm1.Side.unique()])
        out_tab = []
        for prod in tracks.Product.unique():

            # norms
            norm_val = norms.loc[norms.Product == prod]
            norm_1rm = norm_val.loc[norm_val.Parameter == "1RM"]
            avg_1rm, std_1rm = norm_1rm[["mean", "std"]].values.flatten()
            norm_sym = norm_val.loc[norm_val.Parameter == "Symmetry"]
            avg_sym, std_sym = norm_sym[["mean", "std"]].values.flatten()

            # values
            prod_1rm = rm1.loc[rm1.Product == prod].pivot_table(
                index=None,
                columns="Side",
                values="Max",
            )
            if "Left" in sides and "Right" in sides:
                prod_1rm.loc[prod_1rm.index, "Symmetry"] = (
                    200
                    * (prod_1rm.Right - prod_1rm.Left)
                    / (prod_1rm.Right + prod_1rm.Left)
                )
            else:
                prod_1rm.loc[prod_1rm.index, "Symmetry"] = None
            prod_tracks = tracks.loc[tracks.Product == prod]

            # prepare the output table
            tab = prod_1rm.melt(var_name="Parameter", value_name="Value")

            # get the values range
            min_1rm = min(
                prod_1rm[sides].min(axis=0).min() * 0.9, avg_1rm - 2 * std_1rm
            )
            max_1rm = max(
                prod_1rm[sides].max(axis=0).max() * 1.1, avg_1rm + 2 * std_1rm
            )
            min_load = prod_tracks.Load.min()
            max_load = prod_tracks.Load.max()
            if "Symmetry" in prod_1rm.columns:
                min_sym = min(prod_1rm.Symmetry.min() * 0.9, avg_sym - 2 * std_sym)
                max_sym = max(prod_1rm.Symmetry.min() * 1.1, avg_sym + 2 * std_sym)

            # generate the figure and the subplot grid
            specs = [[{} for _ in sides], [{} for _ in sides]]
            row_titles = ["ISOKINETIC FORCE", "ESTIMATED<br>ISOTONIC 1RM"]
            if "Symmetry" in prod_1rm.columns:
                specs += [[{"colspan": len(sides)}] + [None for i in sides[:-1]]]
                row_titles += ["SYMMETRY"]
            fig = make_subplots(
                rows=3 if "Symmetry" in prod_1rm.columns else 2,
                cols=len(sides),
                subplot_titles=None,
                specs=specs,
                shared_xaxes=False,
                shared_yaxes=False,
                horizontal_spacing=0.15,
                vertical_spacing=0.15,
                row_titles=row_titles,
                column_titles=sides,
                x_title=None,
                y_title=None,
            )
            for i, side in enumerate(sides):

                # isokinetic data
                side_track = prod_tracks.loc[prod_tracks.Side == side]
                for rep, vals in side_track.groupby("Repetition"):
                    fig.add_trace(
                        row=1,
                        col=i + 1,
                        trace=go.Scatter(
                            x=vals.Time,
                            y=vals.Load,
                            name=f"Repetition {rep}",
                            legendgroup=f"Repetition {rep}",
                            mode="lines",
                            opacity=0.3,
                            line_width=3,
                            line_color=px.colors.qualitative.Plotly[rep],
                            showlegend=bool(i == 0),
                            legend="legend",
                        ),
                    )

                # 1RM data
                val_1rm = prod_1rm[side].values[0]
                rank = "Normal"
                if val_1rm < avg_1rm - std_1rm:
                    rank = "Poor"
                elif val_1rm > avg_1rm + std_1rm:
                    rank = "Good"
                tab.loc[tab.Parameter == side, "Rank"] = rank
                fig.add_trace(
                    row=2,
                    col=i + 1,
                    trace=go.Bar(
                        x=[side],
                        y=[val_1rm],
                        text=[str(val_1rm)[:5]],
                        name=side,
                        marker_pattern_shape=patterns[side],
                        marker_color=colors[rank],
                        marker_cornerradius="30%",
                        marker_line_color=colors[rank],
                        marker_line_width=3,
                        textposition="outside",
                        marker_pattern_fillmode="replace",
                        showlegend=False,
                        zorder=0,
                        textfont_size=14,
                    ),
                )

                # 1RM norms
                fig.add_hrect(
                    y0=min_1rm,
                    y1=avg_1rm - std_1rm,
                    name="Poor Strength",
                    showlegend=bool(i == 0),
                    fillcolor=colors["Poor"],
                    line_width=0,
                    opacity=0.1,
                    legend="legend2",
                    row=2,  # type: ignore
                    col=i + 1,  # type: ignore
                )
                fig.add_hline(
                    y=avg_1rm - std_1rm,
                    name="line",
                    showlegend=False,
                    line_width=2,
                    line_dash="dash",
                    line_color=colors["Poor"],
                    opacity=0.5,
                    row=2,  # type: ignore
                    col=i + 1,  # type: ignore
                )
                fig.add_hrect(
                    y0=avg_1rm - std_1rm,
                    y1=avg_1rm + std_1rm,
                    name="Normal Strength",
                    showlegend=bool(i == 0),
                    fillcolor=colors["Normal"],
                    line_width=0,
                    opacity=0.1,
                    legend="legend2",
                    row=2,  # type: ignore
                    col=i + 1,  # type: ignore
                )
                fig.add_hline(
                    y=avg_1rm,
                    name="line",
                    showlegend=False,
                    line_width=2,
                    line_dash="dash",
                    line_color=colors["Normal"],
                    opacity=0.5,
                    row=2,  # type: ignore
                    col=i + 1,  # type: ignore
                )
                fig.add_hrect(
                    y0=avg_1rm + std_1rm,
                    y1=max_1rm,
                    name="Good Strength",
                    showlegend=bool(i == 0),
                    fillcolor=colors["Good"],
                    line_width=0,
                    opacity=0.1,
                    legend="legend2",
                    row=2,  # type: ignore
                    col=i + 1,  # type: ignore
                )
                fig.add_hline(
                    y=avg_1rm + std_1rm,
                    name="line",
                    showlegend=False,
                    line_width=2,
                    line_dash="dash",
                    line_color=colors["Good"],
                    opacity=0.5,
                    row=2,  # type: ignore
                    col=i + 1,  # type: ignore
                )

                # symmetry data
                if i == 0 and "Symmetry" in prod_1rm.columns:
                    sym_val = prod_1rm.Symmetry.values[0]
                    rank = "Normal"
                    if sym_val < -std_sym or sym_val > std_sym:
                        rank = "Poor"
                    tab.loc[tab.Parameter == "Symmetry", "Rank"] = rank
                    fig.add_trace(
                        row=3,
                        col=1,
                        trace=go.Bar(
                            x=[sym_val],
                            y=["Y"],
                            text=[str(abs(sym_val))[:5] + " %"],
                            marker_color=colors[rank],
                            marker_pattern_shape="|",
                            marker_cornerradius="30%",
                            marker_line_color=colors[rank],
                            marker_line_width=3,
                            textposition="outside",
                            marker_pattern_fillmode="replace",
                            showlegend=False,
                            zorder=0,
                            opacity=1,
                            orientation="h",
                            textfont_size=16,
                        ),
                    )

                    # plot the normative areas
                    fig.add_vrect(
                        row=3,  # type: ignore
                        col=1,  # type: ignore
                        x0=-std_sym,
                        x1=+std_sym,
                        name="Normal Symmetry",
                        showlegend=True,
                        fillcolor=colors["Normal"],
                        line_width=0,
                        opacity=0.1,
                        legend="legend3",
                    )
                    fig.add_vline(
                        row=3,  # type: ignore
                        col=1,  # type: ignore
                        x=0,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color="black",
                        opacity=0.5,
                    )
                    fig.add_vline(
                        row=3,  # type: ignore
                        col=1,  # type: ignore
                        x=-std_sym,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color=colors["Normal"],
                        opacity=0.5,
                    )
                    fig.add_vline(
                        row=3,  # type: ignore
                        col=1,  # type: ignore
                        x=std_sym,
                        name="line",
                        showlegend=False,
                        line_width=2,
                        line_dash="dash",
                        line_color=colors["Normal"],
                        opacity=0.5,
                    )
                    fig.add_vrect(
                        row=3,  # type: ignore
                        col=1,  # type: ignore
                        x0=min_sym,
                        x1=-std_sym,
                        name="Poor Symmetry",
                        showlegend=True,
                        fillcolor=colors["Poor"],
                        line_width=0,
                        opacity=0.1,
                        legend="legend3",
                    )
                    fig.add_vrect(
                        row=3,  # type: ignore
                        col=1,  # type: ignore
                        x0=std_sym,
                        x1=max_sym,
                        name="Poor Symmetry",
                        showlegend=False,
                        fillcolor=colors["Poor"],
                        line_width=0,
                        opacity=0.1,
                    )
                    fig.add_annotation(
                        row=3,
                        col=1,
                        text="Left",
                        x=min_sym,
                        y=1,
                        xref="x",
                        yref="y",
                        align="left",
                        valign="top",
                        xanchor="left",
                        yanchor="top",
                        font_size=16,
                        showarrow=False,
                    )
                    fig.add_annotation(
                        row=3,
                        col=1,
                        text="Right",
                        x=max_sym,
                        y=1,
                        xref="x",
                        yref="y",
                        align="right",
                        valign="top",
                        xanchor="right",
                        yanchor="top",
                        font_size=16,
                        showarrow=False,
                    )

            # update the layout
            fig.update_yaxes(row=1, title="kg", range=[min_load, max_load])
            fig.update_xaxes(row=1, title="Repetition time (s)")
            fig.update_yaxes(row=2, title="kg", range=[min_1rm, max_1rm])
            fig.update_xaxes(row=2, matches=None, title="")
            if "Symmetry" in prod_1rm.columns:
                fig.update_yaxes(row=3, title="", showticklabels=False)
                fig.update_xaxes(row=3, title="Asymmetry", range=[min_sym, max_sym])
            fig.update_layout(
                template="simple_white",
                title=f"{prod} Isokinetic 1RM Test",
                height=600 + (300 if "Symmetry" in prod_1rm.columns else 0),
                width=300 * len(sides),
                legend=dict(
                    x=1.1,
                    y=0.87,
                    xanchor="left",
                    yanchor="bottom",
                ),
                legend2=dict(
                    x=1.1,
                    y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    traceorder="normal",
                ),
                legend3=dict(
                    x=1.1,
                    y=0.17,
                    xanchor="left",
                    yanchor="top",
                ),
            )

            out_fig[prod] = go.FigureWidget(fig)
            tab.loc[tab.index, "Parameter"] = tab.Parameter.map(lambda x: "1RM " + x)
            units = ["%" if x.endswith("Symmetry") else "kg" for x in tab.Parameter]
            tab.insert(tab.shape[1] - 1, "Unit", units)
            tab.insert(0, "Product", np.tile(prod, tab.shape[0]))
            out_tab += [tab]

        return out_fig, pd.concat(out_tab, ignore_index=True)
