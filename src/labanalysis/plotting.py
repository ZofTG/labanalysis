"""
plotting module

a set of functions for standard plots creation.

Functions
---------
plot_comparisons_plotly
    A combination of regression and bland-altmann plots which returns a
    Plotly FigureWidget object.
"""

# imports

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, ttest_ind, ttest_rel


__all__ = ["plot_comparisons_plotly"]

# types

NumericArray1D = np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]]

# function


def plot_comparisons_plotly(
    xarr: NumericArray1D,
    yarr: NumericArray1D,
    xlabel: str = "",
    ylabel: str = "",
    confidence: float = 0.95,
    parametric: bool = False,
):
    """
    A combination of regression and bland-altmann plots

    Parameters
    ----------
    xarr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
        the array defining the x-axis in the regression plot.

    yarr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
        the array defining the y-axis in the regression plot.

    xlabel: str
        the label of the x-axis in the regression plot.

    ylabel: str
        the label of the y-axis in the regression plot.

    confidence: float (default = 0.95)
        the confidence interval to be displayed on the Bland-Altmann plot.

    parametric: bool (default = False)
        if True, parametric Bland-Altmann confidence intervals are reported.
        Otherwise, non parametric confidence intervals are provided.
    """

    # generate the figure

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        column_titles=[
            "FITTING MEASURES",
            " ".join(["" if parametric else "NON PARAMETRIC", "BLAND-ALTMAN"]),
        ],
    )

    # update the layout

    fig.update_layout(
        template="plotly",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(title=xlabel, col=1)
    fig.update_yaxes(title=ylabel, col=1)
    fig.update_xaxes(title="MEAN", col=2)
    fig.update_yaxes(title="DELTA", col=2)

    # add the identity line to the regression plot
    ymin = min(np.min(yarr), np.min(xarr))
    ymax = max(np.max(yarr), np.max(xarr))

    fig.add_trace(
        row=1,
        col=1,
        trace=go.Scatter(
            x=[ymin, ymax],
            y=[ymin, ymax],
            mode="lines",
            line_dash="dash",
            line_color="red",
            name="IDENTITY LINE",
            legendgroup="IDENTITY LINE",
        ),
    )

    # add the scatter points to the regression plot

    fig.add_trace(
        row=1,
        col=1,
        trace=go.Scatter(
            x=xarr,
            y=yarr,
            mode="markers",
            marker_color="navy",
            showlegend=False,
            opacity=0.5,
        ),
    )

    # add the fitting metrics

    rmse = np.mean((yarr - xarr) ** 2) ** 0.5
    mape = np.mean(abs(yarr - xarr) / xarr) * 100
    r2 = np.corrcoef(xarr, yarr)[0][1] ** 2
    tt_rel = ttest_rel(xarr, yarr)
    tt_ind = ttest_ind(xarr, yarr)
    txt = [f"RMSE = {rmse:0.4f}"]
    txt += [f"MAPE = {mape:0.1f} %"]
    txt += [f"R<sup>2</sup> = {r2:0.2f}"]
    txt += [
        f"Paired T<sub>df={tt_rel.df:0.0f}</sub> = "  # type: ignore
        + f"{tt_rel.statistic:0.2f} (p={tt_rel.pvalue:0.3f})"  # type: ignore
    ]
    txt += [
        f"Indipendent T<sub>df={tt_ind.df:0.0f}</sub> = "  # type: ignore
        + f"{tt_ind.statistic:0.2f} (p={tt_ind.pvalue:0.3f})"  # type: ignore
    ]
    txt = "<br>".join(txt)

    fig.add_annotation(
        row=1,
        col=1,
        x=ymin,
        y=ymax,
        text=txt,
        showarrow=False,
        xanchor="left",
        align="left",
        valign="top",
        font=dict(family="sans serif", size=12, color="black"),
        bgcolor="white",
        opacity=0.7,
    )

    # plot the data on the bland-altman subplot
    means = (xarr + yarr) / 2
    diffs = yarr - xarr
    xrng = [np.min(means), np.max(means)]
    loa_lbl = f"{confidence * 100:0.0f}% LIMITS OF AGREEMENT"
    if not parametric:
        ref = (1 - confidence) / 2
        loalow, loasup, bias = np.quantile(diffs, [ref, 1 - ref, 0.5])
    else:
        bias = np.mean(diffs)
        scale = np.std(diffs)
        loalow, loasup = norm.interval(confidence, loc=bias, scale=scale)

    fig.add_trace(
        row=1,
        col=2,
        trace=go.Scatter(
            x=means,
            y=diffs,
            mode="markers",
            marker_color="navy",
            showlegend=False,
            opacity=0.5,
        ),
    )

    # plot the bias
    fig.add_trace(
        row=1,
        col=2,
        trace=go.Scatter(
            x=xrng,
            y=[bias, bias],
            mode="lines",
            line_color="green",
            line_dash="dash",
            line_width=4,
            name="BIAS",
            opacity=0.8,
        ),
    )

    fig.add_annotation(
        row=1,
        col=2,
        x=xrng[-1],
        y=bias,
        text=f"{bias:0.2f}",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(family="sans serif", size=12, color="green"),
    )

    # plot the limits of agreement
    fig.add_trace(
        row=1,
        col=2,
        trace=go.Scatter(
            x=[xrng[0], xrng[1], xrng[1], xrng[0], xrng[0]],
            y=[loalow, loalow, loasup, loasup, loalow],
            mode="lines",
            fill="toself",
            line_color="black",
            line_width=0,
            name=loa_lbl,
            legendgroup=loa_lbl,
            opacity=0.3,
        ),
    )

    fig.add_annotation(
        row=1,
        col=2,
        x=xrng[-1],
        y=loalow,
        text=f"{loalow:0.2f}",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(family="sans serif", size=12, color="black"),
        name=loa_lbl,
    )

    fig.add_annotation(
        row=1,
        col=2,
        x=xrng[-1],
        y=loasup,
        text=f"{loasup:0.2f}",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(family="sans serif", size=12, color="black"),
        name=loa_lbl,
    )

    return go.FigureWidget(data=fig.data, layout=fig.layout)
