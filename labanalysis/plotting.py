"""
plotting module

a set of functions for standard plots creation.

Functions
---------
plot_comparisons_plotly
    A combination of regression and bland-altmann plots which returns a
    Plotly FigureWidget object.
"""

#! IMPORTS

from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
import plotly.express.colors as pcolors
from plotly.subplots import make_subplots
from scipy.stats import norm, ttest_ind, ttest_rel

__all__ = ["plot_comparisons_plotly"]

#! TYPES

NumericArray1D = np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]]
ObjectArray1D = np.ndarray[Literal[1], Any]

#! FUNCTION


def plot_comparisons_plotly(
    xarr: NumericArray1D,
    yarr: NumericArray1D,
    color: ObjectArray1D | None = None,
    xlabel: str = "",
    ylabel: str = "",
    confidence: float = 0.95,
    parametric: bool = False,
    fig: go.Figure | go.FigureWidget | None = None,
    row: int = 1,
):
    """
    A combination of regression and bland-altmann plots

    Parameters
    ----------
    xarr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
        the array defining the x-axis in the regression plot.

    yarr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
        the array defining the y-axis in the regression plot.

    color: np.ndarray[Literal[1], np.dtype[Any]] | None (default = None)
        the array defining the color of each sample in the regression plot.

    xlabel: str (default = "")
        the label of the x-axis in the regression plot.

    ylabel: str (default = "")
        the label of the y-axis in the regression plot.

    confidence: float (default = 0.95)
        the confidence interval to be displayed on the Bland-Altmann plot.

    parametric: bool (default = False)
        if True, parametric Bland-Altmann confidence intervals are reported.
        Otherwise, non parametric confidence intervals are provided.

    fig: go.Figure | go.FigureWidget | None (default = None)
        an already existing figure where to add the plot along the passed row

    row: int (default = 1)
        the index of the row where to put the plots
    """

    # generate the figure
    if fig is None:
        fig = make_subplots(
            rows=max(1, row),
            cols=2,
            shared_xaxes=False,
            shared_yaxes=False,
            column_titles=[
                "FITTING MEASURES",
                " ".join(["" if parametric else "NON PARAMETRIC", "BLAND-ALTMAN"]),
            ],
        )
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

    fig.update_xaxes(title=xlabel, col=1, row=row)
    fig.update_yaxes(title=ylabel, col=1, row=row)
    fig.update_xaxes(title="MEAN", col=2, row=row)
    fig.update_yaxes(title="DELTA", col=2, row=row)

    # set the colormap
    if color is None:
        color = np.tile("none", len(xarr))
    pmap = pcolors.qualitative.Plotly
    colmap = np.unique(color.astype(str))
    colmap = [(i, color == i, k) for i, k in zip(colmap, pmap)]

    # add the scatter points to the regression plot
    for name, idx, col in colmap:
        fig.add_trace(
            row=row,
            col=1,
            trace=go.Scatter(
                x=xarr[idx],
                y=yarr[idx],
                mode="markers",
                marker_color=col,
                showlegend=color is not None,
                opacity=0.5,
                name=name,
                legendgroup=name,
            ),
        )

    # add the identity line to the regression plot
    ymin = min(np.min(yarr), np.min(xarr))
    ymax = max(np.max(yarr), np.max(xarr))
    fig.add_trace(
        row=row,
        col=1,
        trace=go.Scatter(
            x=[ymin, ymax],
            y=[ymin, ymax],
            mode="lines",
            line_dash="dash",
            line_color=pmap[len(colmap)],
            name="IDENTITY LINE",
            legendgroup="IDENTITY LINE",
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
        row=row,
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
    for name, idx, col in colmap:
        fig.add_trace(
            row=row,
            col=2,
            trace=go.Scatter(
                x=means[idx],
                y=diffs[idx],
                mode="markers",
                marker_color=col,
                showlegend=False,
                opacity=0.5,
                name=name,
                legendgroup=name,
            ),
        )

    # plot the bias
    x_idx = np.argsort(means)
    x_bias = means[x_idx]
    f_bias = np.polyfit(means, diffs, 1)
    y_bias = np.polyval(f_bias, x_bias)
    fig.add_trace(
        row=row,
        col=2,
        trace=go.Scatter(
            x=x_bias,
            y=y_bias,
            mode="lines",
            line_color="black",
            line_dash="dash",
            name="BIAS",
            opacity=0.8,
        ),
    )
    chrs = np.max([len(str(i).split(".")[0]) for i in f_bias] + [5])
    fig.add_annotation(
        row=row,
        col=2,
        x=x_bias[0],
        y=y_bias[0],
        text=f"y={str(f_bias[0])[:chrs]}x {str(f_bias[1])[:chrs]}",
        textangle=np.sign(f_bias[0]) * np.degrees(np.arctan(f_bias[0])),
        showarrow=False,
        xanchor="left",
        align="left",
        valign="bottom",
        font=dict(
            family="sans serif",
            size=12,
            color="black",
        ),
    )
    fig.add_annotation(
        row=row,
        col=2,
        x=xrng[-1],
        y=bias,
        text=f"{bias:0.2f}",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(
            family="sans serif",
            size=12,
            color="black",
        ),
    )

    # plot the limits of agreement
    fig.add_trace(
        row=row,
        col=2,
        trace=go.Scatter(
            x=[xrng[0], xrng[1]],
            y=[loalow, loalow],
            mode="lines",
            line_color="black",
            line_dash="dashdot",
            name=loa_lbl,
            legendgroup=loa_lbl,
            opacity=0.3,
            showlegend=True,
        ),
    )
    fig.add_trace(
        row=row,
        col=2,
        trace=go.Scatter(
            x=[xrng[0], xrng[1]],
            y=[loasup, loasup],
            mode="lines",
            line_color="black",
            line_dash="dashdot",
            name=loa_lbl,
            legendgroup=loa_lbl,
            opacity=0.3,
            showlegend=False,
        ),
    )

    fig.add_annotation(
        row=row,
        col=2,
        x=xrng[-1],
        y=loalow,
        text=f"{loalow:0.2f}",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(
            family="sans serif",
            size=12,
            color="black",
        ),
        name=loa_lbl,
    )

    fig.add_annotation(
        row=row,
        col=2,
        x=xrng[-1],
        y=loasup,
        text=f"{loasup:0.2f}",
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(
            family="sans serif",
            size=12,
            color="black",
        ),
        name=loa_lbl,
    )

    return go.FigureWidget(data=fig.data, layout=fig.layout)
