"""
plotting module

a set of functions for standard plots creation.

Functions
---------
plot_comparisons
    A combination of regression and bland-altmann plots which returns a
    Plotly FigureWidget object.

bars_with_normative_bands
    Return a plotly FigureWidget and a dataframe with bars defining values
    and normative data in the background.


"""

#! IMPORTS

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.express.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, ttest_ind, ttest_rel

__all__ = ["plot_comparisons", "bars_with_normative_bands"]


#! FUNCTION


def plot_comparisons(
    xarr: np.ndarray,
    yarr: np.ndarray,
    color: np.ndarray | None = None,
    xlabel: str = "",
    ylabel: str = "",
    confidence: float = 0.95,
    parametric: bool = False,
    figure: go.Figure | go.FigureWidget | None = None,
    row: int = 1,
    showlegend: bool = True,
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

    figure: go.Figure | go.FigureWidget | None (default = None)
        an already existing figure where to add the plot along the passed row

    row: int (default = 1)
        the index of the row where to put the plots

    showlegend: bool (default = True)
        If True show the legend of the figure.
    """

    # generate the figure
    if figure is None:
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
    else:
        fig = figure

    fig.update_xaxes(title=xlabel, col=1, row=row)
    fig.update_yaxes(title=ylabel, col=1, row=row)
    fig.update_xaxes(title="MEAN", col=2, row=row)
    fig.update_yaxes(title="DELTA", col=2, row=row)

    # set the colormap
    if color is None:
        color = np.tile("none", len(xarr))
    pmap = pcolors.qualitative.Plotly
    colmap = np.unique(np.array(color).flatten().astype(str))
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
                showlegend=color is not None and showlegend,
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
            line_color="black",
            name="IDENTITY LINE",
            legendgroup="IDENTITY LINE",
            showlegend=showlegend,
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
            line_dash="dot",
            name="BIAS",
            opacity=0.8,
            showlegend=showlegend,
        ),
    )
    chrs = np.max([len(str(i).split(".")[0]) + 2 for i in f_bias] + [6])
    msg = [f"{i:+}" for i in f_bias]
    msg = f"y = {msg[0][:chrs]}x {msg[1][:chrs]}"
    fig.add_annotation(
        row=row,
        col=2,
        x=x_bias[-1],
        y=y_bias[-1],
        text=msg,
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        standoff=5,
        align="right",
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
            showlegend=showlegend,
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

    if figure is None:
        return go.FigureWidget(data=fig.data, layout=fig.layout)


def bars_with_normative_bands(
    data_frame: pd.DataFrame | None = None,
    xarr: str | np.ndarray | list = np.array([]),
    yarr: str | np.ndarray | list = np.array([]),
    patterns: str | np.ndarray | list | None = None,
    orientation: Literal["h", "v"] = "v",
    unit: str | None = None,
    intervals: pd.DataFrame = pd.DataFrame(),
):
    """
    Return a plotly FigureWidget and a dataframe with bars defining values
    and normative data in the background.

    Parameters
    ----------
    data_frame : pd.DataFrame | None, optional
        the dataframe containing the data, by default None

    xarr : str | np.ndarray | list, optional
        the xaxis label in data_frame or the array defining the xaxes labels.
        Please note that if data_frame is provided, xarr must be a str defining
        a column of the provided dataframe. by default np.array([])

    yarr : str | np.ndarray | list, optional
        the yaxis label in data_frame or the array defining the yaxes labels.
        Please note that if data_frame is provided, yarr must be a str defining
        a column of the provided dataframe. by default np.array([])

    patterns : str | np.ndarray | list | None, optional
        the column in data_frame defining the bar patterns or the array
        defining the  labels. Please note that if data_frame is provided,
        patterns must be a str defining a column of the provided dataframe.
        by default np.array([])

    orientation : Literal["h", "v"], optional
        the bars orientation, by default "v"

    unit: str | None, optional
        The unit of measurement of the bars.

    intervals: pd.DataFrame, optional
            all the normative intervals. The dataframe must have the following
            columns:

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
    fig: FigureWidget
        a plotly FigureWidget instance

    dfr: DataFrame
        the dataframe used to generate the figure
    """

    #  check the input data
    def _as_array(obj: object, lbl: str):
        """check if obj is a numeric 1D numpy array"""
        if not isinstance(lbl, str):
            msg = "if data_frame is provided, xarr, yarr, facet_col and "
            msg += "facet_row must be strings denoting a single row in"
            msg += " data_frame."
            raise ValueError(msg)
        msg = f"{lbl} must be a 1D array/list of int or float."
        if not isinstance(obj, (np.ndarray, list)):
            raise ValueError(msg)
        arr = np.array(obj) if isinstance(obj, list) else obj
        if arr.ndim > 1:
            raise ValueError(msg)
        return arr

    def _is_part(dfr: pd.DataFrame, lbl: object):
        """check if the lbl is a column of dfr"""
        msg = f"{lbl} must be a string defining one column of data_frame."
        if not isinstance(lbl, str):
            raise ValueError(msg)
        if not any([i == lbl for i in dfr.columns]):
            raise ValueError(msg)

    def _get_rank(x: float | int, intervals: pd.DataFrame):
        """return the rank corresponding to the x value"""
        if intervals.shape[0] > 0:
            for row in np.arange(intervals.shape[0]):
                rnk, low, upp, clr = intervals.iloc[row].values.flatten()[-4:]
                if x >= low and x <= upp:
                    return str(rnk), str(clr)
        return None, None

    def _format_value(x: float | int, unit: str | None):
        """return formatted value with unit of measurement"""
        return str(x)[:5] + ("" if unit is None else f" {unit}")

    if data_frame is None:
        xlbl = "X"
        ylbl = "Y"
        dfr = {xlbl: _as_array(xarr, "xarr"), ylbl: _as_array(yarr, "yarr")}
        if not np.diff([len(v) for v in dfr.values()]).sum() == 0:
            raise ValueError("all the provided arrays must have the same length.")
        dfr = pd.DataFrame(dfr)
    else:
        if not isinstance(data_frame, pd.DataFrame):
            msg = "data_frame must be None or a valid pandas DataFrame."
            raise ValueError(msg)
        _is_part(data_frame, xarr)
        xlbl = str(xarr)
        _is_part(data_frame, yarr)
        ylbl = str(yarr)
        dfr = data_frame

    if patterns is not None:
        if data_frame is None:
            dfr.insert(0, "Pattern", _as_array(patterns, "patterns"))
            plbl = "Pattern"
        else:
            _is_part(dfr, patterns)
            plbl = str(patterns)
    else:
        plbl = None

    if not isinstance(orientation, str) or orientation not in ["h", "v"]:
        raise ValueError('orientation must be "h" or "v".')

    if unit is not None:
        if not isinstance(unit, str):
            raise ValueError("unit must be a str object or None.")

    # check the intervals
    columns = ["Rank", "Lower", "Upper", "Color"]
    msg = "intervals must be a pandas.DataFrame containing the "
    msg += "following columns: " + str(columns)
    msg2 = "Lower and Upper columns must contain only int or float-like values."
    if not isinstance(intervals, pd.DataFrame):
        raise ValueError(msg)
    if intervals.shape[0] > 0:
        for col in columns:
            if col not in intervals.columns.tolist():
                raise ValueError(msg)
            if col in ["Lower", "Upper"]:
                try:
                    _ = intervals[col].astype(float)
                except Exception:
                    raise ValueError(msg2)

    if orientation == "h":
        dfr.loc[dfr.index, ["_Text"]] = dfr[xlbl].map(lambda x: _format_value(x, unit))
        amp = dfr[xlbl].abs().max() * 2
        rng = [-amp, amp]
        rank_arr = dfr[xlbl]
    else:
        dfr.loc[dfr.index, ["_Text"]] = dfr[ylbl].map(lambda x: _format_value(x, unit))
        rng = [float(dfr[ylbl].min()) * 0.9, float(dfr[ylbl].max()) * 1.1]
        rank_arr = dfr[ylbl]
    for i, v in enumerate(rank_arr):
        index = dfr.index[i]
        rnk, clr = _get_rank(v, intervals)
        dfr.loc[index, ["_Rank"]] = rnk
        dfr.loc[index, ["_Color"]] = clr

    # get the output figure
    fig = px.bar(
        data_frame=dfr.reset_index(drop=True),
        x=xlbl,
        y=ylbl,
        pattern_shape=plbl,
        orientation=orientation,
        text="_Text",
        barmode="stack" if plbl is None else "group",
        template="simple_white",
    )

    # add the intervals
    if intervals.shape[0] > 0:
        rnks = []
        for row in np.arange(intervals.shape[0]):
            rnk, low, upp, clr = intervals.iloc[row].values.flatten()[-4:]
            if not np.isfinite(low):
                low = rng[0]
            if not np.isfinite(upp):
                upp = rng[1]
            if rnk not in rnks:
                showlegend = True
                rnks += [rnk]
            else:
                showlegend = False
            if orientation == "h":
                fig.add_vrect(
                    x0=low,
                    x1=upp,
                    name=rnk,
                    showlegend=showlegend,
                    fillcolor=clr,
                    line_width=0,
                    opacity=0.1,
                    legendgroup="norms",
                    legendgrouptitle_text="Normative data",
                )
            else:
                fig.add_hrect(
                    y0=low,
                    y1=upp,
                    name=rnk,
                    showlegend=showlegend,
                    fillcolor=clr,
                    line_width=0,
                    opacity=0.1,
                    legendgroup="norms",
                    legendgrouptitle_text="Normative data",
                )

    # update the layout
    fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[1]))
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    if orientation == "v":
        fig.update_yaxes(title="" if unit is None else unit, range=rng)
        fig.update_xaxes(title="")
    if orientation == "h":
        fig.update_yaxes(title="")
        fig.update_xaxes(title="" if unit is None else unit, range=rng)

    if intervals.shape[0] > 0:
        if plbl is not None:
            for pattern in dfr[plbl].unique():
                for trace in fig.data:
                    if trace["name"] == pattern:  # type: ignore
                        clrs = dfr.loc[dfr[plbl] == pattern, "_Rank"].values  # type: ignore
                        clrs = clrs.astype(str)
                        trace.update(  # type: ignore
                            marker_color=clrs,
                            marker_line_color=clrs,
                        )
        else:
            colors = dfr["_Color"].values.tolist()
            fig.update_traces(marker_color=colors, marker_line_color=colors)
    else:
        colors = pcolors.qualitative.Plotly[0]
        fig.update_traces(marker_color=colors, marker_line_color=colors)
    hover_tpl = f"<i>{xlbl}</i>: " + "%{x}<br>" + f"<i>{ylbl}</i>: " + "%{y}"
    fig.update_traces(
        marker_cornerradius="30%",
        marker_line_width=3,
        marker_pattern_fillmode="replace",
        textposition="outside",
        hovertemplate=hover_tpl,
        opacity=1,
    )

    return go.FigureWidget(fig), dfr
