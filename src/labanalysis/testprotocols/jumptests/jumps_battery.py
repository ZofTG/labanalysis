"""jumps battery module"""

#! IMPORTS

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ...plotting.plotly import bars_with_normative_bands
from ...testing import TestBattery
from .counter_movement_jump import CounterMovementJumpTest
from .side_jump import SideJumpTest
from .single_leg_jump import SingleLegJumpTest
from .squat_jump import SquatJumpTest

__all__ = ["JumpTestBattery"]


#! CLASSES


class JumpTestBattery(TestBattery):
    """
    generate a test battery from jump tests

    Attributes
    ----------
    *tests: SquatJumpTest | CounterMovementJumpTest | SideJumpTest | SingleLegJumpTest
        the list of jump tests to be analysed

    Methods
    -------
    summary:
        Return a summary table reporting the test battery results and dictionary
        with figures representing the outcomes for each investigated parameter.

    save
        a method allowing the saving of the data in an appropriate format.

    load
        a class method to load a LabTest object saved in its own format.
    """

    # methods

    def _make_summary_table(
        self,
        normative_intervals: pd.DataFrame = pd.DataFrame(),
    ):
        """
        return a table highlighting the test summary and a table reporting
        the summary data.

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
        table: pd.DataFrame
            return the summary table
        """
        results = []
        for test in self.tests:
            idx = normative_intervals.Test == test.name
            tab = test._make_summary_table(normative_intervals.loc[idx])
            tab.insert(0, "Test", np.tile(test.name, tab.shape[0]))
            idx = tab.loc[tab.Parameter == "Elevation"]
            best_jump = idx.iloc[idx.Value.argmax()].Jump
            tab = tab.loc[tab.Jump == best_jump]
            if isinstance(tab, pd.Series):
                tab = pd.DataFrame(tab).T
            results += [tab]
        return pd.concat(results, ignore_index=True)

    def _make_summary_plot(
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
        """

        # get the data
        data = self._make_summary_table(normative_intervals)

        # build the figure
        figures: dict[str, go.FigureWidget] = {}
        for parameter, dfp in data.groupby("Parameter"):
            unit = str(data.Unit.values[0])
            dfp.loc[dfp.index, ["Side"]] = dfp.Side.map(
                lambda x: "Bilateral" if not isinstance(x, str) else x
            )
            dfp.loc[dfp.index, ["Text"]] = dfp.Value.map(
                lambda x: str(x)[:5] + " " + unit
            )

            # norms
            if normative_intervals.shape[0] > 0:
                idx = normative_intervals.Parameter == str(parameter)
                norms = normative_intervals.loc[idx]
            else:
                norms = pd.DataFrame()
            for row in np.arange(norms.shape[0]):
                test, param, rnk, low, upp, clr = norms.iloc[row].values
                idx = dfp.loc[dfp.Test == test].index
                vals = dfp.loc[idx].Value.values
                for i, v in zip(idx, vals):
                    if param == str(parameter) and v >= low and v <= upp:
                        dfp.loc[i, ["Rank"]] = rnk
                        dfp.loc[i, ["Color"]] = clr

            # get the output figure
            xlbl = "Value" if "Imbalance" in str(parameter) else "Test"
            ylbl = "Test" if "Imbalance" in str(parameter) else "Value"
            fig = px.bar(
                data_frame=dfp.reset_index(drop=True),
                x=xlbl,
                y=ylbl,
                pattern_shape="Side",
                facet_col="Test",
                orientation="h" if "Imbalance" in str(parameter) else "v",
                text="Text",
                barmode="group",
                template="simple_white",
            )

            # get the test boundaries
            if "Imbalance" in str(parameter):
                low = -dfp.Value.abs().max() * 2
                upp = dfp.Value.abs().max() * 2
            else:
                low = dfp.Value.min() * 0.9
                upp = dfp.Value.max() * 1.1
            lown = norms.Lower.values.astype(float)
            lown = lown[np.isfinite(lown)]
            if len(lown) > 0:
                low = min(low, lown.min())
            uppn = norms.Upper.values.astype(float)
            uppn = uppn[np.isfinite(uppn)]
            if len(uppn) > 0:
                upp = max(upp, uppn.max())

            # add the intervals
            legends = []
            fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
            anns = fig.layout.annotations  # type: ignore
            tests = [v["text"] for v in anns]
            for i in np.arange(norms.shape[0]):
                test, param, rnk, lowb, uppb, clr = norms.iloc[i].values
                if not np.isfinite(lowb):
                    lowb = low
                if not np.isfinite(uppb):
                    uppb = upp
                if rnk not in legends:
                    showlegend = True
                    legends += [rnk]
                else:
                    showlegend = False
                col = int([i for i, v in enumerate(tests) if v == test][0]) + 1
                if "Imbalance" in str(parameter):
                    fig.add_vrect(
                        x0=lowb,
                        x1=uppb,
                        name=rnk,
                        showlegend=showlegend,
                        fillcolor=clr,
                        line_width=0,
                        opacity=0.1,
                        legendgroup="norms",
                        legendgrouptitle_text="Normative data",
                        col=col,  # type: ignore
                    )
                else:
                    fig.add_hrect(
                        y0=lowb,
                        y1=uppb,
                        name=rnk,
                        showlegend=showlegend,
                        fillcolor=clr,
                        line_width=0,
                        opacity=0.1,
                        legendgroup="norms",
                        legendgrouptitle_text="Normative data",
                        col=col,  # type: ignore
                    )

            # update the layout
            if "Imbalance" in str(parameter):
                fig.update_xaxes(
                    matches=None,
                    showticklabels=True,
                    range=[low, upp],
                )
                fig.update_yaxes(
                    matches=None,
                    showticklabels=False,
                    visible=False,
                    title="",
                )
                fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="black",
                    line_width=1,
                )
            else:
                fig.update_yaxes(
                    matches=None,
                    showticklabels=True,
                    range=[0, upp],
                )
                fig.update_yaxes(title=unit, col=1)
                fig.update_xaxes(
                    matches=None,
                    showticklabels=False,
                    visible=False,
                    title="",
                )
            fig.update_layout(title=parameter)

            # update the traces
            for trace in fig.data:
                test = trace["y" if "Imbalance" in str(parameter) else "x"][0]  # type: ignore
                side = trace["name"]  # type: ignore
                clr = dfp.loc[(dfp.Test == test) & (dfp.Side == side)]
                clr = str(clr.Color.values[0])
                clr = px.colors.qualitative.Plotly[0] if clr == "nan" else clr
                trace.update(  # type: ignore
                    marker_color=clr,
                    marker_line_color=clr,
                )
            hover_tpl = f"<i>{xlbl}</i>: " + "%{x}<br>" + f"<i>{ylbl}</i>: " + "%{y}"
            fig.update_traces(
                marker_cornerradius="30%",
                marker_line_width=3,
                marker_pattern_fillmode="replace",
                textposition="outside",
                hovertemplate=hover_tpl,
                opacity=1,
                width=0.35,
            )

            figures[str(parameter)] = go.FigureWidget(fig)

        return figures

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
