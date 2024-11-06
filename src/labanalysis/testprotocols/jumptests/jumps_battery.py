"""jumps battery module"""

#! IMPORTS


import numpy as np
import plotly.express as px
from pandas import concat
from plotly.graph_objects import FigureWidget
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
    def summary_table(self):
        """return a table with summary stats from all the tests"""
        tab = super().summary_table
        if not any(i == "Side" for i in tab.columns):
            tab.insert(0, "Side", np.tile("Bilateral", tab.shape[0]))
        else:
            tab.loc[tab.Side.isna(), "Side"] = "Bilateral"
        tab.loc[tab.index, "Test"] = tab.Test.map(
            lambda x: x.replace("JumpTest", "") + " Jump"
        )

        # adjust the imbalance measures
        out = []
        for lbl, dfr in tab.groupby("Parameter"):
            if str(lbl).endswith("Imbalance"):
                left = dfr.copy()
                left.loc[left.index, "Side"] = "Left"
                left.loc[left.index, "Best"] = 50 + left.Best
                right = dfr.copy()
                right.loc[right.index, "Side"] = "Right"
                right.loc[right.index, "Best"] = 50 - right.Best
                out += [left, right]
            else:
                out += [dfr]

        return concat(out, ignore_index=True).drop(["Mean", "Std"], axis=1)

    @property
    def summary_plots(self):
        """return a set of plotly FigureWidget for each relevant metric"""
        tab = self.summary_table
        tab.loc[tab.index, "Text"] = tab.Best.map(lambda x: str(x)[:5])
        colors = px.colors.qualitative.Plotly
        sides = np.sort(tab.Side.unique())
        tab.loc[tab.index, "Color"] = tab.Side.map(
            lambda x: colors[np.where(sides == x)[0][0]]
        )
        tab.sort_values("Side", inplace=True)
        out: dict[str, FigureWidget] = {}
        for parameter, dfr in tab.groupby("Parameter"):
            vmin = dfr.Best.min() * 0.9
            vmax = dfr.Best.max() * 1.1
            tests = dfr.Test.unique().flatten().tolist()
            bilateral_legend = False
            monopodalic_legend = False
            fig = make_subplots(
                rows=1,
                cols=len(tests),
                subplot_titles=[i.replace(" ", "<br>") for i in tests],
                y_title=dfr.Unit.values[0],
            )
            for i, test in enumerate(tests):
                dft = dfr.loc[dfr.Test == test]
                fig0 = px.bar(
                    data_frame=dft,
                    x="Side",
                    y="Best",
                    text="Text",
                    color="Side",
                    color_discrete_sequence=np.unique(
                        dft.Color.values.astype(str)
                    ).tolist(),
                )
                if len(dft.Side.unique()) == 1 and not bilateral_legend:
                    show_legend = True
                    bilateral_legend = True
                elif len(dft.Side.unique()) == 2 and not monopodalic_legend:
                    show_legend = True
                    monopodalic_legend = True
                else:
                    show_legend = False
                fig0.update_traces(
                    legend="legend",
                    showlegend=show_legend,
                    width=1.3,
                )
                for trace in fig0.data:
                    fig.add_trace(row=1, col=i + 1, trace=trace)
            fig.update_yaxes(title="", visible=False, range=[vmin, vmax])
            fig.update_yaxes(visible=True, col=1)
            fig.update_xaxes(title="", showticklabels=False, matches=None)
            fig.update_layout(title=parameter)
            out[str(parameter)] = FigureWidget(fig)

        return out
