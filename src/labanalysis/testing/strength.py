"""strength testing module"""

#! IMPORTS


from math import ceil
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from labio.read.biostrength import Product

from ..signalprocessing import butterworth_filt, find_peaks
from ..utils import magnitude


#! CONSTANTS


__all__ = ["Isokinetic1RM"]


#! CLASSES


class Isokinetic1RM:
    """Isokinetic Test 1RM instance"""

    _repetitions: dict[str, Product]
    _product: Product

    @property
    def repetitions(self):
        """return the tracked repetitions data"""
        return self._repetitions

    @property
    def rom1(self):
        """return the ending position of the repetitions"""
        roms = []
        for dfr in self.repetitions.values():
            pos: list[float] = dfr.position_m  # type: ignore
            roms += [float(np.max(pos))]
        if len(roms) == 0:
            return None
        return float(np.mean(roms))

    @property
    def rom0(self):
        """return the starting position of the repetitions"""
        roms = []
        for dfr in self.repetitions.values():
            pos: list[float] = dfr.position_m  # type: ignore
            roms += [float(np.min(pos))]
        if len(roms) == 0:
            return None
        return float(np.mean(roms))

    @property
    def rom(self):
        """return the Range of Movement"""
        if self.rom0 is None or self.rom1 is None:
            return None
        return self.rom1 - self.rom0

    @property
    def peak_load(self):
        """return the peak load recorded"""
        loads = []
        for dfr in self.repetitions.values():
            load: list[float] = dfr.load_kgf  # type: ignore
            loads += [float(np.max(load))]
        if len(loads) == 0:
            return None
        return float(np.max(loads))

    def calculate_1rm(self):
        """return the predicted 1RM"""
        load = self.peak_load
        if load is None:
            return None
        a, b = self.product.rm1_coefs  # type: ignore
        return a * load + b

    @property
    def product(self):
        return self._product

    def _find_repetitions(
        self,
        time: np.ndarray[Literal[1], np.dtype[np.float_]],
        force: np.ndarray[Literal[1], np.dtype[np.float_]],
        position: np.ndarray[Literal[1], np.dtype[np.float_]],
    ):
        """
        private method used to extract the samples defining the start
        and stop of each repetition according to position and speed.

        Parameters
        ----------
        time : np.ndarray[Any, np.dtype[np.float_]]
            the force readings resulting from processed biodrive data

        force : np.ndarray[Any, np.dtype[np.float_]]
            the force readings resulting from processed biodrive data

        position : np.ndarray[Any, np.dtype[np.float_]]
            the position readings resulting from processed biodrive data

        Results
        -------
        reps: np.ndarray[Any, np.dtype[np.float_]]
            a Nx2 array where each row denotes a repetition and the columns
            respectively the starting and stopping samples of the rep.
        """
        prange = np.max(position) - np.min(position)
        thr1 = float(0.75 * prange + np.min(position))
        thr2 = float(0.1 * prange + np.min(position))
        dis = int(round(1 / np.mean(np.diff(time))))
        ppks = find_peaks(position, thr1, dis)
        mmns = find_peaks(-position, -thr1)
        reps = []
        for ppk in ppks:
            if len(reps) == 0:
                start = np.where(position[:ppk] < thr2)[0]
                start = start[-1] if len(start) > 0 else np.argmin(position[:ppk])
            else:
                start = mmns[mmns < ppk][-1]
            stop = ppk
            reps += [[start, stop]]
        reps = np.atleast_2d(reps).astype(int)

        # remove those reps having same start or stop
        for i in np.arange(2):
            reps = reps[np.unique(reps[:, i], return_index=True)[1]]

        # get at most the 3 reps with the highest force output
        idx = np.argsort([np.max(force[np.arange(i[0], i[1])]) for i in reps])
        reps = reps[np.sort(idx[-3:]), :]

        return reps

    def _get_bounds(
        self,
        xval0: float | int,
        xval1: float | int,
    ):
        """return the ticks for one axis defined by the make_figure method."""
        magx1 = magnitude(xval0)
        magx2 = magnitude(xval1)
        magx = 10.0 ** np.min([magx1, magx2, 0])
        xbounds = np.array([(xval0 / magx) // 5, ceil(xval1 / magx / 5)])
        xbounds *= 5 * magx
        return np.linspace(*xbounds, 6)  # type: ignore

    def make_figure(self):
        """generate a figure representing the test data"""
        # figure
        fig, axs = plt.subplots(2, 1)
        cmap = plt.get_cmap("tab10")  # type: ignore

        # add data
        axs[0].plot(
            self.product.time_s,
            self.product.position_m,
            linewidth=1,
            color=cmap(0),
        )
        axs[1].plot(
            self.product.time_s,
            self.product.load_kgf,
            linewidth=1,
            color=cmap(0),
        )
        reps = list(self.repetitions.values())
        for i, rep in enumerate(reps):
            time_arr: list[float] = rep.time_s  # type: ignore
            pos_arr: list[float] = rep.position_m  # type: ignore
            for_arr: list[float] = rep.load_kgf  # type: ignore
            axs[0].plot(
                time_arr,
                pos_arr,
                linewidth=2,
                color=cmap(i + 1),
                label=f"REP {int(i + 1):02d}",
            )

            axs[0].plot(
                [time_arr[0], time_arr[0]],
                [np.min(pos_arr), np.max(pos_arr)],
                linewidth=0.5,
                linestyle="dashed",
                color=cmap(i + 1),
                label=f"REP {int(i + 1):02d}",
            )
            axs[0].plot(
                [time_arr[-1], time_arr[-1]],
                [np.min(pos_arr), np.max(pos_arr)],
                linewidth=0.5,
                linestyle="dashed",
                color=cmap(i + 1),
                label=f"REP {int(i + 1):02d}",
            )
            axs[1].plot(
                time_arr,
                for_arr,
                linewidth=2,
                color=cmap(i + 1),
                label=f"REP {int(i + 1):02d}",
            )
            axs[1].plot(
                [time_arr[0], time_arr[0]],
                [np.min(for_arr), np.max(for_arr)],
                linewidth=0.5,
                linestyle="dashed",
                color=cmap(i + 1),
                label=f"REP {int(i + 1):02d}",
            )
            axs[1].plot(
                [time_arr[-1], time_arr[-1]],
                [np.min(for_arr), np.max(for_arr)],
                linewidth=0.5,
                linestyle="dashed",
                color=cmap(i + 1),
                label=f"REP {int(i + 1):02d}",
            )

        # setup the layout
        time0: float = reps[0].time_s[0]  # type: ignore
        time1: float = reps[-1].time_s[-1]  # type: ignore
        xticks = self._get_bounds(time0, time1)
        for i, ax in enumerate(axs):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim((xticks[0], xticks[-1]))
            yticks = self._get_bounds(*ax.get_ylim())
            ax.set_ylim((yticks[0], yticks[-1]))
            ax.set_yticks(yticks)
            ax.spines["left"].set_bounds((yticks[0], yticks[-1]))
            if i == 0:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False, labelbottom=False)
                ax.set_ylabel("Position (m)")
            else:
                ax.spines["bottom"].set_bounds((xticks[0], xticks[-1]))
                ax.set_xticks(xticks)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Force (kgf)")

        # render the legend
        headers, labels = axs[0].get_legend_handles_labels()
        headers = [headers[i] for i in np.arange(0, len(headers), 3)]
        labels = [labels[i] for i in np.arange(0, len(labels), 3)]
        plt.figlegend(
            headers,
            labels,
            loc="upper right",
            ncol=len(reps),
            bbox_to_anchor=(1, 0.92),
            frameon=False,
        )

        # apply a tight layout
        plt.tight_layout(rect=(0, 0, 1, 0.93))

        return fig, axs

    def __init__(self, product: Product):
        self._product = product
        fsamp = 1 / np.mean(np.diff(np.array(self._product.time_s)))
        self._product._raw_position_rad = butterworth_filt(
            arr=np.array(self._product.raw_position_rad),
            fcut=1,
            fsamp=fsamp,
            order=6,
            ftype="lowpass",
            phase_corrected=True,
        )
        self._product._raw_torque_nm = butterworth_filt(
            arr=self._product.raw_torque_nm,
            fcut=1,
            fsamp=fsamp,
            order=6,
            ftype="lowpass",
            phase_corrected=True,
        )

        # get the peak load and position (if possible)
        self._repetitions = {}
        if not self.product.is_empty():
            time = np.array(self.product.time_s)
            force = np.array(self.product.load_kgf)
            position = np.array(self.product.position_m)
            if force is not None:
                reps = self._find_repetitions(time, force, position)
                if reps is not None:
                    for start, stop in reps:
                        lbl = f"REP{len(self._repetitions) + 1}"
                        val = self.product.slice(start, stop)
                        self._repetitions[lbl] = val
        reps = list(self.repetitions.values())
        self._peak_load = np.nan
        if len(reps) > 0:
            pks = []
            lcs = []
            for dfr in reps:
                frz: list[float] = dfr.load_kgf  # type: ignore
                pos: list[float] = dfr.position_m  # type: ignore
                loc = np.argmax(frz)
                pks += [frz[loc]]
                lcs += [pos[loc]]
            loc = np.argmax(pks)
            self._peak_load = float(pks[loc])

            # adjust the size of the product data
            tmr = self.product.time_s
            rep0 = reps[0]
            time0: list[float] = rep0.time_s  # type: ignore
            start = time0[0] - (time0[-1] - time0[0])
            start = max(tmr[0], start)  # type: ignore
            rep1 = reps[-1]
            time1: list[float] = rep1.time_s  # type: ignore
            stop = time1[-1] + (time1[-1] - time1[0])
            stop = min(tmr[-1], stop)  # type: ignore
            init = np.where(np.array(self.product.time_s) <= start)[0][-1]
            end = np.where(np.array(self.product.time_s) >= stop)[0][0]
            self._product = self.product.slice(init, end)  # type: ignore
