"""kinematics module"""

#! IMPORTS


import numpy as np
import pandas as pd

from ..signalprocessing import continuous_batches, fillna
from .events import RunningStep, RunningStride, WalkingStep, WalkingStride


#! CONSTANTS


__all__ = ["find_gait_events"]


#! FUNCTIONS


def find_gait_events(
    lheel: pd.DataFrame,
    ltoe: pd.DataFrame,
    rheel: pd.DataFrame,
    rtoe: pd.DataFrame,
    lmidfoot: pd.DataFrame | None = None,
    rmidfoot: pd.DataFrame | None = None,
    preprocess: bool = True,
    height_thresh: float | int = 0.02,
):
    """
    detect steps and strides from kinematic data

    Parameters
    ----------
    lheel: pd.DataFrame
        Left Heel coordinates.
        A 3D dataframe with X-Y-Z columns and index defining the time of each
        sample.

    ltoe: pd.DataFrame
        Left Toe coordinates.
        A 3D dataframe with X-Y-Z columns and index defining the time of each
        sample.

    rheel: pd.DataFrame
        Right Heel coordinates.
        A 3D dataframe with X-Y-Z columns and index defining the time of each
        sample.

    rtoe: pd.DataFrame
        Right Toe coordinates.
        A 3D dataframe with X-Y-Z columns and index defining the time of each
        sample.

    lmidfoot: pd.DataFrame | None (default = None)
        Left Midfoot coordinates.
        If provided, a 3D dataframe with X-Y-Z columns and index defining the
        time of each sample.

    rmidfoot: pd.DataFrame | None (default = None)
        Right midfoot coordinates.
        If provided, a 3D dataframe with X-Y-Z columns and index defining the
        time of each sample.

    preprocess: bool (default = True)
        If True, the missing samples in kinematic data are filled and the
        resulting signal is smoothed by means of a 6th order, phase-corrected,
        lowpass Butterworth filter with cutoff of 12 Hz.

    height_thresh: float | int (default = 0.02)
        Height from the ground in meters defining the minimum height of the
        markers to be considered as in contact to the ground.

    Returns
    -------
    steps: list[RunningStep | WalkingStep]
        the list of detected steps.

    strides: list[RunningStride | WalkingStride]
        the list of detected strides.
    """

    # check the entries
    vlc = {"lHeel": lheel.copy(), "lToe": ltoe.copy()}
    if lmidfoot is not None:
        vlc["lMidfoot"] = lmidfoot.copy()
    vrc = {"rHeel": rheel.copy(), "rToe": rtoe.copy()}
    if rmidfoot is not None:
        vrc["rMidfoot"] = rmidfoot.copy()
    coords = {**vlc, **vrc}
    time = coords["lHeel"].index.to_numpy()
    for lbl, coord in coords.items():

        # check the presence of X, Y and Z columns
        msg = f"{lbl} must be a pandas DataFrame with ['X', 'Y', 'Z'] columns."
        if not isinstance(coord, pd.DataFrame):
            raise TypeError(msg)
        coord.columns = pd.Index(coord.columns.get_level_values(0))
        if not all(i in coord.columns.tolist() for i in ["X", "Y", "Z"]):
            raise ValueError(msg)

        # check the index
        if not coord.shape[0] == len(time):
            raise ValueError(f"{lbl} must have shape [{len(time)}, 3]")
        tarr = coord.index.to_numpy()
        if np.sum(tarr - time) != 0:
            msg = "time index is not consistent between the input dataframes."
            raise ValueError(msg)

    # check preprocess
    if not isinstance(preprocess, bool):
        raise TypeError("preprocess must be True or False")

    # check helght_thresh
    if not isinstance(height_thresh, (int, float)):
        raise ValueError("height_thresh must be an int or float.")

    # wrap
    for lbl, coord in coords.items():
        coord.columns = pd.MultiIndex.from_product([[lbl], coord.columns])
    coords = pd.concat(list(coords.values()), axis=1)

    # preprocess (if required)
    if preprocess:
        fsamp = 1 / np.mean(np.diff(time))

        # fill missing values
        coords = pd.DataFrame(fillna(coords, n_regressors=6))

        # smooth all marker coordinates
        coords = coords.map(
            laban.butterworth_filt,  # type: ignore
            fcut=12,
            fsamp=fsamp,
            order=6,
            ftype="lowpass",
            phase_corrected=True,
            raw=True,
        )

    # get the vertical coordinates of all relevant markers
    vcoords = coords[[i for i in coords.columns if i[1] == "Z"]].copy()
    vcoords.columns = pd.Index([i[0] for i in vcoords.columns])
    vlc = vcoords[[i for i in vcoords.columns if i[0] == "l"]]
    vlc.columns = pd.Index([i[1:] for i in vlc.columns])
    vrc = vcoords[[i for i in vcoords.columns if i[0] == "r"]]
    vrc.columns = pd.Index([i[1:] for i in vrc.columns])

    # get the instants where heels and toes are on ground
    vlc -= vlc.min(axis=0)
    vrc -= vrc.min(axis=0)

    # get the mean values (they are used for mid-stance detection)
    mlc = vlc.mean(axis=1)
    mrc = vrc.mean(axis=1)

    # get the batches of time with part of the feet on ground
    glc = vlc < height_thresh
    grc = vrc < height_thresh
    blc = continuous_batches(glc.any(axis=1).values.astype(bool))
    brc = continuous_batches(grc.any(axis=1).values.astype(bool))

    # exclude those batches that start at the beginning of the data acquisition
    # or are continuning at the end of the data acquisition
    if len(blc) > 0:
        if blc[0][0] == 0:
            blc = blc[1:]
        if blc[-1][-1] >= len(vlc) - 1:
            blc = blc[:-1]
    if len(brc) > 0:
        if brc[0][0] == 0:
            brc = brc[1:]
        if brc[-1][-1] >= len(vrc) - 1:
            brc = brc[:-1]

    # get the events
    time = lheel.index.to_numpy().astype(float).flatten()
    evts_map = {
        "FS LEFT": np.array([time[i[0]] for i in blc]),
        "FS RIGHT": np.array([time[i[0]] for i in brc]),
        "MS LEFT": np.array([time[np.argmin(mlc.iloc[i]) + i[0]] for i in blc]),
        "MS RIGHT": np.array([time[np.argmin(mrc.iloc[i]) + i[0]] for i in brc]),
        "TO LEFT": np.array([time[i[-1]] for i in blc]),
        "TO RIGHT": np.array([time[i[-1]] for i in brc]),
    }
    evts_val = np.concatenate(list(evts_map.values()))
    evts_lbl = np.concatenate([np.tile(i, len(v)) for i, v in evts_map.items()])
    evts_idx = np.argsort(evts_val)
    evts_val = evts_val[evts_idx]
    evts_side = np.array([i.split(" ")[1] for i in evts_lbl[evts_idx]])
    evts_lbl = np.array([i.split(" ")[0] for i in evts_lbl[evts_idx]])

    # get the steps
    steps = []
    run_seq = ["FS", "MS", "TO", "LD"]
    walk_seq = ["FS", "TO", "MS", "LD"]
    for n in np.arange(0, len(evts_lbl) - 4, 3):
        idx = np.arange(4) + n
        seq = evts_lbl[idx].copy()
        seq[-1] = "LD"
        sides = evts_side[idx].copy()
        vals = evts_val[idx].copy()
        s0 = sides[0]
        if (
            all([i == v for i, v in zip(seq, run_seq)])
            & all(i == s0 for i in sides[:-1])
            & (sides[-1] != s0)
        ):  # running
            steps += [RunningStep(*vals, side=s0.upper())]
        elif (
            all([i == v for i, v in zip(seq, walk_seq)])
            & all(i == s0 for i in sides[2:-1])
            & (sides[1] != s0)
            & (sides[-1] != s0)
        ):  # walking
            steps += [WalkingStep(*vals, side=s0.upper())]

    # get the strides
    strides = []
    for st1, st2 in zip(steps[:-1], steps[1:]):
        if (
            st1.landing_s == st2.foot_strike_s
            and st1.side is not None
            and st2.side is not None
            and st1.side != st2.side
        ):
            if all([isinstance(i, RunningStep) for i in [st1, st2]]):
                strides += [RunningStride(st1, st2)]
            elif all([isinstance(i, WalkingStep) for i in [st1, st2]]):
                strides += [WalkingStride(st1, st2)]

    return steps, strides


"""
    # get biofeedback
    out_list = [
        _get_biofeedback(
            time=ftime,
            vforce=fvrt,  # type: ignore
            copx=cop_x,  # type: ignore
            cycle1=a,
            cycle2=b,
        )
        for a, b in zip(cycles[:-1], cycles[1:])
    ]
    dfr = pd.concat(out_list, ignore_index=True).describe([])

    # initialize the output figure
    lines = {
        "GRF": (ftime, fvrt / np.max(fvrt)),
        "COP<sub>X</sub>": (ftime, cop_x / np.max(abs(cop_x))),  # type: ignore
        **{i: (mtime, v / np.max(v)) for i, v in vcoords.items()},
    }
    fig = go.Figure()
    for i, v in lines.items():
        fig.add_trace(go.Scatter(x=v[0], y=v[1], mode="lines", name=i))

    # plot the events
    labels = list(pool.values())
    times = list(pool.keys())
    events = np.arange(len(pool))
    phases = {}
    for i0, i1 in zip(events[:-1], events[1:]):
        side0, lbl0 = labels[i0].split(" ")
        side1, lbl1 = labels[i1].split(" ")
        if lbl0 == "foot-strike" and lbl1 == "mid-stance" and side0 == side1:
            name = " ".join([side0, "load-response"])
        elif lbl0 == "foot-strike" and lbl1 == "toe-off" and side0 != side1:
            name = " ".join([side0, "double-support"])
        elif lbl0 == "mid-stance" and lbl1 == "toe-off" and side0 == side1:
            name = " ".join([side0, "propulsion"])
        elif lbl0 == "toe-off" and lbl1 == "foot-strike" and side0 != side1:
            name = " ".join([side0, "flight"])
        elif lbl0 == "toe-off" and lbl1 == "mid-stance" and side0 != side1:
            name = " ".join([side0, "single-support"])
        elif lbl0 == "mid-stance" and lbl1 == "foot-strike" and side0 != side1:
            name = " ".join([side0, "double-support"])
        else:
            name = ""
        if name != "":
            if not any(i == name for i in phases):
                phases[name] = [(times[i0], times[i1])]
            else:
                phases[name] += [(times[i0], times[i1])]
    clrs = list(phases.keys())
    clrs = {v: cmap[i + len(lines)] for i, v in enumerate(clrs)}
    for phase, evts in phases.items():
        for t0, t1 in evts:
            fig.add_vrect(
                x0=t0,
                x1=t1,
                line_width=0,
                fillcolor=clrs[phase],
                opacity=0.2,
                name=phase,
                legendgroup=phase,
            )

    return dfr, fig
"""
