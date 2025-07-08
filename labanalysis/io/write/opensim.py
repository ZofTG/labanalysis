"""
io.write.opensim

write specific opensim file formats such as .trc and .mot
extensions.

Functions
---------
write_trc
    read the data contained in a tdf file.

write_mot
    read the data contained in a emt file.
"""

__all__ = ["write_trc", "write_mot"]


#! IMPORTS


from itertools import product

import numpy as np
import pandas as pd

from ...utils import assert_file_extension, check_entry, check_writing_file


#! FUNCTIONS


def write_trc(
    file: str,
    dfr: pd.DataFrame,
):
    """
    Write the provided data into a .trc file.

    Parameters
    ----------
    file: str
        an existing tdf path.

    dfr: pd.DataFrame
        the dataframe containing the data to be stored. They must have:
            - a time-based index
            - columns provided as MultiIndex with the following levels:
                1. list of points
                2. ['X', 'Y', 'Z'] for each first level.
                3. [mm] as unit of measurement for each sublevel.
    """
    raise NotImplementedError

    # check the file
    assert_file_extension(file, "trc")
    filename = check_writing_file(file)

    # check the entries and merge the data
    mask = np.atleast_2d(list(product(["X", "Y", "Z"], ["mm"]))).astype(str)
    check_entry(dfr, mask)

    # prepare the data
    out = dfr.copy()
    col = out.columns.to_frame().iloc[:, :2]
    for i, lbl in enumerate(np.unique(col.iloc[:, 0].values.astype(str))):
        loc = np.where(col == lbl)[0]
        col.iloc[loc, 1] = [k + str(i + 1) for k in col.iloc[loc, 1]]  # type: ignore
        col.iloc[loc, 0] = [col.iloc[loc[0], 0], "", ""]  # type: ignore
    col = col.T.values.astype(str).tolist()
    col[0] = ["Frame#", "Time"] + col[0]
    col[1] = ["", ""] + col[1]
    val = np.atleast_2d(dfr.index.to_numpy()).T
    val = np.concatenate([val, out.values.astype(float)], axis=1)
    val = val.astype(str)
    val = [np.atleast_2d(np.arange(dfr.shape[0])).T.astype(str), val]
    val = np.concatenate(val, axis=1)

    # prepare the output string
    sampf = int(round(1 / np.mean(np.diff(dfr.index.to_numpy().astype(float)))))
    opt = {
        "DataRate": sampf,
        "CameraRate": sampf,
        "NumFrames": dfr.shape[0],
        "NumMarkers": len(np.unique(dfr.columns.get_level_values(0))),
        "Units": "mm",
        "OrigDataRate": sampf,
        "OrigDataStartFrame": 1,
        "OrigNumFrames": dfr.shape[0],
    }
    txt = ["\t".join(["PathFileType", "4", "(X/Y/Z)", file])]
    txt += ["\t".join(list(opt.keys()))]
    txt += ["\t".join([str(i) for i in opt.values()])]
    txt += ["\t".join(i) for i in col]
    txt += ["\t"]
    txt += ["\t".join(i) for i in val]
    txt = "\n".join(txt) + "\n"

    # store the data
    with open(filename, "w", encoding="utf-8") as buf:
        buf.write(txt)


def write_mot(
    file: str,
    dfr: pd.DataFrame,
):
    """
    write the provided data into a .mot file.

    Parameters
    ----------
    file: str
        the file where the data have to be stored.

    dfr: pd.DataFrame
        the object to be stored. It must have:
            - a time-based index
            - columns provided as MultiIndex with the following levels:
                1. the list of force objects
                2. ['ORIGIN', 'FORCE', 'TORQUE']
                3. ['X', 'Y', 'Z'] for each first level.
    """
    raise NotImplementedError

    # check the file
    assert_file_extension(file, "mot")
    filename = check_writing_file(file)

    # check the entries and merge the data
    mask = list(product(["ORIGIN", "FORCE", "TORQUE"], ["X", "Y", "Z"]))
    mask = np.atleast_2d(mask).astype(str)
    check_entry(dfr, mask)
    out = []
    forces = np.unique(dfr.columns.get_level_values(0).to_numpy().astype(str))
    for lbl in forces:
        dfi = dfr[lbl]
        col = dfi.columns.to_frame().values.astype(str).T
        for i in np.arange(col.shape[1]):
            col[0, i] = col[0, i].lower()
            if col[0, i] == "force":
                col[1, i] = "v" + col[1, i].lower()
            elif col[0, i] == "origin":
                col[1, i] = "p" + col[1, i].lower()
            else:
                col[1, i] = col[1, i].lower()
        dfi.columns = pd.Index(["_".join([lbl] + i) for i in col.T.tolist()])
        out += [dfi]
    out = pd.concat(out, axis=1)
    out.insert(0, "time", out.index.to_numpy())

    # get the output file txt
    txt = [file]
    txt += ["version=1"]
    txt += [f"nRows={out.shape[0]}"]
    txt += [f"nColumns={out.shape[1]}"]
    txt += [f"inDegrees=yes"]
    txt += [""]
    txt += ["rslib.io.opensim.write.write_mot"]
    txt += [""]
    txt += ["endheader"]
    txt += ["\t".join(out.columns.tolist())]
    txt += ["\t".join(i) for i in out.values.astype(str).tolist()]
    txt = "\n".join(txt)

    # store the data
    with open(filename, "w", encoding="utf-8") as buf:
        buf.write(txt)
