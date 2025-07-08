"""
opensim.read

read specific opensim file formats such as .trc and .mot
extensions.

Functions
---------
read_trc
    read the data contained in a tdf file.

read_mot
    read the data contained in a emt file.
"""

__all__ = ["read_trc", "read_mot"]


#! IMPORTS


import numpy as np
import pandas as pd

from ...utils import assert_file_extension


#! FUNCTIONS


def read_trc(
    file: str,
):
    """
    Return the readings from a .trc file as dicts of 3D objects.

    Parameters
    ----------
    file: str
        an existing tdf path.

    Returns
    -------
    trc: pandas.DataFrame
        a dict where each key is a specific 3D object contained in the trc file.
    """

    # read the file
    assert_file_extension(file, "trc")
    with open(file, "r", encoding="utf-8") as buf:
        lines = [i[:-1].split("\t") for i in buf.readlines()]
    lines = [[i for i in line if i != ""] for line in lines]

    # get the unit of measurement
    unit = lines[2][4]

    # get the data
    data = np.array(lines[6:])
    vals = data[:, 2:].astype(float)
    indx = pd.Index(data[:, 1].astype(float), name="Time [s]")
    cols = [i for i in lines[3][2:] if i != ""]
    cols = [cols, ["X", "Y", "Z"], [unit]]
    names = ["OBJECT", "DIMENSION", "UNIT"]
    cols = pd.MultiIndex.from_product(cols, names=names)

    return pd.DataFrame(vals, indx, cols)


def read_mot(
    file: str,
):
    """
    Return the readings from a .mot file as dicts of 3D objects.

    Parameters
    ----------
    file: str
        an existing tdf path.

    Returns
    -------
    mot: dict[str, pandas.DataFrame]
        a dict where each key is a specific 3D object contained in the mot file.
    """

    # read the file
    assert_file_extension(file, "mot")
    with open(file, "r", encoding="utf-8") as buf:
        lines = [i[:-1].split("\t") for i in buf.readlines()]

    # get the data
    idx = np.where([i[0] == "endheader" for i in lines])[0][0]
    data = np.array(lines[idx + 1 :])

    # get the headers the time index and data values
    headers = np.atleast_2d([i.split("_") for i in data[0][1:]]).T
    for i in np.arange(headers.shape[1]):
        if headers[2, i][0] == "p":
            headers[1, i] = "origin"
            headers[2, i] = headers[2, i][-1]
        elif headers[2, i][0] == "v":
            headers[2, i] = headers[2, i][-1]
        headers[1, i] = headers[1, i].upper()
        headers[2, i] = headers[2, i].upper()
    names = ["OBJECT", "QUANTITY", "DIMENSION"]
    headers = pd.MultiIndex.from_arrays(headers, names=names)
    values = np.array(data[1:]).astype(float)[:, 1:]
    index = pd.Index(np.array(data[1:])[:, 0], name="Time [s]")

    return pd.DataFrame(values, index, headers)
