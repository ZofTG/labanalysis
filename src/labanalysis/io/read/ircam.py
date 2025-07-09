"""
read and write image streams from npz files
"""

__all__ = ["read_npz"]


#! IMPORTS


from typing import Any
import numpy as np
from datetime import datetime


#! FUNCTIONS


def read_npz(filename: str):
    """
    Return the readings from a .npz file containing a stream of images

    Parameters
    ----------
    filename: str
        an existing npz file.

    Returns
    -------
    stream: dict[datetime.datetime, np.ndarray[Any, np.dtype[Any]]]
        a dataframe containing the input data
    """
    assert isinstance(filename, str), "filename must be a str instance."
    assert filename.endswith(".npz"), "filename must have .npz extension."
    try:
        out: dict[datetime, np.ndarray]
        out = dict(np.load(filename))
    except Exception:
        raise ValueError(f"Data in {filename} cannot be converted into dict.")
    msg = f"all keys in the imported dict must be {datetime} objects."
    assert all(isinstance(i, datetime) for i in out), msg
    msg = f"all values in the imported dict must be {np.ndarray} objects."
    assert all(isinstance(i, np.ndarray) for i in out.values()), msg

    return out
