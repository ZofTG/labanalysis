"""
utils

module containing several utilities that can be used for multiple purposes.

Functions
---------
magnitude
    get the order of magnitude of a numeric scalar value according to the
    specified base.

get_files
    get the full path of the files contained within the provided folder
    (and optionally subfolders) having the provided extension.

split_data
    get the indices randomly separating the input data into subsets according
    to the given proportions.
"""

#! IMPORTS


import os
from typing import Any

import numpy as np

__all__ = [
    "magnitude",
    "get_files",
    "split_data",
]


#! FUNCTIONS


def magnitude(
    value: int | float,
    base: int | float = 10,
):
    """
    return the order in the given base of the value

    Parameters
    ----------
        value: int | float
            the value to be checked

        base:int | float=10
            the base to be used to define the order of the number

    Returns
    -------
        mag float
            the number required to elevate the base to get the value
    """
    if value == 0 or base == 0:
        return int(0)
    else:
        val = np.log(abs(value)) / np.log(base)
        if val < 0:
            return -int(np.ceil(-val))
        return int(np.ceil(val))


def get_files(
    path: str,
    extension: str = "",
    check_subfolders: bool = False,
):
    """
    list all the files having the required extension in the provided folder
    and its subfolders (if required).

    Parameters
    ----------
        path: str
            a directory where to look for the files.

        extension: str
            a str object defining the ending of the files that have to be
            listed.

        check_subfolders: bool
            if True, also the subfolders found in path are searched,
            otherwise only path is checked.

    Returns
    -------
        files: list
            a list containing the full_path to all the files corresponding
            to the input criteria.
    """

    # output storer
    out = []

    # surf the path by the os. walk function
    for root, _, files in os.walk(path):
        for obj in files:
            if obj[-len(extension) :] == extension:
                out += [os.path.join(root, obj)]

        # handle the subfolders
        if not check_subfolders:
            break

    # return the output
    return out


def split_data(
    data: np.ndarray[Any, np.dtype[np.float64]],
    proportion: dict[str, float],
    groups: int,
):
    """
    get the indices randomly separating the input data into subsets according
    to the given proportions.

    Note
    ----
    the input array is firstly divided into quantiles according to the groups
    argument. Then the indices are randomly drawn from each subset according
    to the entered proportions. This ensures that the resulting groups
    will mimic the same distribution of the input data.

    Parameters
    ----------
    data : np.ndarray[Any, np.dtype[np.float64]]
        a 1D input array

    proportion : dict[str, float]
        a dict where each key contains the proportion of the total samples
        to be given. The proportion must be a value within the (0, 1] range
        and the sum of all entered proportions must be 1.

    groups : int
        the number of quantilic groups to be used.

    Returns
    -------
    splits: dict[str, np.ndarray[Any, np.dtype[np.int64]]]
        a dict with the same keys of proportion, which contains the
        corresponding indices.
    """

    # get the grouped data by quantiles
    nsamp = len(data)
    if groups <= 1:
        grps = [np.arange(nsamp)]
    else:
        qnts = np.quantile(data, np.linspace(0, 1, groups + 1)[1:])
        grps = np.digitize(data, qnts, right=True)
        idxs = np.arange(nsamp)
        grps = [idxs[grps == i] for i in np.arange(groups)]

    # split each group
    dss = {i: [] for i in proportion.keys()}
    for grp in grps:
        arr = np.random.permutation(grp)
        nsamp = len(arr)
        for i, k in enumerate(list(dss.keys())):
            if i < len(proportion) - 1:
                n = int(np.round(nsamp * proportion[k]))
            else:
                n = len(arr)
            dss[k] += [arr[:n]]
            arr = arr[n:]

    # aggregate
    return {i: np.concatenate(v) for i, v in dss.items()}
