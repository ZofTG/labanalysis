"""
io.read.cosmed

module containing functions for reading Cosmed Omnia export files.

Constants
---------
COSMED_DATETIME_FORMAT
    the datetime format used within the Cosmed Omnia file export.

Functions
---------
read_cosmed_xlsx
    read .xlsx files generated trough the Cosmed Omnia software.
"""

__all__ = ["COSMED_DATETIME_FORMAT", "read_cosmed_xlsx"]


#! IMPORTS


import os
import datetime
import pandas as pd

from ...tests import Participant


#! CONSTANTS


COSMED_DATETIME_FORMAT = "%d/%m/%Y-%H:%M:%S"


#! FUNCTIONS


def _get_participant(
    raw: pd.DataFrame,
):
    """
    return the Participant object read by a Cosmed Omnia excel export.

    Parameters
    ----------
    raw: pd.DataFrame
        the input dataframe

    Returns
    -------
    prt: Participant
        a Participant instance.
    """
    surname, name, gender, _, height, weight, birth_date = raw.iloc[:7, 1]
    birth_date = birth_date + "-00:00:00"
    birth_date = datetime.datetime.strptime(birth_date, COSMED_DATETIME_FORMAT)
    birth_date = birth_date.date()
    test_date = raw.columns.to_numpy()[4] + "-00:00:00"
    test_date = datetime.datetime.strptime(test_date, COSMED_DATETIME_FORMAT)
    test_date = test_date.date()
    return Participant(
        surname=surname,
        name=name,
        gender=gender,
        height=height,
        weight=weight,
        birthdate=birth_date,
        recordingdate=test_date,
    )


def _get_data(
    raw: pd.DataFrame,
):
    """
    return the Participant object read by a Cosmed Omnia excel export.

    Parameters
    ----------
    raw: pd.DataFrame
        the input dataframe

    Returns
    -------
    data: pd.DataFrame
        a dataframe containing the data
    """

    # get the data values
    out = raw.iloc[2:, 10:]

    # update the column headers
    labels = raw.columns[10:].to_numpy().astype(str)
    units = [i.replace("---", "") for i in raw.iloc[0, 10:].values.astype(str)]
    cols = pd.MultiIndex.from_tuples([(i, j) for i, j in zip(labels, units)])
    out.columns = cols

    # update the indices
    date0 = raw.columns.to_numpy()[4]
    time0 = str(raw.iloc[0, 4])
    datetime0 = "-".join([date0, time0])
    datetime0 = datetime.datetime.strptime(datetime0, COSMED_DATETIME_FORMAT)
    unit = raw.iloc[0, 9]
    idxs = []
    for time in raw.iloc[2:, 9].values:
        hour, minute, second = [int(i) for i in str(time).split(":")]
        tdelta = datetime.timedelta(hours=hour, minutes=minute, seconds=second)
        idxs += [datetime0 + tdelta]
    out.index = pd.Index(idxs, name=f"Time [{unit}]")

    return out


def read_cosmed_xlsx(
    path: str,
):
    """
    Return the readings from a .xlsx file containing cosmed Omnia xlsx file.

    Parameters
    ----------
    path: str
        an existing xlsx path.

    Returns
    -------
    data: pandas.DataFrame
        a dataframe containing the input data

    participant: rslib.utils.Participant
        an instance of the Participant class contaning all the relevant data
        about the participant having performed the acquisition.
    """

    # check the validity of the entered path
    assert isinstance(path, str), "path must be a str object."
    assert os.path.exists(path), path + " does not exist."
    assert path.endswith(".xlsx"), path + ' must be a ".xlsx" file.'

    # get the raw data
    raw = pd.read_excel(path)

    # extract the test and participant data
    return _get_data(raw), _get_participant(raw)
