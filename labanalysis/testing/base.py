"""base test module"""

#! IMPORTS


from abc import abstractmethod


__all__ = ["G", "LabTest"]

#! CONSTANTS


G = 9.80665  # acceleration of gravity in m/s^2


#! CLASSES


class LabTest:
    """
    abstract class defining the general methods expected from a test

    Attributes
    -------
    summary_plot
        return a matplotlib figure highlighting the test' results.

    summary_table
        return a pandas dataframe containing the summary data.
    """

    @property
    @abstractmethod
    def summary_plot(self):
        return NotImplementedError

    @property
    @abstractmethod
    def summary_table(self):
        return NotImplementedError