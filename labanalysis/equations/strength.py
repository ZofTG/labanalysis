"""strength module containing 1RM predicting equations"""


class Brzycki1RM:
    """
    class allowing the calculation of the 1RM and its derivatives according
    to the Brzycki equaiton.
    """

    def _validate_reps(self, reps: object):
        """private method used to validate the reps input"""
        if not isinstance(reps, int):
            raise ValueError("reps must be an int.")
        if reps < 0 or reps >= 37:
            raise ValueError("reps must be an int within the [1, 36] range.")

    def _validate_load(self, load: object):
        """private method used to validate the load input"""
        if not isinstance(load, (float, int)):
            raise ValueError("load must be a float or int.")
        if load < 0:
            raise ValueError("load must be >= 0")

    def _validate_1rm(self, rm1: object):
        """private method used to validate the 1rm input"""
        if not isinstance(rm1, (float, int)):
            raise ValueError("rm1 must be a float or int.")
        if rm1 < 0:
            raise ValueError("rm1 must be >= 0")

    def predict_1rm(self, reps: int, load: float | int):
        """
        return the 1RM in kg from the given reps and load in kg

        Parameters
        ----------
        reps: int
            the number of repetitions

        load: float | int
            the number of load

        Returns
        -------
        1rm: float
            return the predicted 1RM in kg.
        """
        self._validate_reps(reps)
        self._validate_load(load)
        return load * 36 / (37 - reps)

    def predict_reps(self, rm1: float | int, load: float | int):
        """
        return the 1RM in kg from the given reps and load in kg

        Parameters
        ----------
        rm1: float | int
            the 1rm in kg

        load: float | int
            the number of load

        Returns
        -------
        1rm: float
            return the predicted 1RM in kg.
        """
        self._validate_1rm(rm1)
        self._validate_load(load)
        if load > rm1:
            raise ValueError("load must be <= rm1.")
        return 37 - 36 * load / rm1

    def predict_load(self, rm1: float | int, reps: int):
        """
        return the 1RM in kg from the given reps and load in kg

        Parameters
        ----------
        rm1: float | int
            the 1rm in kg

        reps: int
            the number of reps

        Returns
        -------
        load: float
            return the load allowing the required number of reps given the 1RM.
        """
        self._validate_reps(reps)
        self._validate_1rm(rm1)
        return rm1 * (37 - reps) / 36
