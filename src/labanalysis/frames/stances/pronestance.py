"""static tests module containing Static Tests"""

#! IMPORTS

import numpy as np
import pandas as pd

from ...regression.ols.geometry import Circle

from ...constants import G
from ..timeseries.emgsignal import EMGSignal
from ..timeseries.point3d import Point3D
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseriesrecords.forceplatform import ForcePlatform
from ..timeseriesrecords.timeseriesrecord import TimeseriesRecord

__all__ = ["ProneStance"]


#! CLASSES


class ProneStance(TimeseriesRecord):

    def _get_coordination_and_balance(
        self, left: np.ndarray, right: np.ndarray, unit: str
    ):
        line = {
            f"left_{unit}": np.mean(left),
            f"right_{unit}": np.mean(right),
        }
        line["coordination_%"] = np.corrcoef(left, right)[0][1] * 100
        den = line["right_avg"] + line["left_avg"]
        balance = line["right_avg"] / den * 100 - 50
        line["balance_%"] = balance
        return line

    @property
    def side(self):
        return "bilateral"

    @property
    def bodymass_kg(self):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        return float(self.vertical_force.mean() / G)

    @property
    def muscle_coordination_and_balance(self):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = self.emgsignals
        if len(emgs) == 0:
            return pd.DataFrame()

        # check the presence of left and right muscles
        muscles = {}
        for emg in emgs.values():
            name = emg.muscle_name
            side = emg.side
            if side not in ["left", "right"]:
                continue
            if name not in list(muscles.keys()):
                muscles[name] = {}

            # get the area under the curve of the muscle activation
            muscles[name][side] = np.asarray(emg.data, float).flatten()

        # remove those muscles not having both sides
        muscles = {i: v for i, v in muscles.items() if len(v) == 2}

        # calculate coordination and imbalance between left and right side
        out = {}
        for muscle, sides in muscles.items():
            params = self._get_coordination_and_balance(
                **sides,
                unit=emgs[muscle].unit,
            )
            out.update(**{f"{muscle}_{i}": v for i, v in params.items()})

        return pd.DataFrame(pd.Series(out)).T

    @property
    def force_coordination_and_balance(self):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics, or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        left_hand = self.get("left_hand_ground_reaction_force")
        right_hand = self.get("right_hand_ground_reaction_force")
        if (
            left_foot is None
            or right_foot is None
            or left_hand is None
            or right_hand is None
        ):
            return pd.DataFrame()
        left_foot = np.asarray(left_foot["force"][self.vertical_axis], float).flatten()
        right_foot = np.asarray(
            right_foot["force"][self.vertical_axis], float
        ).flatten()
        left_hand = np.asarray(left_hand["force"][self.vertical_axis], float).flatten()
        right_hand = np.asarray(
            right_hand["force"][self.vertical_axis], float
        ).flatten()

        # get the pairs to be tested
        pairs = {
            "lower_limbs": {"left_foot": left_foot, "right_foot": right_foot},
            "upper_limbs": {"left_hand": left_hand, "right_hand": right_hand},
            "whole_body": {
                "left_body": left_foot + left_hand,
                "right_body": right_foot + right_hand,
            },
        }

        # calculate balance and coordination
        out = []
        unit = self.vertical_force.unit
        for region, pair in pairs.items():
            left, right = list(pair.values())
            fit = self._get_coordination_and_balance(
                left / G / self.bodymass_kg,
                right / G / self.bodymass_kg,
                unit,
            )
            line = {f"force_{i}": v for i, v in fit.items()}
            line = pd.DataFrame(pd.Series(line)).T
            line.insert(0, "region", region)
            out += [line]

        return pd.concat(out, ignore_index=True)

    @property
    def area_of_stability_sqm(self):
        cop = self.centre_of_pressure
        ap, ml = cop[[self.anteroposterior_axis, self.lateral_axis]].T
        circle = Circle().fit(ap, ml)
        return circle.area

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = {
            "type": self.__class__.__name__,
            "side": self.side,
            "bodymass_kg": self.bodymass_kg,
            "area_of_stability_sqm": self.area_of_stability_sqm,
        }
        new = pd.DataFrame(pd.Series(new)).T
        return pd.concat(
            [
                new,
                self.force_coordination_and_balance,
                self.muscle_coordination_and_balance,
            ],
            axis=1,
        )

    def __init__(
        self,
        left_foot_ground_reaction_force: ForcePlatform,
        right_foot_ground_reaction_force: ForcePlatform,
        left_hand_ground_reaction_force: ForcePlatform,
        right_hand_ground_reaction_force: ForcePlatform,
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
        strip: bool = True,
        reset_time: bool = True,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a Jump object.

        Parameters
        ----------
        left_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the left foot.
        right_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the right foot.
        vertical_axis : str, optional
            Name of the vertical axis in the force data (default "Y").
        anteroposterior_axis : str, optional
            Name of the anteroposterior axis in the force data (default "X").
        **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
            Additional signals to include in the record.

        Raises
        ------
        TypeError
            If left_foot or right_foot is not a ForcePlatform.
        ValueError
            If axes are not valid or bodymass_kg is not a float or int.
        """

        # check the inputs
        forces = {}
        if not isinstance(left_foot_ground_reaction_force, ForcePlatform):
            raise ValueError(
                "left_foot_ground_reaction_force must be a ForcePlatform"
                + " instance or None."
            )
            forces["left_foot_ground_reaction_force"] = left_foot_ground_reaction_force
        if not isinstance(right_foot_ground_reaction_force, ForcePlatform):
            raise ValueError(
                "right_foot_ground_reaction_force must be a ForcePlatform"
                + " instance or None."
            )
            forces["right_foot_ground_reaction_force"] = (
                right_foot_ground_reaction_force
            )
        if not isinstance(left_hand_ground_reaction_force, ForcePlatform):
            raise ValueError(
                "left_hand_ground_reaction_force must be a ForcePlatform"
                + " instance or None."
            )
            forces["left_hand_ground_reaction_force"] = left_hand_ground_reaction_force
        if not isinstance(right_hand_ground_reaction_force, ForcePlatform):
            raise ValueError(
                "right_hand_ground_reaction_force must be a ForcePlatform"
                + " instance or None."
            )
            forces["right_hand_ground_reaction_force"] = (
                right_hand_ground_reaction_force
            )
        if len(forces) == 0:
            raise ValueError(
                "at least one of 'left_foot_ground_reaction_force' or"
                + "'right_foot_ground_reaction_force' must be ForcePlatform"
                + " instances."
            )

        # build the object
        super().__init__(
            vertical_axis=vertical_axis,
            anteroposterior_axis=anteroposterior_axis,
            strip=strip,
            reset_time=reset_time,
            **signals,
            **forces,
        )
