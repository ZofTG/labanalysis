"""processing pipeline module"""

# -*- coding: utf-8 -*-


#! IMPORTS


from copy import deepcopy
from typing import Any, Callable, Dict, List
import pint

from .timeseries import *
from .timeseriesrecords import *

ureg = pint.UnitRegistry()


__all__ = ["ProcessingPipeline"]


class ProcessingPipeline:
    """
    A pipeline for processing various types of TimeseriesRecord-compatible
    objects.
    This class allows the user to define a sequence of processing functions
    for each supported object type and apply them to a collection of objects.
    """

    def __init__(
        self,
        signal1d_funcs: List[Callable[[Signal1D], Signal1D]] = [],
        signal3d_funcs: List[Callable[[Signal3D], Signal3D]] = [],
        point3d_funcs: List[Callable[[Point3D], Point3D]] = [],
        emgsignal_funcs: List[Callable[[EMGSignal], EMGSignal]] = [],
        forceplatform_funcs: List[Callable[[ForcePlatform], ForcePlatform]] = [],
    ):
        """
        Initialize a ProcessingPipeline.

        Parameters
        ----------
        signal1d_funcs : list of callables, optional
            Processing functions for Signal1D.
        signal3d_funcs : list of callables, optional
            Processing functions for Signal3D.
        point3d_funcs : list of callables, optional
            Processing functions for Point3D.
        emgsignal_funcs : list of callables, optional
            Processing functions for EMGSignal.
        forceplatform_funcs : list of callables, optional
            Processing functions for ForcePlatform.
        """
        self.pipeline: Dict[type, List[Callable[[Any], Any]]] = {
            Signal1D: signal1d_funcs,
            Signal3D: signal3d_funcs,
            Point3D: point3d_funcs,
            EMGSignal: emgsignal_funcs,
            ForcePlatform: forceplatform_funcs,
        }

    def apply(
        self,
        *objects: Signal1D | Signal3D | Point3D | EMGSignal | ForcePlatform,
        inplace: bool = False,
    ):
        """
        Apply the processing pipeline to the given objects.

        Parameters
        ----------
        *objects : variable length argument list
            Objects to process. Can be individual Signal1D, Signal3D, Point3D,
            EMGSignal, ForcePlatform, or TimeseriesRecord instances.
        inplace : bool, optional
            If True, modifies the objects in place. If False, returns the
            processed copies.

        Returns
        -------
        list or None
            If inplace is False, returns a list of processed objects. Otherwise, returns None.
        """
        processed_objects = []

        for obj in objects:
            obj_type = type(obj)
            funcs = self.pipeline.get(obj_type, [])
            if not inplace:
                obj = deepcopy(obj)
            for func in funcs:
                obj = func(obj)
            if not inplace:
                processed_objects.append(obj)

        if not inplace:
            return processed_objects
        # if inplace is True, nothing is returned
