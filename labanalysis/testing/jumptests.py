"""jumptests module"""

#! IMPORTS

from typing import Any, Iterable, Literal
from os.path import exists
import pandas as pd
import numpy as np

from labio import read_tdf
from .. import signalprocessing

__all__ = []

#! FUNCTIONS


#! CLASSES


class _Baseline:
    """
    baseline class used internally for analysing compute the mass of the subject and
    the emg baseline signal.

    Parameters
    ----------
    file_path: str
        the path to the .tdf file containing baseline data

    marker_fcut: int | float | np._NumberType = 6
        cut frequency of the lowpass filter, in Hz

    marker_order: int | float | np._NumberType = 4
        order of the lowpass filter

    marker_phase_corrected: bool | np.bool_ = True
        if True the phase correction is applyed by appliyng a filter second time with the
        signal provided in the reverse order

    force_fcut: int | float | np._NumberType = 50
        cut frequency of the lowpass filter, in Hz

    force_order: int | float | np._NumberType = 4
        order of the lowpass filter

    force_phase_corrected: bool | np.bool_ = True
        if True the phase correction is applyed by appliyng a filter second time with the
        signal provided in the reverse order

    emg_fcut: tuple[int | float, int | float] | list[int | float] | np.ndarray[Literal[1], np._NumberType] = (30, 450)
        cut frequency (the upper and lower limit) of the passband filter in Hz

    emg_order: int | float | np._NumberType = 4
        order of the passband filter

    emg_phase_corrected: bool | np.bool_ = True
        if True the phase correction is applyed by appliyng a filter second time with the signal
        provided in the reverse order

    emg_rms_window: int | float | np._NumberType = 0.2
        the Root Mean Square window provided in seconds

    Attributes
    ----------
    file_path : str
        the path of the file to analyze

    markers : pd.DataFrame
        the processed kinematic data

    forces : pd.DataFrame
        the processed force data

    emgs : pd.DataFrame
        the processed EMG data

    emg_processing_options : dict[str, Any]
        the parameters to set the filtering of the EMG signal

    force_processing_options: dict[str, Any]
        the parameters to set the filtering of the force signal

    marker_processing_options: dict[str, Any]
        the parameters to set the filtering of the kinematic signals

    Procedure
    ---------

    Markers
        1. only the 'S2' marker is used
        2. missing values at the beginning and end of the data are removed
        3. the data are low-pass filtered by means of a lowpass, Butterworth filter with the entered marker options

    Forces
        1. only the force amplitude of the 'fRes' force platform data are retained
        2. the data in between the start and end of the 'S2' marker are retained.
        3. missing values in the middle of the data are replaced by zeros
        4. the data are low-pass filtered by means of a lowpass, Butterworth filter with the entered force options

    EMGs (optional)
        1. the data in between the start and end of the 'S2' marker are retained.
        2. the signals are bandpass filtered with the provided emg options
        3. the root-mean square filter with the given time window is applied to get the envelope of the signals
    """

    # *class variables

    _file_path: str
    _tdf: dict[str, Any]
    _markers: pd.DataFrame
    _forces: pd.DataFrame
    _emgs: pd.DataFrame
    _emg_processing_options: dict[str, Any] = dict(
        fcut=[30, 400],
        order=5,
        phase_corrected=True,
        rms_window=0.2,
    )
    _force_processing_options: dict[str, Any] = dict(
        fcut=50,
        order=5,
        phase_corrected=True,
    )
    _marker_processing_options: dict[str, Any] = dict(
        fcut=6,
        order=5,
        phase_corrected=True,
    )

    # *setters

    def _set_file_path(self, file_path: str):
        """
        set the file path for the baseline acquisition

        Parameters
        ----------
        file_path: str
            the path to the .tdf file containing baseline data
        """
        if (
            not isinstance(file_path, str)
            or not file_path.endswith(".tdf")
            or not exists(file_path)
        ):
            raise ValueError("file_path must be a valid .tdf file")
        self._file_path = file_path

    def _set_emg_processing_options(
        self,
        fcut: (
            tuple[int | float, int | float]
            | list[int | float]
            | np.ndarray[Literal[1], np._NumberType]
            | None
        ) = None,
        order: int | float | np._NumberType | None = None,
        phase_corrected: bool | np.bool_ | None = None,
        rms_window: int | float | np._NumberType | None = None,
    ):
        """
        set the parameters of the filtering (bandpass filter and Root Mean Square) of the EMG signal

        Parameters
        ----------
        fcut: tuple[int | float, int | float] | list[int | float] | np.ndarray[Literal[1], np._NumberType] | None = None
            cut frequency (the upper and lower limit) of the passband filter in Hz

        order: int | float | np._NumberType | None = None
            order of the passband filter

        phase_corrected: bool | np.bool_ | None = None
            if True the phase correction is applyed by appliyng a filter second time with the signal provided in the reverse order

        rms_window: int | float | np._NumberType | None = None
            the Root Mean Square window provided in seconds
        """
        if fcut is not None:
            cut = np.array([fcut]).flatten()  # lo trasformo in un array unidimensionale
            if not self._is_1darray(fcut) or len(fcut) != 2:
                raise ValueError("fcut must be an iterable with len = 2")
            if not all([self._is_numeric(i) for i in fcut]):
                raise ValueError("All elements of fcut must be numeric")
            self._emg_processing_options["fcut"] = np.array(cut).astype(float)

        if order is not None:
            if not self._is_numeric(order):
                raise ValueError("order must be a number")
            if order <= 0:
                raise ValueError("order must be a number > 0")
            self._emg_processing_options["order"] = order

        if phase_corrected is not None:
            msg = "phase_corrected must be a boolean"
            ph_cor = np.array([phase_corrected]).flatten()
            if len(ph_cor) != 1:
                raise ValueError(msg)
            try:
                ph_cor = bool(ph_cor[0])
            except Exception as exc:
                raise ValueError(msg) from exc
            self._emg_processing_options["phase_corrected"] = ph_cor

        if rms_window is not None:
            if not self._is_numeric(rms_window):
                raise ValueError("order must be a number")
            if rms_window <= 0:
                raise ValueError("order must be a number > 0")
            self._emg_processing_options["rms_window"] = rms_window

    def _set_force_processing_options(
        self,
        fcut: int | float | np._NumberType | None = None,
        order: int | float | np._NumberType | None = None,
        phase_corrected: bool | np.bool_ | None = None,
    ):
        """
        set the parameters of the lowband filtering of the force data

        Attributes
        ----------
        fcut: int | float | np._NumberType | None = None
            cut frequency of the lowband filter, in Hz

        order: int | float | np._NumberType | None = None
            order of the lowband filter

        phase_corrected: bool | np.bool_ | None = None
            if True the phase correction is applyed by appliyng a filter second time with the signal provided in the reverse order

        """

        if fcut is not None:
            if not self._is_numeric(fcut):
                raise ValueError("order must be a number")
            if fcut <= 0:
                raise ValueError("order must be a number > 0")
            self._force_processing_options["fcut"] = fcut

        if order is not None:
            if not self._is_numeric(order):
                raise ValueError("order must be a number")
            if order <= 0:
                raise ValueError("order must be a number > 0")
            self._force_processing_options["order"] = order

        if phase_corrected is not None:
            msg = "phase_corrected must be a boolean"
            ph_cor = np.array([phase_corrected]).flatten()
            if len(ph_cor) != 1:
                raise ValueError(msg)
            try:
                ph_cor = bool(ph_cor[0])
            except Exception as exc:
                raise ValueError(msg) from exc
            self._force_processing_options["phase_corrected"] = ph_cor

    def _set_marker_processing_options(
        self,
        fcut: int | float | np._NumberType | None = None,
        order: int | float | np._NumberType | None = None,
        phase_corrected: bool | np.bool_ | None = None,
    ):
        """
        set the parameters of the lowband filtering of the kinematic data

        Attributes
        ----------
        fcut: int | float | np._NumberType | None = None
            cut frequency of the lowband filter, in Hz

        order: int | float | np._NumberType | None = None
            order of the lowband filter

        phase_corrected: bool | np.bool_ | None = None
            if True the phase correction is applyed by appliyng a filter second time with the signal provided in the reverse order

        """

        if fcut is not None:
            if not self._is_numeric(fcut):
                raise ValueError("order must be a number")
            if fcut <= 0:
                raise ValueError("order must be a number > 0")
            self._marker_processing_options["fcut"] = fcut

        if order is not None:
            if not self._is_numeric(order):
                raise ValueError("order must be a number")
            if order <= 0:
                raise ValueError("order must be a number > 0")
            self._marker_processing_options["order"] = order

        if phase_corrected is not None:
            msg = "phase_corrected must be a boolean"
            ph_cor = np.array([phase_corrected]).flatten()
            if len(ph_cor) != 1:
                raise ValueError(msg)
            try:
                ph_cor = bool(ph_cor[0])
            except Exception as exc:
                raise ValueError(msg) from exc
            self._marker_processing_options["phase_corrected"] = ph_cor

    # *getters

    @property
    def file_path(self):
        """
        the path to the .tdf file
        """
        return self._file_path

    @property
    def emg_processing_options(self):
        """
        the EMG processing options
        """
        return self._emg_processing_options

    @property
    def force_processing_options(self):
        """
        the force processing options
        """
        return self._force_processing_options

    @property
    def marker_processing_options(self):
        """
        the marker processing options
        """
        return self._marker_processing_options

    @property
    def markers(self):
        """
        the markers coordinates
        """
        return self._markers

    @property
    def forces(self):
        """
        the forces amplitudes
        """
        return self._forces

    @property
    def emgs(self):
        """
        the EMGs signals
        """
        return self._emgs

    # *methods

    def _is_numeric(self, obj: object):
        """
        check if obj is a number

        Parameter
        ---------
        obj : object
            the object to be checked

        Return
        ------
        True if obj is a number, False otherwise
        """
        wind = np.array([obj]).flatten()  # lo trasformo in un array unidimensionale
        if len(wind) != 1:
            return False
        try:
            wind = float(wind[0])
        except Exception as exc:
            return False
        return True

    def _is_1darray(self, obj: object, size: int | None = None):
        """
        check if 'obj' is a 1d iterable with (optional) len 'size'

        Parameter
        ---------
        obj : object
            the object to be checked

        size: int | None = None
            the required size of the array (if provided)

        Return
        ------
        True if obj is an iterable with the required length
        """
        if not isinstance(obj, Iterable):
            return False
        wind = np.array(obj)
        if wind.ndim != 1:
            return False
        if size is not None:
            if not isinstance(size, int):
                raise ValueError("size must be None or an int object")
            if len(wind) != size:
                return False
        return True

    def _process_data(self):
        """
        process of all tdf data

        Markers
        -------
        1. only the 'S2' marker is used
        2. missing values at the beginning and end of the data are removed
        3. the data are low-pass filtered by means of a lowpass, Butterworth filter with the entered marker options

        Forces
        ------
        1. only the force amplitude of the 'fRes' force platform data are retained
        2. the data in between the start and end of the 'S2' marker are retained.
        3. missing values in the middle of the data are replaced by zeros
        4. the data are low-pass filtered by means of a lowpass, Butterworth filter with the entered force options

        EMGs (optional)
        ----
        1. the data in between the start and end of the 'S2' marker are retained.
        2. the signals are bandpass filtered with the provided emg options
        3. the root-mean square filter with the given time window is applied to get the envelope of the signals
        """

        # controllo che sia disponibile il marker S2
        try:
            markers_raw: pd.DataFrame = self._tdf["CAMERA"]["TRACKED"]["TRACKS"]
        except Exception as exc:
            raise ValueError(
                "the provided .tdf file does not contain marker data."
            ) from exc
        marker_lbls = np.unique(markers_raw.columns.get_level_values(level=0).to_list())
        if not any([i == "S2" for i in marker_lbls]):
            raise ValueError("marker label S2 not found in the .tdf file")
        marker_raw: pd.DataFrame = markers_raw["S2"]

        # tolgo i valori nulli all'inizio ed alla fine del file
        marker_valid = marker_raw.notna().all(axis=1)
        index_start, index_stop = np.where(marker_valid.values)[0][0, -1]
        marker_pro: pd.DataFrame = marker_raw.iloc[
            np.arange(index_start, index_stop + 1)
        ]
        marker_pro = signalprocessing.fillna(marker_pro)

        # applico il filtro previsto
        marker_fsamp = float(1 / np.mean(np.diff(marker_pro.index.to_numpy())))
        self._markers = pd.DataFrame(
            marker_pro.apply(
                signalprocessing.butterworth_filt,
                raw=True,
                fsamp=marker_fsamp,
                ftype="lowpass",
                **self.marker_processing_options,
            )
        )

        # processo il segnale di forza
        try:
            forces_raw: pd.DataFrame = self._tdf["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]
        except Exception as exc:
            raise ValueError(
                "the provided .tdf file does not contain force data."
            ) from exc

        # mantengo solo i dati di forza di fRes
        try:
            forces_raw = forces_raw["fRes"]["FORCE"]
        except Exception as exc:
            raise ValueError(
                "the provided .tdf file doe not contain any force platform data named 'fRes'."
            ) from exc

        # rimuovo i dati mancanti all'inizio ed alla fine della prova
        index_start = np.where(forces_raw.index == self.markers.index[0])[0]
        index_stop = np.where(forces_raw.index == self.markers.index[-1])[0]
        forces_pro = forces_raw.iloc[np.arange(index_start, index_stop)]

        # sostituisco con zeri i nan presenti all'interno della prova
        forces_pro = signalprocessing.fillna(forces_pro, value=0)

        # filtro i dati di forza
        force_fsamp = float(1 / np.mean(np.diff(forces_pro.index.to_numpy())))
        self._forces = forces_pro.apply(
            signalprocessing.butterworth_filt,
            raw=True,
            fsamp=force_fsamp,
            ftype="lowpass",
            **self.force_processing_options,
        )

        # controllo sia disponibile il segnale emg
        try:
            emgs_raw: pd.DataFrame = self._tdf["EMG"]["TRACKS"]
        except Exception as exc:
            emgs_raw = pd.DataFrame()
            raise UserWarning(
                "the provided .tdf file does not contain EMG data."
            ) from exc

        # processo il segnale emg
        if emgs_raw.shape[0] > 0:
            index_start = np.where(emgs_raw.index == self.markers.index[0])[0]
            index_stop = np.where(emgs_raw.index == self.markers.index[-1])[0]
            emgs_pro = emgs_raw.iloc[np.arange(index_start, index_stop)]
            bp_options = self.emg_processing_options.copy()
            bp_options.pop("rms_window")
            bp_options.update(
                fsamp=float(1 / np.mean(np.diff(emgs_pro.index.to_numpy())))
            )
            emgs_pro = pd.DataFrame(
                emgs_pro.apply(
                    signalprocessing.butterworth_filt,
                    raw=True,
                    fsamp=marker_fsamp,
                    ftype="bandpass",
                    **bp_options,
                )
            )
            self._emgs = pd.DataFrame(
                emgs_pro.apply(
                    signalprocessing.rms_filt,
                    raw=True,
                    order=int(
                        bp_options["fsamp"] * self.emg_processing_options["rms_window"]
                    ),
                )
            )
        else:
            self._emgs = pd.DataFrame()

    # *costruttore
    def __init__(
        self,
        file_path: str,
        marker_fcut: int | float | np._NumberType = 6,
        marker_order: int | float | np._NumberType = 4,
        marker_phase_corrected: bool | np.bool_ = True,
        force_fcut: int | float | np._NumberType = 50,
        force_order: int | float | np._NumberType = 4,
        force_phase_corrected: bool | np.bool_ = True,
        emg_fcut: (
            tuple[int | float, int | float]
            | list[int | float]
            | np.ndarray[Literal[1], np._NumberType]
        ) = (30, 400),
        emg_order: int | float | np._NumberType = 4,
        emg_phase_corrected: bool | np.bool_ = True,
        emg_rms_window: int | float | np._NumberType = 0.2,
    ):
        # inizializzo le variabili
        self._set_file_path(file_path)
        self._set_marker_processing_options(
            fcut=marker_fcut,
            order=marker_order,
            phase_corrected=marker_phase_corrected,
        )
        self._set_force_processing_options(
            fcut=force_fcut,
            order=force_order,
            phase_corrected=force_phase_corrected,
        )
        self._set_emg_processing_options(
            fcut=emg_fcut,
            order=emg_order,
            phase_corrected=emg_phase_corrected,
            rms_window=emg_rms_window,
        )

        # leggo il file tdf
        self._tdf = read_tdf(self.file_path)

        # effettuo il data processing
        self._process_data()


class _SJ:
    """
    Squat jump class is used internally for processing the single squat jump

    Parameters
    ----------


    """


class _CMJ:
    """
    Counter movement jump class is used internally for processing the single counter movement jump

    Parameters
    ----------

    """


class Squat_Jump_test:
    """
    Class Squat_Jump_test is used to analyze the processed file of the single squat jump

    Parameters
    ----------

    """


class Counter_Movement_Jump_test:
    """
    Class Counter_Movement_Jump_test is used to analyze the processed file of the single counter movement jump

    Parameters
    ----------

    """
