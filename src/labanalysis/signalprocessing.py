"""
signalprocessing

a set of functions dedicated to the processing and analysis of 1D signals

Functions
---------
find_peaks
    find peaks in the signal

contiguous_batches
    get the indices defining contiguous samples in the signal.

nextpow
    the next power of the selected base.

winter_derivative1
    obtain the first derivative of a 1D signal according to Winter 2009 method.

winter_derivative2
    obtain the second derivative of a 1D signal according to Winter 2009 method.

feedman_diaconis_bins
    digitize a 1D signal in bins defined according to the freedman-diaconis rule

fir_filt
    apply a FIR (Finite Impulse Response) filter to a 1D signal

mean_filt
    apply a moving average filter to a 1D signal

median_filt
    apply a median filter to a 1D signal

rms_filt
    apply a rms filter to a 1D signal

butterworth_filt
    apply a butterworth filter to a 1D signal

cubicspline_interp
    apply cubic spline interpolation to a 1D signal

residual_analysis
    get the optimal cut-off frequency for a filter on 1D signals according
    to Winter 2009 'residual analysis' method

crossovers
    get the x-axis coordinates of the junction between the lines best fitting
    a 1D signal in a least-squares sense.

psd
    obtain the power spectral density estimate of a 1D signal using the
    periodogram method.

crossings
    obtain the location of the samples being across a target value.

xcorr
    get the cross/auto-correlation and lag of of multiple/one 1D signal.

outlyingness
    return the adsjusted outlyingness factor.

gram_schmidt
    return the orthogonal basis defined by a set of points using the
    Gram-Schmidt algorithm.

fillna
    fill missing data in numpy ndarray or pandas dataframe

tkeo
    obtain the discrete Teager-Keiser Energy of the input signal.

padwin
    pad the signal according to the given order and return the mask of
    indices defining each window on the signal.

to_reference_frame
    rotate a 3D array or dataframe to the provided reference frame
"""

#! IMPORTS

from types import FunctionType, MethodType
from typing import Any, Literal
from itertools import product
from pandas import DataFrame, Series
from scipy import signal  # type: ignore
from scipy.interpolate import CubicSpline  # type: ignore
from scipy.spatial.transform import Rotation
import numpy as np

from .regression import PolynomialRegression


__all__ = [
    "find_peaks",
    "continuous_batches",
    "nextpow",
    "winter_derivative1",
    "winter_derivative2",
    "freedman_diaconis_bins",
    "fir_filt",
    "padwin",
    "mean_filt",
    "median_filt",
    "rms_filt",
    "butterworth_filt",
    "cubicspline_interp",
    "residual_analysis",
    "crossovers",
    "psd",
    "crossings",
    "xcorr",
    "outlyingness",
    "gram_schmidt",
    "fillna",
    "tkeo",
    "to_reference_frame",
]


#! FUNCTIONS


def find_peaks(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    height: int | float | None = None,
    distance: int | None = None,
):
    """
    find peaks in the signal

    Parameters
    ----------
    arr : np.ndarray[Any, np.dtype[np.float64]]
        the input signal

    height : Union[int, float, None]
        the minimum height of the peaks

    distance : Union[int, None]
        the minimum distance between the peaks

    Returns
    -------
    p: np.ndarray[Any, np.dtype[np.int64]]
        the array containing the samples corresponding to the detected peaks
    """
    # get all peaks
    d1y = arr[1:] - arr[:-1]
    all_peaks = np.where((d1y[1:] < 0) & (d1y[:-1] >= 0))[0] + 1

    # select those peaks at minimum height
    if len(all_peaks) > 0 and height is not None:
        all_peaks = all_peaks[arr[all_peaks] >= height]

    # select those peaks separated at minimum by the given distance
    if len(all_peaks) > 1 and distance is not None:
        i = 1
        while i < len(all_peaks):
            if all_peaks[i] - all_peaks[i - 1] < distance:
                if arr[all_peaks[i]] > arr[all_peaks[i - 1]]:
                    all_peaks = np.append(all_peaks[: i - 1], all_peaks[i:])
                else:
                    all_peaks = np.append(all_peaks[:i], all_peaks[i + 1 :])
            else:
                i += 1

    return all_peaks.astype(int)


def continuous_batches(
    arr: np.ndarray[Any, np.dtype[np.bool_]],
):
    """
    return the list of indices defining batches where consecutive arr
    values are True.

    Parameters
    ----------
    arr : np.ndarray[Any, np.dtype[np.bool_]]
        a 1D boolean array

    Returns
    -------
    samples: list[list[bool]]
        a list of lists containing the samples defining a batch of consecutive
        True values.
    """
    locs = arr.astype(int)
    idxs = np.diff(locs)
    idxs = np.concatenate([[locs[0]], idxs])
    crs = locs + idxs
    if locs[-1] == 1:
        crs = np.concatenate([crs, [-1]])
    starts = np.where(crs == 2)[0]
    stops = np.where(crs == -1)[0]
    return [np.arange(i, v).tolist() for i, v in zip(starts, stops)]


def nextpow(
    val: int | float,
    base: int = 2,
):
    """
    get the next power of the provided value according to the given base.

    Parameters
    ----------
    val : Union[int, float]
        the target value

    base : int, optional
        the base to be elevated

    Returns
    -------
    out: int
        the next power of the provided value according to the given base.
    """
    return int(round(base ** np.ceil(np.log(val) / np.log(base))))


def winter_derivative1(
    y_signal: np.ndarray[Any, np.dtype[np.float64]],
    x_signal: None | np.ndarray[Any, np.dtype[np.float64]] = None,
    time_diff: float | int = 1,
):
    """
    return the first derivative of y.

    Parameters
    ----------

    y_signal: np.ndarray[Any, np.dtype[np.float64]]
        the signal to be derivated

    x_signal: None | np.ndarray[Any, np.dtype[np.float64]]
        the optional signal from which y has to  be derivated (default = None)

    time_diff: float | int
        the difference between samples in y.
        NOTE: if x is provided, this parameter is ignored

    Returns
    -------

    z: ndarray
        an array being the first derivative of y

    References
    ----------

    Winter DA. Biomechanics and Motor Control of Human Movement. Fourth Ed.
        Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
    """

    # get x
    if x_signal is None:
        x_sig = np.arange(len(y_signal)) * time_diff
    else:
        x_sig = x_signal

    # get the derivative
    return (y_signal[2:] - y_signal[:-2]) / (x_sig[2:] - x_sig[:-2])


def winter_derivative2(
    y_signal: np.ndarray[Any, np.dtype[np.float64]],
    x_signal: None | np.ndarray[Any, np.dtype[np.float64]] = None,
    time_diff: float | int = 1,
):
    """
    return the second derivative of y.

    Parameters
    ----------

    y_signal: np.ndarray[Any, np.dtype[np.float64]]
        the signal to be derivated

    x_signal: None | np.ndarray[Any, np.dtype[np.float64]]
        the optional signal from which y has to  be derivated (default = None)

    time_diff: float | int
        the difference between samples in y.
        NOTE: if x is provided, this parameter is ignored

    Returns
    -------

    z: np.ndarray[Any, np.dtype[np.float64]]
        an array being the second derivative of y

    References
    ----------

    Winter DA. Biomechanics and Motor Control of Human Movement. Fourth Ed.
        Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
    """

    # get x
    if x_signal is None:
        x_sig = np.arange(len(y_signal)) * time_diff
    else:
        x_sig = np.copy(x_signal)

    # get the derivative
    num = y_signal[2:] + y_signal[:-2] - 2 * y_signal[1:-1]
    den = np.mean(np.diff(x_sig)) ** 2
    return num / den


def freedman_diaconis_bins(
    y_signal: np.ndarray[Any, np.dtype[np.float64]],
):
    """
    return a digitized version of y where each value is linked to a
    bin (i.e an int value) according to the rule.

                             IQR(x)
            width = 2 * ---------------
                        len(x) ** (1/3)

    Parameters
    ----------

    y_signal: np.ndarray[Any, np.dtype[np.float64]]
        the signal to be digitized.

    Returns
    -------

    d: np.ndarray[Any, np.dtype[np.float64]]
        an array with the same shape of y containing the index
        of the bin of which the corresponding sample of y is part.

    References
    ----------
    Freedman D, Diaconis P.
        (1981) On the histogram as a density estimator:L 2 theory.
        Z. Wahrscheinlichkeitstheorie verw Gebiete 57: 453-476.
        doi: 10.1007/BF01025868
    """

    # y IQR
    qnt1 = np.quantile(y_signal, 0.25)
    qnt3 = np.quantile(y_signal, 0.75)
    iqr = qnt3 - qnt1

    # get the width
    wdt = 2 * iqr / (len(y_signal) ** (1 / 3))

    # get the number of intervals
    samp = int(np.floor(1 / wdt)) + 1

    # digitize z
    digitized = np.zeros(y_signal.shape)
    for i in np.arange(samp) + 1:
        loc = np.argwhere((y_signal >= (i - 1) * wdt) & (y_signal < i * wdt))
        digitized[loc] = i - 1
    return digitized


def padwin(
    arr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
):
    """
    pad the signal according to the given order and return the mask of
    indices defining each window on the signal.

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]],
        the signal to be filtered.

    order: int = 1,
        the number of samples to be considered as averaging window.

    pd: str = "edge"
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    offset: float
        a value within the [0, 1] range defining how the averaging window is
        obtained.
        Offset=0,
            indicate that for each sample, the filtered value will be the mean
            of the subsequent n-1 values plus the current sample.
        Offset=1,
            on the other hand, calculates the filtered value at each sample as
            the mean of the n-1 preceding values plus the current sample.
        Offset=0.5,
            centers the averaging window around the actual sample being
            evaluated.

    Returns
    -------
    pad: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
        The padded signal

    mask: np.ndarray[Literal[2], np.dtype[np.int64]],
        a 2D mask where each row denotes the indices of one window.
    """
    # get the window range
    stop = order - int(np.floor(order * offset)) - 1
    init = order - stop - 1

    # get the indices of the samples
    idx = np.arange(len(arr)) + init

    # padding
    pad = np.pad(arr, [init, stop], mode=pad_style)  # type: ignore

    # get the windows mask
    rng = np.arange(-init, stop + 1)
    mask = np.atleast_2d([rng + i for i in idx])

    return pad, mask


def mean_filt(
    arr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
):
    """
    apply a moving average filter to the signal.

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]],
        the signal to be filtered.

    order: int = 1,
        the number of samples to be considered as averaging window.

    pd: str = "edge"
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    offset: float
        a value within the [0, 1] range defining how the averaging window is
        obtained.
        Offset=0,
            indicate that for each sample, the filtered value will be the mean
            of the subsequent n-1 values plus the current sample.
        Offset=1,
            on the other hand, calculates the filtered value at each sample as
            the mean of the n-1 preceding values plus the current sample.
        Offset=0.5,
            centers the averaging window around the actual sample being
            evaluated.

    Returns
    -------
    z: 1D array
        The filtered signal.
    """

    # get the window range
    stop = order - int(np.floor(order * offset)) - 1
    init = order - stop - 1

    # get the indices of the samples
    idx = np.arange(len(arr)) + init

    # padding
    pad = np.pad(arr, [init, stop], mode=pad_style)  # type: ignore

    # get the cumulative sum of the signal
    csum = np.cumsum(pad).astype(float)

    # get the mean
    return (csum[idx + stop] - csum[idx - init]) / order


def median_filt(
    arr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
):
    """
    apply a median filter to the signal.

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]],
        the signal to be filtered.

    order: int = 1,
        the number of samples to be considered as averaging window.

    pd: str = "edge"
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    offset: float
        a value within the [0, 1] range defining how the averaging window is
        obtained.
        Offset=0,
            indicate that for each sample, the filtered value will be the mean
            of the subsequent n-1 values plus the current sample.
        Offset=1,
            on the other hand, calculates the filtered value at each sample as
            the mean of the n-1 preceding values plus the current sample.
        Offset=0.5,
            centers the averaging window around the actual sample being
            evaluated.

    Returns
    -------
    z: 1D array
        The filtered signal.
    """
    pad, mask = padwin(arr, order, pad_style, offset)
    return np.array([np.median(pad[i]) for i in mask])


def rms_filt(
    arr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
):
    """
    obtain the root-mean square of the signal with the given sampling window

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]],
        the signal to be filtered.

    order: int = 1,
        the number of samples to be considered as averaging window.

    pd: str = "edge"
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    offset: float
        a value within the [0, 1] range defining how the averaging window is
        obtained.
        Offset=0,
            indicate that for each sample, the filtered value will be the mean
            of the subsequent n-1 values plus the current sample.
        Offset=1,
            on the other hand, calculates the filtered value at each sample as
            the mean of the n-1 preceding values plus the current sample.
        Offset=0.5,
            centers the averaging window around the actual sample being
            evaluated.

    Returns
    -------
    z: 1D array
        The filtered signal.
    """

    # get the window range
    stop = order - int(np.floor(order * offset)) - 1
    init = order - stop - 1

    # get the indices of the samples
    idx = np.arange(len(arr)) + init

    # padding
    pad = np.pad(arr, [init, stop], mode=pad_style)  # type: ignore

    # get the squares of the signal
    sqe = pad**2

    # get the cumulative sum of the signal
    csum = np.cumsum(sqe).astype(float)

    # get the root mean of the squares
    return ((csum[idx + stop] - csum[idx - init]) / order) ** 0.5


def fir_filt(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    fcut: float | int | list[float | int] | tuple[float | int] = 1,
    fsamp: float | int = 2,
    order: int = 5,
    ftype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    wtype: Literal[
        "boxcar",
        "triang",
        "blackman",
        "hamming",
        "hann",
        "bartlett",
        "flattop",
        "parzen",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
        "cosine",
        "exponential",
        "tukey",
        "taylor",
    ] = "hamming",
    pstyle: Literal[
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
    ] = "edge",
):
    """
    apply a FIR filter with the specified specs to the signal.

    Parameters
    ----------

    arr: np.ndarray[Any, np.dtype[np.float64]],
        the signal to be filtered.

    fcut: float | int | list[float | int], tuple[float | int] = 1,
        the cutoff frequency of the filter.

    fsamp: float | int = 2,
        The sampling frequency of the signal.

    order: int = 5,
        the order of the filter

    ftype: str = "lowpass",
        the type of filter. Any of "bandpass", "lowpass", "highpass",
        "bandstop".

    wn: str
        the type of window to be applied. Any of:
            "boxcar",
            "triang",
            "blackman",
            "hamming",
            "hann",
            "bartlett",
            "flattop",
            "parzen",
            "bohman",
            "blackmanharris",
            "nuttall",
            "barthann",
            "cosine",
            "exponential",
            "tukey",
            "taylor"

    pd: str
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    Returns
    -------

    filtered: 1D array
        the filtered signal.
    """
    coefs = signal.firwin(
        order,
        fcut,
        window=wtype,
        pass_zero=ftype,  # type: ignore
        fs=fsamp,
    )
    val = arr[0] if pstyle == "constant" else 0
    padded = np.pad(
        arr,
        pad_width=(2 * order - 1, 0),
        mode=pstyle,
        constant_values=val,
    )
    avg = np.mean(padded)
    out = signal.lfilter(coefs, 1.0, padded - avg)[(2 * order - 1) :]
    return np.array(out).flatten().astype(float) + avg


def butterworth_filt(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    fcut: float | int | list[float | int] | tuple[float | int] = 1,
    fsamp: float | int = 2,
    order: int = 5,
    ftype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    phase_corrected: bool = True,
):
    """
    Provides a convenient function to call a Butterworth filter with the
    specified parameters.

    Parameters
    ----------

    arr: np.ndarray[Any, np.dtype[np.float64]],
        the signal to be filtered.

    fcut: float | int | list[float | int], tuple[float | int] = 1,
        the cutoff frequency of the filter.

    fsamp: float | int = 2,
        The sampling frequency of the signal.

    order: int = 5,
        the order of the filter

    ftype: str = "lowpass",
        the type of filter. Any of "bandpass", "lowpass", "highpass",
        "bandstop".

    phase_corrected: bool, optional
        should the filter be applied twice in opposite directions
        to correct for phase lag?

    Returns
    -------

    z: np.ndarray[Any, np.dtype[np.float64]],
        the resulting 1D filtered signal.
    """

    # get the filter coefficients
    sos = signal.butter(
        order,
        (np.array([fcut]).flatten() / (0.5 * fsamp)),
        ftype,
        analog=False,
        output="sos",
    )

    # get the filtered data
    if phase_corrected:
        arr = signal.sosfiltfilt(sos, arr)
    else:
        arr = signal.sosfilt(sos, arr)  # type: ignore
    return np.array([arr]).astype(float).flatten()


def cubicspline_interp(
    y_old: np.ndarray[Any, np.dtype[np.float64]],
    nsamp: int | None = None,
    x_old: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    x_new: np.ndarray[Any, np.dtype[np.float64]] | None = None,
):
    """
    Get the cubic spline interpolation of y.

    Parameters
    ----------

    y_old: np.ndarray[Any, np.dtype[np.float64]],
        the data to be interpolated.

    nsamp: int | None = None,
        the number of points for the interpolation.

    x_old: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        the x coordinates corresponding to y. It is ignored if n is provided.

    x_new: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        the newly (interpolated) x coordinates corresponding to y.
        It is ignored if n is provided.

    Returns
    -------
    z: np.ndarray[Any, np.dtype[np.float64]]
        the interpolated y axis
    """

    # control of the inputs
    if nsamp is None:
        if x_old is None or x_new is None:
            raise ValueError("the pair x_old / x_new or nsamp must be defined")
    else:
        x_old = np.arange(len(y_old))  # type: ignore
        x_new = np.linspace(np.min(x_old), np.max(x_old), nsamp)  # type: ignore

    # get the cubic-spline interpolated y
    cspline = CubicSpline(x_old, y_old)
    return cspline(x_new).flatten().astype(float)


def residual_analysis(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    ffun: FunctionType | MethodType,
    fnum: int = 1000,
    fmax: float | int | None = None,
    nseg: int = 2,
    minsamp: int = 2,
):
    """
    Perform Winter's residual analysis of the input signal.

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]],
        the signal to be investigated

    ffun: FunctionType | MethodType,
        the filter to be used for the analysis. The function must receive two
        inputs: the raw signal and the filter cut-off. The output must be the
        filtered signal.

    fnum: int = 1000,
        the number of frequencies to be tested within the (0, f_max) range to
        create the residuals curve of the Winter's residuals analysis approach.

    fmax: float | int | None = None,
        the maximum frequency to be tested in normalized units in the (0, 0.5)
        range. If None, it is defined as the frequency covering the 99% of
        the cumulative signal power.

    nseg: int = 2,
        the number of segments that can be used to fit the residuals curve in
        order to identify the best deflection point.
        NOTE: values above 3 will greatly increase the computation time.

    minsamp: int = 2,
        the minimum number of elements that have to be considered for each
        segment during the calculation of the best deflection point.

    Returns
    -------

    cutoff: float
        the suggested cutoff value

    frequencies: np.ndarray[Any, np.dtype[np.float64]],
        the tested frequencies

    residuals: np.ndarray[Any, np.dtype[np.float64]],
        the residuals corresponding to the given frequency

    Notes
    -----

    The signal is filtered over a range of frequencies and the sum of squared
    residuals (SSE) against the original signal is computer for each tested
    cut-off frequency. Next, a series of fitting lines are used to estimate the
    optimal disruption point defining the cut-off frequency optimally
    discriminating between noise and good quality signal.

    References
    ----------

    Winter DA 2009, Biomechanics and Motor Control of Human Movement.
        Fourth Ed. John Wiley & Sons Inc, Hoboken, New Jersey (US).

    Lerman PM 1980, Fitting Segmented Regression Models by Grid Search.
        Appl Stat. 29(1):77.
    """

    # data check
    if fmax is None:
        pwr, frq = psd(arr, 1)
        idx = int(np.where(np.cumsum(pwr) / np.sum(pwr) >= 0.99)[0][0])  # type: ignore
        fmax = max(float(frq[frq < 0.5][-1]), float(frq[idx]))
    assert 0 < fmax < 0.5, "fmax must lie in the (0, 0.5) range."
    assert minsamp >= 2, "'min_samples' must be >= 2."

    # get the optimal crossing over point
    frq = np.linspace(0, fmax, fnum + 1)[1:].astype(float)
    res = np.array([np.sum((arr - ffun(arr, i)) ** 2) for i in frq])
    iopt = crossovers(res, nseg, minsamp)[0][-1]
    fopt = float(frq[iopt])

    # return the parameters
    return fopt, frq, res.astype(float)


def _sse(
    xval: np.ndarray[Any, np.dtype[np.float64 | np.int64]],
    yval: np.ndarray[Any, np.dtype[np.float64 | np.int64]],
    segm: list[tuple[int]],
):
    """
    method used to calculate the residuals

    Parameters
    ----------

    xval: np.ndarray[Any, np.dtype[np.float64]],
        the x axis signal

    yval: np.ndarray[Any, np.dtype[np.float64]],
        the y axis signal

    segm: list[tuple[int]],
        the extremes among which the segments have to be fitted

    Returns
    -------

    sse: float
        the sum of squared errors corresponding to the error obtained
        fitting the y-x relationship according to the segments provided
        by s.
    """
    sse = 0.0
    for i in np.arange(len(segm) - 1):
        coords = np.arange(segm[i], segm[i + 1] + 1)
        coefs = np.polyfit(xval[coords], yval[coords], 1)
        vals = np.polyval(coefs, xval[coords])
        sse += np.sum((yval[coords] - vals) ** 2)
    return float(sse)


def crossovers(
    arr: np.ndarray[Any, np.dtype[np.float64 | np.int64]],
    segments: int = 2,
    min_samples: int = 5,
):
    """
    Detect the position of the crossing over points between K regression
    lines used to best fit the data.

    Parameters
    ----------
    arr:np.ndarray[Any, np.dtype[np.float64]],
        the signal to be fitted.

    segments:int=2,
        the number of segments that can be used to fit the residuals curve in
        order to identify the best deflection point.
        NOTE: values above 3 will greatly increase the computation time.

    min_samples:int=5,
        the minimum number of elements that have to be considered for each
        segment during the calculation of the best deflection point.

    Returns
    -------

    crossings: list[int]
        An ordered array of indices containing the samples corresponding to the
        detected crossing over points.

    coefs: list[tuple[float]]
        A list of tuples containing the slope and intercept of the line
        describing each fitting segment.

    Notes
    -----

    the steps involved in the calculations can be summarized as follows:

        1)  Get all the segments combinations made possible by the given
            number of crossover points.
        2)  For each combination, calculate the regression lines corresponding
            to each segment.
        3)  For each segment calculate the residuals between the calculated
            regression line and the effective data.
        5)  Once the sum of the residuals have been calculated for each
            combination, sort them by residuals amplitude.

    References
    ----------

    Lerman PM 1980, Fitting Segmented Regression Models by Grid Search.
    Appl Stat. 29(1):77.
    """

    # control the inputs
    assert min_samples >= 2, "'min_samples' must be >= 2."

    # get the X axis
    xaxis = np.arange(len(arr))

    # get all the possible combinations of segments
    combs = []
    for i in np.arange(1, segments):
        start = min_samples * i
        stop = len(arr) - min_samples * (segments - i)
        combs += [np.arange(start, stop)]
    combs = list(product(*combs))

    # remove those combinations having segments shorter than "samples"
    combs = [i for i in combs if np.all(np.diff(i) >= min_samples)]

    # generate the crossovers matrix
    combs = (
        np.zeros((len(combs), 1)),
        np.atleast_2d(combs),
        np.ones((len(combs), 1)) * len(arr) - 1,
    )
    combs = np.hstack(combs).astype(int)

    # calculate the residuals for each combination
    sse = np.array([_sse(xaxis, arr, i) for i in combs])

    # sort the residuals
    sortedsse = np.argsort(sse)

    # get the optimal crossovers order
    crs = xaxis[combs[sortedsse[0]]]

    # get the fitting slopes
    slopes = [np.arange(i0, i1) for i0, i1 in zip(crs[:-1], crs[1:])]
    slopes = [np.polyfit(i, arr[i], 1).astype(float) for i in slopes]

    # return the crossovers
    return crs[1:-1].astype(int).tolist(), slopes


def psd(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    fsamp: float | int = 1.0,
):
    """
    compute the power spectrum of signal using fft

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]],
        A 1D numpy array

    fssamp: float | int = 1.0,
        the sampling frequency (in Hz) of the signal. If not provided the
        power spectrum frequencies are provided as normalized values within the
        (0, 0.5) range.

    Returns
    -------
    frq: np.ndarray[Any, np.dtype[np.float64]],
        the frequency corresponding to each element of pwr.

    pwr: np.ndarray[Any, np.dtype[np.float64]],
        the power of each frequency
    """

    # get the psd
    fft = np.fft.rfft(arr - np.mean(arr)) / len(arr)
    amp = abs(fft)
    pwr = np.concatenate([[amp[0]], 2 * amp[1:-1], [amp[-1]]]).flatten() ** 2
    frq = np.linspace(0, fsamp / 2, len(pwr))

    # return the data
    return frq.astype(float), pwr.astype(float)


def crossings(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    value: int | float = 0.0,
):
    """
    Dectect the crossing points in x compared to value.

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]],
        the 1D signal from which the crossings have to be found.

    value: int | float = 0.0,
        the crossing value.

    Returns
    -------
    crs: 1D array
        the samples corresponding to the crossings.

    sgn: 1D array
        the sign of the crossings. Positive sign means crossings
        where the signal moves from values lower than "value" to
        values higher than "value". Negative sign indicate the
        opposite trend.
    """

    # get the sign of the signal without the offset
    sgn = np.sign(arr - value)

    # get the location of the crossings
    crs = np.where(abs(sgn[1:] - sgn[:-1]) == 2)[0].astype(int)

    # return the crossings
    return crs, -sgn[crs]


def xcorr(
    sig1: np.ndarray[Any, np.dtype[np.float64]],
    sig2: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    biased: bool = False,
    full: bool = False,
):
    """
    set the (multiple) auto/cross correlation of the data in y.

    Parameters
    ----------
    sig1: np.ndarray[Any, np.dtype[np.float64]],
        the signal from which the auto or cross-correlation is provided.

    sig2: np.ndarray[Any, np.dtype[np.float64]] | None = None,
        the signal from which the auto or cross-correlation is provided.
        if None. The autocorrelation of x is provided. Otherwise the x-y
        cross-correlation is returned.

    biased:bool=False,
        if True, the biased auto/cross-correlation is provided.
        Otherwise, the 'unbiased' estimator is returned.

    full:bool=False,
        Should the negative lags be reported?

    Returns
    -------
    xcr: np.ndarray[Any, np.dtype[np.float64]]
        the auto/cross-correlation value.

    lag: np.ndarray[Any, np.dtype[np.float64]]
        the lags in sample units.
    """

    # take the autocorrelation if only y is provided
    if sig2 is None:
        sigx = np.atleast_2d(sig1)
        sigz = np.vstack([sigx, sigx])

    # take the cross-correlation (ensure the shortest signal is zero-padded)
    else:
        sigx = np.zeros((1, max(len(sig1), len(sig2))))
        sigy = np.copy(sigx)
        sigx[:, : len(sig1)] = sig1
        sigy[:, : len(sig2)] = sig2
        sigz = np.vstack([sigx, sigy])

    # get the matrix shape
    rows, cols = sigz.shape

    # remove the mean from each dimension
    sigv = sigz - np.atleast_2d(np.mean(sigz, 1)).T

    # take the cross-correlation
    xcr = []
    for i in np.arange(rows - 1):
        for j in np.arange(i + 1, rows):
            res = signal.fftconvolve(sigv[i], sigv[j][::-1], "full")
            xcr += [np.atleast_2d(res)]

    # average over all the multiples
    xcr = np.mean(np.concatenate(xcr, axis=0), axis=0)  # type: ignore

    # adjust the output
    lags = np.arange(-(cols - 1), cols)
    if not full:
        xcr = xcr[(cols - 1) :]
        lags = lags[(cols - 1) :]

    # normalize
    xcr /= (cols + 1 - abs(lags)) if not biased else (cols + 1)

    # return the cross-correlation data
    return xcr.astype(float), lags.astype(int)


def outlyingness(
    arr: np.ndarray[Any, np.dtype[np.float64]],
):
    """
    return the adsjusted outlyingness factor.

    Parameters
    ----------
    arr: np.ndarray[Any, np.dtype[np.float64]]
        the input array

    Returns
    -------
    out: np.ndarray[Any, np.dtype[np.float64]]
        the outlyingness score of each element

    References
    ----------
    Hubert, M., & Van der Veeken, S. (2008).
        Outlier detection for skewed data.
        Journal of Chemometrics: A Journal of the Chemometrics Society,
        22(3‐4), 235-246.
    """
    qr1, med, qr3 = np.percentile(arr, [0.25, 0.50, 0.75])
    iqr = qr3 - qr1
    low = arr[arr < med]
    upp = arr[arr > med]
    mcs = [((j - med) - (med - i)) / (j - i) for i, j in product(low, upp)]
    mcs = np.median(mcs)
    if mcs > 0:
        wt1 = qr1 - 1.5 * np.e ** (-4 * mcs) * iqr
        wt2 = qr3 + 1.5 * np.e ** (3 * mcs) * iqr
    else:
        wt1 = qr1 - 1.5 * np.e ** (-3 * mcs) * iqr
        wt2 = qr3 + 1.5 * np.e ** (4 * mcs) * iqr
    out = []
    for i in arr:
        if i == med:
            out += [0]
        elif i > med:
            out += [(i - med) / (wt2 - med)]
        else:
            out += [(med - i) / (med - wt1)]
    return np.array(out)


def gram_schmidt(
    *points: np.ndarray[Any, np.dtype[np.number]],
):
    """
    Return the orthogonal basis defined by a set of points using the
    Gram-Schmidt algorithm.

    Parameters
    ----------
    points: np.ndarray[Any, np.dtype[np.float64]]
        a NxN numpy.ndarray to be orthogonalized (by row).

    Returns
    -------
    a tuple of orthonormal versors
    """

    # calculate the projection points
    w_mat = []
    for i, proj in enumerate(points):
        w_arr = proj.astype(float).flatten()
        for j in points[:i]:
            w_arr -= np.inner(proj, j) / np.inner(j, j) * j
        w_mat += [w_arr]

    # normalize
    w_mat = np.vstack(np.atleast_2d(w_mat))  # type: ignore
    return w_mat / (np.ones_like(w_mat) * (w_mat**2).sum(axis=1) ** 0.5).T


def fillna(
    arr: np.ndarray | DataFrame | Series,
    value: float | int | None = None,
    n_regressors: int | None = None,
):
    """
    fill missing values in the array or dataframe.

    Parameters
    ----------
    arr : np.ndarray | DataFrame | Series,
        array with nans to be filled

    value : float or None
        the value to be used for missing data replacement.
        if None, nearest neighbours imputation from the sklearn package is
        used.

    n_regressors : int | None, default=NOne
        Number of regressors to be used in a Multiple Linear Regression model.
        The model used the "n_regressors" most correlated columns of
        arr as indipendent variables to fit the missing values. The procedure
        is repeated for each dimension separately.
        If None, cubic spline interpolation is used on each column separately.

    Returns
    -------
    filled: np.ndarray
        the vector without missing data.
    """
    # check if missing values exist
    if not isinstance(arr, (DataFrame, np.ndarray, Series)):
        raise TypeError(
            "'arr' must be a numpy.ndarray a pandas.DataFrame or a pandas.Series."
        )
    if isinstance(arr, np.ndarray):
        obj = DataFrame(arr, copy=True)
    elif isinstance(arr, Series):
        obj = DataFrame(arr, copy=True).T
    else:
        obj = arr.copy().astype(float)
    miss = np.isnan(obj.values)

    # otherwise return a copy of the actual vector
    if not miss.any():
        return arr.copy()

    # fill with the given value
    if value is not None:
        obj.iloc[miss] = value
        if isinstance(arr, np.ndarray):
            return obj.values.astype(float)
        elif isinstance(arr, Series):
            return Series(obj[obj.columns[0]])
        else:
            return obj

    # check if linear regression models have to be used
    if n_regressors is not None:
        # get the correlation matrix
        cmat = obj.corr(numeric_only=True).values

        # predict the missing values via linear regression over each column
        cols = obj.columns.tolist()
        for i, ycol in enumerate(obj.columns):

            # get the best regressors
            corrs = abs(cmat[i])
            cor_idx = np.argsort(corrs)[-n_regressors - 1 : -1]
            xcols = [cols[i] for i in cor_idx]

            # get the indices of the samples that can be used for training
            # the regression model and those samples that can be predicted
            # with that model
            i_old = obj.loc[obj[[ycol] + xcols].notna().all(axis=1)].index
            i_new = obj.loc[obj[ycol].isna() & obj[xcols].notna().all(axis=1)].index

            # if there are enough valid samples get the predictions and replace
            # the missing data
            if len(i_old) > 2 and len(i_new) > 0:
                xmat = obj.loc[i_old, xcols]
                yarr = obj.loc[i_old, [ycol]]
                lrm = PolynomialRegression(degree=1).fit(xmat, yarr)
                preds = lrm.predict(obj.loc[i_new, xcols])
                obj.loc[i_new, ycol] = preds.values.astype(float).flatten()

    # fill the missing data of each set via cubic spline
    for i, col in enumerate(obj.columns):
        x_new = np.where(np.isnan(obj[col].values.astype(float)))[0]
        x_old = np.where(~np.isnan(obj[col].values.astype(float)))[0]
        if len(x_new) > 0 and len(x_old) > 0:
            y_old = obj[col].values[x_old].astype(float)
            obj.iloc[x_new, i] = CubicSpline(x_old, y_old)(x_new).astype(float)

    # return the filled array
    if isinstance(arr, np.ndarray):
        return obj.values.astype(float)
    elif isinstance(arr, Series):
        return Series(obj[obj.columns[0]])
    else:
        return obj


def tkeo(
    arr: np.ndarray[Literal[1], np.dtype[np.float64 | np.int64]],
):
    """
    obtain the discrete Teager-Keiser Energy of the input signal.

    Parameters
    ----------
    arr : np.ndarray[Literal[1], np.dtype[np.float64  |  np.int64]]
        a 1D input signal

    Returns
    -------
    tke: np.ndarray[Literal[1], np.dtype[np.float64]]
        the Teager-Keiser energy
    """
    out = arr[1:-1] ** 2 - arr[2:] * arr[:-2]
    return np.concatenate([[out[0]], out, [out[-1]]]).astype(float)


def to_reference_frame(
    obj: DataFrame | np.ndarray,
    origin: np.ndarray | list[float | int] = [0, 0, 0],
    axis1: np.ndarray | list[float | int] = [1, 0, 0],
    axis2: np.ndarray | list[float | int] = [0, 1, 0],
    axis3: np.ndarray | list[float | int] = [0, 0, 1],
):
    """
    rotate a 3D array or dataframe to the provided reference frame.

    Parameters
    ----------
    obj: DataFrame | np.ndarray
        the 3D array or dataframe to be rotated.

    origin: np.ndarray | list[float | int]
        an array of len = 3 with the coordinates of the target origin

    axis1: np.ndarray | list[float | int]
        an array of len = 3 with the coordinates representing the orientation
        of the first axis of the new reference frame

    axis2: np.ndarray | list[float | int]
        an array of len = 3 with the coordinates representing the orientation
        of the second axis of the new reference frame

    axis3: np.ndarray | list[float | int]
        an array of len = 3 with the coordinates representing the orientation
        of the third axis of the new reference frame

    Returns
    -------
    rotated: DataFrame | np.ndarray
        the rotated data.
    """

    def _validate_array(arr: object):
        msg = "origin, axis1, axis2 and axis3 have to be"
        msg += " castable to 1D arrays of len = 3."
        try:
            out = np.array([arr]).astype(float).flatten()
        except Exception:
            raise ValueError(msg)
        if len(out) != 3:
            raise ValueError(msg)
        return out

    # check inputs
    msg = "'obj' must be a numeric pandas DataFrame or a 2D numpy array"
    msg += " with 3 elements along the second dimension."
    try:
        dfr = DataFrame(obj)
        if dfr.shape[1] != 3:
            raise ValueError(msg)
    except Exception:
        raise ValueError(msg)
    ori = np.ones(dfr.shape) * _validate_array(origin)
    ax1 = _validate_array(axis1)
    ax2 = _validate_array(axis2)
    ax3 = _validate_array(axis3)

    # create the rotation matrix
    rmat = Rotation.from_matrix(gram_schmidt(ax1, ax2, ax3))

    # apply
    rotated = rmat.apply(dfr.values - ori).astype(float)
    if not isinstance(obj, DataFrame):
        return rotated
    return DataFrame(rotated, columns=obj.columns, index=obj.index)
