"""
signalprocessing

A set of functions dedicated to the processing and analysis of 1D signals.

Functions
---------
find_peaks
    Find peaks in the signal.
continuous_batches
    Get the indices defining contiguous samples in the signal.
nextpow
    The next power of the selected base.
winter_derivative1
    Obtain the first derivative of a 1D signal according to Winter 2009 method.
winter_derivative2
    Obtain the second derivative of a 1D signal according to Winter 2009 method.
freedman_diaconis_bins
    Digitize a 1D signal in bins defined according to the Freedman-Diaconis rule.
fir_filt
    Apply a FIR (Finite Impulse Response) filter to a 1D signal.
mean_filt
    Apply a moving average filter to a 1D signal.
median_filt
    Apply a median filter to a 1D signal.
rms_filt
    Apply a RMS filter to a 1D signal.
butterworth_filt
    Apply a Butterworth filter to a 1D signal.
cubicspline_interp
    Apply cubic spline interpolation to a 1D signal.
residual_analysis
    Get the optimal cut-off frequency for a filter on 1D signals according to Winter 2009 'residual analysis' method.
crossovers
    Get the x-axis coordinates of the junction between the lines best fitting a 1D signal in a least-squares sense.
psd
    Obtain the power spectral density estimate of a 1D signal using the periodogram method.
crossings
    Obtain the location of the samples being across a target value.
xcorr
    Get the cross/auto-correlation and lag of multiple/one 1D signal.
outlyingness
    Return the adjusted outlyingness factor.
gram_schmidt
    Return the orthogonal basis defined by a set of points using the Gram-Schmidt algorithm.
fillna
    Fill missing data in numpy ndarray or pandas dataframe.
tkeo
    Obtain the discrete Teager-Kaiser Energy of the input signal.
padwin
    Pad the signal according to the given order and return the mask of indices defining each window on the signal.
to_reference_frame
    Rotate a 3D array or dataframe to the provided reference frame.
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
    "thresholding_filt",
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
    arr: np.ndarray,
    height: int | float | None = None,
    distance: int | None = None,
) -> np.ndarray:
    """
    Find peaks in the signal.

    Parameters
    ----------
    arr : np.ndarray
        The input signal.
    height : int or float or None, optional
        The minimum height of the peaks.
    distance : int or None, optional
        The minimum distance between the peaks.

    Returns
    -------
    np.ndarray
        The array containing the indices of the detected peaks.
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
    arr: np.ndarray,
) -> list[list[int]]:
    """
    Return the list of indices defining batches where consecutive arr values are True.

    Parameters
    ----------
    arr : np.ndarray
        A 1D boolean array.

    Returns
    -------
    list of list of int
        A list of lists containing the indices defining each batch of consecutive True values.
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
) -> int:
    """
    Get the next power of the provided value according to the given base.

    Parameters
    ----------
    val : int or float
        The target value.
    base : int, optional
        The base to be elevated.

    Returns
    -------
    int
        The next power of the provided value according to the given base.
    """
    return int(round(base ** np.ceil(np.log(val) / np.log(base))))


def winter_derivative1(
    y_signal: np.ndarray,
    x_signal: np.ndarray | None = None,
    time_diff: float | int = 1,
) -> np.ndarray:
    """
    Return the first derivative of y.

    Parameters
    ----------
    y_signal : np.ndarray
        The signal to be differentiated.
    x_signal : np.ndarray or None, optional
        The optional signal from which y has to be differentiated (default: None).
    time_diff : float or int, optional
        The difference between samples in y. Ignored if x_signal is provided.

    Returns
    -------
    np.ndarray
        The first derivative of y.

    References
    ----------
    Winter DA. Biomechanics and Motor Control of Human Movement. Fourth Ed. Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
    """

    # get x
    if x_signal is None:
        x_sig = np.arange(len(y_signal)) * time_diff
    else:
        x_sig = x_signal

    # get the derivative
    return (y_signal[2:] - y_signal[:-2]) / (x_sig[2:] - x_sig[:-2])


def winter_derivative2(
    y_signal: np.ndarray,
    x_signal: np.ndarray | None = None,
    time_diff: float | int = 1,
) -> np.ndarray:
    """
    Return the second derivative of y.

    Parameters
    ----------
    y_signal : np.ndarray
        The signal to be differentiated.
    x_signal : np.ndarray or None, optional
        The optional signal from which y has to be differentiated (default: None).
    time_diff : float or int, optional
        The difference between samples in y. Ignored if x_signal is provided.

    Returns
    -------
    np.ndarray
        The second derivative of y.

    References
    ----------
    Winter DA. Biomechanics and Motor Control of Human Movement. Fourth Ed. Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
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
    y_signal: np.ndarray,
) -> np.ndarray:
    """
    Digitize a 1D signal in bins defined according to the Freedman-Diaconis rule.

    Parameters
    ----------
    y_signal : np.ndarray
        The signal to be digitized.

    Returns
    -------
    np.ndarray
        An array with the same shape as y containing the index of the bin for each sample.

    References
    ----------
    Freedman D, Diaconis P. (1981) On the histogram as a density estimator: L2 theory. Z. Wahrscheinlichkeitstheorie verw Gebiete 57: 453-476. doi: 10.1007/BF01025868
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
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the signal according to the given order and return the mask of indices defining each window on the signal.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    order : int, optional
        The number of samples to be considered as averaging window.
    pad_style : str, optional
        The type of padding style adopted before filtering.
    offset : float, optional
        Value in [0, 1] defining how the averaging window is obtained.

    Returns
    -------
    pad : np.ndarray
        The padded signal.
    mask : np.ndarray
        A 2D mask where each row denotes the indices of one window on the signal.
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


def thresholding_filt(
    arr: np.ndarray,
    factor: float | int = 3,
    robust: bool = False,
    order: int = 3,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Apply a thresholding filter where only those values being moving average filter to the signal.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    factor : float or int, optional
        The factor multiplied by the standard deviation of the window to detect extremes.
    robust : bool, optional
        If True, use median and MAD; otherwise, use mean and std.
    order : int, optional
        The number of samples for the averaging window.
    pad_style : str, optional
        The type of padding style.
    offset : float, optional
        Value in [0, 1] defining how the averaging window is obtained.

    Returns
    -------
    np.ndarray
        The filtered signal.
    """

    # pad the array
    pads, mask = padwin(arr, order, pad_style, offset)

    # get the required values
    if robust:
        vals = np.array([np.median(pads[i]) for i in mask])
        thresh = [np.median(abs(pads[v] - vals[i])) for i, v in enumerate(mask)]
        thresh = np.array(thresh)
    else:
        vals = np.array([np.mean(pads[i]) for i in mask])
        thresh = np.array([np.std(pads[i]) for i in mask])

    # replace the extreme values
    out = np.copy(arr)
    extremes = np.abs(arr - vals) > factor * thresh
    out[extremes] = vals[extremes]

    return out


def mean_filt(
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Apply a moving average filter to the signal.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    order : int, optional
        The number of samples for the averaging window.
    pad_style : str, optional
        The type of padding style.
    offset : float, optional
        Value in [0, 1] defining how the averaging window is obtained.

    Returns
    -------
    np.ndarray
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
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Apply a median filter to the signal.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    order : int, optional
        The number of samples for the averaging window.
    pad_style : str, optional
        The type of padding style.
    offset : float, optional
        Value in [0, 1] defining how the averaging window is obtained.

    Returns
    -------
    np.ndarray
        The filtered signal.
    """
    pad, mask = padwin(arr, order, pad_style, offset)
    return np.array([np.median(pad[i]) for i in mask])


def rms_filt(
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Obtain the root-mean-square of the signal with the given sampling window.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    order : int, optional
        The number of samples for the averaging window.
    pad_style : str, optional
        The type of padding style.
    offset : float, optional
        Value in [0, 1] defining how the averaging window is obtained.

    Returns
    -------
    np.ndarray
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
    arr: np.ndarray,
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
) -> np.ndarray:
    """
    Apply a FIR filter with the specified specs to the signal.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    fcut : float, int, list, or tuple, optional
        The cutoff frequency of the filter.
    fsamp : float or int, optional
        The sampling frequency of the signal.
    order : int, optional
        The order of the filter.
    ftype : str, optional
        The type of filter: "bandpass", "lowpass", "highpass", "bandstop".
    wtype : str, optional
        The type of window to be applied.
    pstyle : str, optional
        The type of padding style.

    Returns
    -------
    np.ndarray
        The filtered signal.
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
    arr: np.ndarray,
    fcut: float | int | list[float | int] | tuple[float | int] = 1,
    fsamp: float | int = 2,
    order: int = 5,
    ftype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    phase_corrected: bool = True,
) -> np.ndarray:
    """
    Apply a Butterworth filter with the specified parameters.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    fcut : float, int, list, or tuple, optional
        The cutoff frequency of the filter.
    fsamp : float or int, optional
        The sampling frequency of the signal.
    order : int, optional
        The order of the filter.
    ftype : str, optional
        The type of filter: "bandpass", "lowpass", "highpass", "bandstop".
    phase_corrected : bool, optional
        If True, apply the filter twice in opposite directions to correct for phase lag.

    Returns
    -------
    np.ndarray
        The filtered signal.
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
    y_old: np.ndarray,
    nsamp: int | None = None,
    x_old: np.ndarray | None = None,
    x_new: np.ndarray | None = None,
) -> np.ndarray:
    """
    Get the cubic spline interpolation of y.

    Parameters
    ----------
    y_old : np.ndarray
        The data to be interpolated.
    nsamp : int or None, optional
        The number of points for the interpolation.
    x_old : np.ndarray or None, optional
        The x coordinates corresponding to y. Ignored if nsamp is provided.
    x_new : np.ndarray or None, optional
        The new x coordinates for interpolation. Ignored if nsamp is provided.

    Returns
    -------
    np.ndarray
        The interpolated y axis.
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
    arr: np.ndarray,
    ffun: FunctionType | MethodType,
    fnum: int = 1000,
    fmax: float | int | None = None,
    nseg: int = 2,
    minsamp: int = 2,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Perform Winter's residual analysis of the input signal.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be investigated.
    ffun : FunctionType or MethodType
        The filter to be used for the analysis.
    fnum : int, optional
        The number of frequencies to be tested.
    fmax : float or int or None, optional
        The maximum frequency to be tested.
    nseg : int, optional
        The number of segments for fitting.
    minsamp : int, optional
        The minimum number of elements per segment.

    Returns
    -------
    float
        The suggested cutoff value.
    np.ndarray
        The tested frequencies.
    np.ndarray
        The residuals corresponding to the given frequency.

    Notes
    -----
    The signal is filtered over a range of frequencies and the sum of squared residuals (SSE) against the original signal is computed for each tested cut-off frequency. Next, a series of fitting lines are used to estimate the optimal disruption point defining the cut-off frequency optimally discriminating between noise and good quality signal.

    References
    ----------
    Winter DA 2009, Biomechanics and Motor Control of Human Movement. Fourth Ed. John Wiley & Sons Inc, Hoboken, New Jersey (US).

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
    res = res.astype(float)
    iopt = crossovers(res, nseg, minsamp)[0][-1]
    fopt = float(frq[iopt])

    # return the parameters
    return fopt, frq, res.astype(float)


def _sse(
    xval: np.ndarray,
    yval: np.ndarray,
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
        coords = np.arange(segm[i], segm[i + 1] + 1)  # type: ignore
        coefs = np.polyfit(xval[coords], yval[coords], 1)
        vals = np.polyval(coefs, xval[coords])
        sse += np.sum((yval[coords] - vals) ** 2)
    return float(sse)


def crossovers(
    arr: np.ndarray,
    segments: int = 2,
    min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the position of the crossing over points between K regression lines used to best fit the data.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be fitted.
    segments : int, optional
        The number of segments for fitting.
    min_samples : int, optional
        The minimum number of elements per segment.

    Returns
    -------
    np.ndarray
        Indices of the detected crossing over points.
    np.ndarray
        Slopes and intercepts of the fitting segments.

    Notes
    -----
    Steps:
        1) Get all segment combinations.
        2) For each, calculate regression lines.
        3) For each, calculate residuals.
        4) Sort by residuals.
        5) Return best combination.

    References
    ----------
    Lerman PM 1980, Fitting Segmented Regression Models by Grid Search. Appl Stat. 29(1):77.
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
    slopes = [
        np.polyfit(i, arr[i].astype(float), 1).astype(float).tolist() for i in slopes
    ]
    slopes = np.array(slopes).astype(float)

    # return the crossovers
    return crs[1:-1].astype(int), slopes


def psd(
    arr: np.ndarray,
    fsamp: float | int = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of signal using FFT.

    Parameters
    ----------
    arr : np.ndarray
        A 1D numpy array.
    fsamp : float or int, optional
        The sampling frequency (Hz) of the signal.

    Returns
    -------
    np.ndarray
        The frequency corresponding to each element of power.
    np.ndarray
        The power of each frequency.
    """

    # get the psd
    fft = np.fft.rfft(arr - np.mean(arr)) / len(arr)
    amp = abs(fft)
    pwr = np.concatenate([[amp[0]], 2 * amp[1:-1], [amp[-1]]]).flatten() ** 2
    frq = np.linspace(0, fsamp / 2, len(pwr))

    # return the data
    return frq.astype(float), pwr.astype(float)


def crossings(
    arr: np.ndarray,
    value: int | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the crossing points in arr compared to value.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal from which the crossings have to be found.
    value : int or float, optional
        The crossing value.

    Returns
    -------
    np.ndarray
        The samples corresponding to the crossings.
    np.ndarray
        The sign of the crossings.
    """

    # get the sign of the signal without the offset
    sgn = np.sign(arr - value)

    # get the location of the crossings
    crs = np.where(abs(sgn[1:] - sgn[:-1]) == 2)[0].astype(int)

    # return the crossings
    return crs, -sgn[crs]


def xcorr(
    sig1: np.ndarray,
    sig2: np.ndarray | None = None,
    biased: bool = False,
    full: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Set the (multiple) auto/cross correlation of the data in y.

    Parameters
    ----------
    sig1 : np.ndarray
        The signal for auto/cross-correlation.
    sig2 : np.ndarray or None, optional
        The second signal for cross-correlation. If None, autocorrelation is computed.
    biased : bool, optional
        If True, use the biased estimator.
    full : bool, optional
        If True, report negative lags.

    Returns
    -------
    np.ndarray
        The auto/cross-correlation value.
    np.ndarray
        The lags in sample units.
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
    arr: np.ndarray,
) -> np.ndarray:
    """
    Return the adjusted outlyingness factor.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The outlyingness score of each element.

    References
    ----------
    Hubert, M., & Van der Veeken, S. (2008). Outlier detection for skewed data. Journal of Chemometrics: A Journal of the Chemometrics Society, 22(3‐4), 235-246.
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
    *points: np.ndarray,
) -> np.ndarray:
    """
    Return the orthogonal basis defined by a set of points using the Gram-Schmidt algorithm.

    Parameters
    ----------
    points : np.ndarray
        A NxN numpy.ndarray to be orthogonalized (by row).

    Returns
    -------
    np.ndarray
        A tuple of orthonormal versors.
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
    inplace: bool = False,
):
    """
    Fill missing values in the array or dataframe.

    Parameters
    ----------
    arr : np.ndarray | DataFrame | Series,
        array with nans to be filled

    value : float or None
        the value to be used for missing data replacement.
        if None, nearest neighbours imputation from the sklearn package is
        used.

    n_regressors : int | None, default=None
        Number of regressors to be used in a Multiple Linear Regression model.
        The model used the "n_regressors" most correlated columns of
        arr as independent variables to fit the missing values. The procedure
        is repeated for each dimension separately.
        If None, cubic spline interpolation is used on each column separately.

    inplace : bool, optional
        If True, fill in place (for DataFrame/Series). If False, return a new object.

    Returns
    -------
    filled: np.ndarray, DataFrame, or Series
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
        obj = arr if inplace else arr.copy().astype(float)
    miss = np.isnan(obj.values)

    # otherwise return a copy of the actual vector
    if not miss.any():
        if inplace:
            return arr
        else:
            return arr.copy()

    # fill with the given value
    if value is not None:
        obj.iloc[miss] = value
        if isinstance(arr, np.ndarray):
            if inplace:
                arr[:] = obj.values
                return arr
            return obj.values.astype(float)
        elif isinstance(arr, Series):
            if inplace:
                arr[:] = obj[obj.columns[0]].values
                return arr
            return Series(obj[obj.columns[0]])
        else:
            if inplace:
                arr.loc[:, :] = obj.values
                return arr
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
        if inplace:
            arr[:] = obj.values
            return arr
        return obj.values.astype(float)
    elif isinstance(arr, Series):
        if inplace:
            arr[:] = obj[obj.columns[0]].values
            return arr
        return Series(obj[obj.columns[0]])
    else:
        if inplace:
            arr.loc[:, :] = obj.values
            return arr
        else:
            return obj


def tkeo(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Obtain the discrete Teager-Keiser Energy of the input signal.

    Parameters
    ----------
    arr : np.ndarray
        A 1D input signal.

    Returns
    -------
    np.ndarray
        The Teager-Keiser energy.
    """
    out = arr[1:-1] ** 2 - arr[2:] * arr[:-2]
    return np.concatenate([[out[0]], out, [out[-1]]]).astype(float)


def to_reference_frame(
    obj: DataFrame | np.ndarray,
    origin: np.ndarray | list[float | int] = [0, 0, 0],
    axis1: np.ndarray | list[float | int] = [1, 0, 0],
    axis2: np.ndarray | list[float | int] = [0, 1, 0],
    axis3: np.ndarray | list[float | int] = [0, 0, 1],
) -> DataFrame | np.ndarray:
    """
    Rotate a 3D array or dataframe to the provided reference frame.

    Parameters
    ----------
    obj : DataFrame or np.ndarray
        The 3D array or dataframe to be rotated.
    origin : np.ndarray or list, optional
        Coordinates of the target origin.
    axis1 : np.ndarray or list, optional
        Orientation of the first axis of the new reference frame.
    axis2 : np.ndarray or list, optional
        Orientation of the second axis of the new reference frame.
    axis3 : np.ndarray or list, optional
        Orientation of the third axis of the new reference frame.

    Returns
    -------
    DataFrame or np.ndarray
        The rotated data.
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
