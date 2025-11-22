#!/usr/bin/env python3
"""Combine CH1 traces from two scope captures, fit, and compare."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import traceback


def _ensure_strict_bounds(lower, upper, rel_eps=1e-8, abs_eps=1e-12):
    """Return (lower, upper) arrays where each lower[i] < upper[i].

    If any lower[i] >= upper[i], expand the interval slightly using a
    relative and absolute epsilon so curve_fit's strict bound requirement
    is satisfied.
    """
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    if l.shape != u.shape:
        raise ValueError("lower/upper must have same shape")

    # compute adaptive eps per-parameter
    for i in range(len(l)):
        if not np.isfinite(l[i]):
            # if lower is -inf and upper maybe finite, leave as-is
            continue
        if not np.isfinite(u[i]):
            # upper is +inf
            continue
        if l[i] >= u[i]:
            scale = max(abs(u[i]), abs(l[i]), 1.0)
            eps = max(abs_eps, rel_eps * scale)
            # set lower slightly below and upper slightly above the midpoint
            mid = 0.5 * (l[i] + u[i])
            l[i] = mid - eps
            u[i] = mid + eps

    return l.tolist(), u.tolist()




DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHANNEL = "CH1"
MIN_TIME = -0.012
MAX_TIME = 0.0381
BASE_DOWNSAMPLE = 25  # Align high-resolution captures with CSVFINAL.
SAMPLE_STRIDE = 20  # Set >1 to average every N points after alignment/windowing.

DOPPLER_DOWNSAMPLE = 25  # Align high-resolution captures with CSVFINAL.
DOPPLER_STRIDE = 5  # Set >1 to average every N points after alignment/windowing.

# Filtering defaults for extrema detection
RELATIVE_AMPLITUDE_THRESHOLD = 0.2  # fraction of full y-range: keep extrema close to global extrema
MAX_RELATIVE_DISTANCE = None  # fraction of x-range; None disables x-distance filtering
# Defaults for peak-finding sensitivity (for CH2 Gaussian fitting)
FIND_PEAKS_PROMINENCE_FACTOR = 0.15  # fraction of y-range to use as minimum prominence
FIND_PEAKS_DISTANCE_FRACTION = 0.005  # fraction of total samples as minimum separation

# Simple debug options (toggle by editing these values)
# Set plotting options: enable/disable individual plots
PLOT_OPTIONS = {
    "show_sum_and_fits": False,
    "show_difference": True,
    "show_ch2": True,
    "show_spacings": True,
    "show_rescaled_diffs": True,
}
# Set printing options: enable/disable diagnostic prints
PRINT_OPTIONS = {
    "print_detected_extrema": False,
    "print_ch2_fits": True,
    "print_spacing_fits": False,
}


def load_scope_csv(csv_path: Path) -> Tuple[str, List[float], Dict[str, List[float]]]:
    """Return axis label, x values, and channel -> samples."""
    with csv_path.open() as handle:
        reader = csv.reader(handle)
        header = None
        units = None

        for row in reader:
            if row and row[0].strip() == "Source":
                header = [col.strip() for col in row]
                units = [col.strip() for col in next(reader, [])]
                break

        if header is None:
            raise ValueError(f"'Source' header not found in {csv_path}")

        data_rows = [row for row in reader if any(field.strip() for field in row)]

    source_label = header[0]
    source_values = [float(row[0]) for row in data_rows]

    channels: Dict[str, List[float]] = {}
    for idx, name in enumerate(header[1:], start=1):
        if not name.strip():
            continue
        channels[name] = [float(row[idx]) for row in data_rows]

    return source_label, source_values, channels


def load_channel(csv_path: Path, channel: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the requested channel as numpy arrays."""
    _, source_values, channels = load_scope_csv(csv_path)
    if channel not in channels:
        raise KeyError(f"{channel} not present in {csv_path.name}")
    return np.array(source_values), np.array(channels[channel])


def sum_ch1_signals(path_a: Path, path_b: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned source axis and the elementwise sum of CH1 traces."""
    src_a, ch1_a = load_channel(path_a, CHANNEL)
    src_b, ch1_b = load_channel(path_b, CHANNEL)

    if len(src_a) != len(src_b):
        raise ValueError("CSV files have different sample counts.")
    if not np.allclose(src_a, src_b):
        raise ValueError("Source axes differ; cannot sum traces safely.")

    return src_a[::2 * BASE_DOWNSAMPLE], (ch1_a + ch1_b)[::2 * BASE_DOWNSAMPLE]
    src_a[::2 * BASE_DOWNSAMPLE], (ch1_a + ch1_b)[::2 * BASE_DOWNSAMPLE]


def restrict_time_window(x: np.ndarray, *ys: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Restrict signals to the configured time window."""
    mask = (x >= MIN_TIME) & (x <= MAX_TIME)
    if not np.any(mask):
        raise ValueError(
            f"No samples fall inside the requested window [{MIN_TIME}, {MAX_TIME}]."
        )
    sliced = (x[mask],) + tuple(arr[mask] for arr in ys)
    return sliced


def apply_sampling_stride(x: np.ndarray, *ys: np.ndarray, stride: int = SAMPLE_STRIDE) -> Tuple[np.ndarray, ...]:
    """Average contiguous `stride` samples (no-op when `stride == 1`).

    The default value for `stride` is the module-level `SAMPLE_STRIDE`, so
    existing calls remain compatible. Pass an explicit `stride` to override.
    """
    if stride <= 1:
        return (x,) + tuple(ys)

    def block_mean(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        usable = len(array) - len(array) % stride
        if usable == 0:
            raise ValueError(f"Not enough samples ({len(array)}) for stride {stride}.")
        trimmed = array[:usable]
        return trimmed.reshape(-1, stride).mean(axis=1)

    avg_x = block_mean(x)
    avg_ys = tuple(block_mean(arr) for arr in ys)
    return (avg_x,) + avg_ys

def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_exponential(x: np.ndarray, y: np.ndarray):
    # --- Good initial guesses ---
    # c ≈ last value (baseline)
    c0 = y[-1]
    # a ≈ y0 - c0 (initial amplitude)
    a0 = y[0] - c0
    # b small positive or negative slope
    b0 = (np.log(max(y[1], 1e-9)) - np.log(max(y[0], 1e-9))) / (x[1] - x[0])

    p0 = [a0, b0, c0]

    try:
        (a, b, c), cov = curve_fit(
            exp_model,
            x,
            y,
            p0=p0,
            maxfev=20000
        )
    except RuntimeError:
        # fallback: return a simple constant model
        a, b, c = 0.0, 0.0, float(np.mean(y))

    return a, b, c


def compute_u(xx: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Compute the rescaled coordinate u(x) for an exponential spacing model s(x)=a*exp(b*x)+c.

    Returns u(x) such that u'(x)=1/s(x). The function handles limiting cases for
    small b or small c to avoid numerical issues.
    """
    xx = np.asarray(xx, dtype=float)
    eps = 1e-12
    # Case: b approximately zero -> s(x) ≈ a + c (constant)
    if abs(b) < eps:
        denom = a + c
        if denom == 0:
            return xx.copy()
        return xx / denom
    # Case: c approximately zero -> s(x)=a*exp(b*x)
    if abs(c) < eps:
        return -(1.0 / (a * b)) * np.exp(-b * xx)
    # General case: u(x) = (1/c)*x - (1/(b*c))*ln(a*exp(b*x)+c)
    return (1.0 / c) * xx - (1.0 / (b * c)) * np.log(a * np.exp(b * xx) + c)


def compute_ch2_diffs(final_path: Path, doppler_path: Path, x_ref: np.ndarray, summed_ch1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and return (src_final, diff_ch1, diff_ch2) aligned to CSVFINAL axis.

    This mirrors the data preparation used in plotting/fitting but does not
    create any figures. `x_ref` should be the axis used for summed CH1.
    """
    # Load CH2 from both files
    src_final, final_ch2 = load_channel(final_path, "CH2")
    src_dop, doppler_ch2 = load_channel(doppler_path, "CH2")

    # Restrict to same time window and downsample / stride exactly as CH1 handling
    src_final, final_ch2 = restrict_time_window(src_final, final_ch2)
    src_final, final_ch2 = src_final[::DOPPLER_DOWNSAMPLE], final_ch2[::DOPPLER_DOWNSAMPLE]
    src_final, final_ch2 = apply_sampling_stride(src_final, final_ch2)

    src_dop, doppler_ch2 = restrict_time_window(src_dop, doppler_ch2)
    src_dop, doppler_ch2 = src_dop[::DOPPLER_DOWNSAMPLE], doppler_ch2[::DOPPLER_DOWNSAMPLE]
    src_dop, doppler_ch2 = apply_sampling_stride(src_dop, doppler_ch2)

    # Interpolate doppler onto final axis
    try:
        doppler_on_final = np.interp(src_final, src_dop, doppler_ch2)
    except Exception:
        if len(src_dop) == len(src_final) and np.allclose(src_dop, src_final):
            doppler_on_final = doppler_ch2
        else:
            raise

    diff_ch2 = doppler_on_final - final_ch2

    # CH1 diff aligned to CSVFINAL axis
    src_f1, final_ch1 = load_channel(final_path, CHANNEL)
    src_f1, final_ch1 = restrict_time_window(src_f1, final_ch1)
    src_f1, final_ch1 = src_f1[::BASE_DOWNSAMPLE], final_ch1[::BASE_DOWNSAMPLE]
    src_f1, final_ch1 = apply_sampling_stride(src_f1, final_ch1)
    diff_ch1 = final_ch1 - summed_ch1

    return src_final, diff_ch1, diff_ch2

def plot_sum_and_fits(x: np.ndarray, summed: np.ndarray) -> None:
    """Plot the summed signal along with linear/exponential fits."""
    slope, intercept = np.polyfit(x, summed, 1)

    def exp_model(xvals: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(b * xvals) + c

    try:
        (a, b, c) = fit_exponential(x, summed)
    except RuntimeError:
        a = b = 0.0
        c = float(np.mean(summed))

    # Polynomial fits (degree 2 and 3)
    coeffs2 = np.polyfit(x, summed, 2)
    p2 = np.poly1d(coeffs2)

    coeffs3 = np.polyfit(x, summed, 3)
    p3 = np.poly1d(coeffs3)

    coeffs2_str = ", ".join(f"{c:.3e}" for c in coeffs2)
    coeffs3_str = ", ".join(f"{c:.3e}" for c in coeffs3)

    plt.figure()
    plt.plot(x, summed, label="CH1 sum")
    plt.plot(x, slope * x + intercept, "--", label=f"Linear fit (y={slope:.3e}x+{intercept:.3e})")
    plt.plot(x, p2(x), "-.", label=f"Poly2 fit (coeffs={coeffs2_str})")
    plt.plot(x, p3(x), "-.", alpha=0.7, label=f"Poly3 fit (coeffs={coeffs3_str})")
    plt.plot(x, exp_model(x, a, b, c), ":", label=f"Exp fit (a={a:.3e}, b={b:.3e}, c={c:.3e})")
    plt.xlabel("Source")
    plt.ylabel("Voltage (scaled)")
    plt.title("Summed CH1 traces with fits")
    plt.legend()
    plt.grid(True)
    


def plot_difference(
    x: np.ndarray,
    summed: np.ndarray,
    final_path: Path,
    peak_x: Optional[np.ndarray] = None,
    min_x: Optional[np.ndarray] = None,
) -> None:
    """Plot CSVFINAL CH1 minus the provided sum.

    If `peak_x` and/or `min_x` are provided, mark those x positions on the plotted difference.
    """
    src_final, final_ch1 = load_channel(final_path, CHANNEL)
    src_final, final_ch1 = restrict_time_window(src_final, final_ch1)
    src_final, final_ch1 = src_final[::BASE_DOWNSAMPLE], final_ch1[:: BASE_DOWNSAMPLE]
    src_final, final_ch1 = apply_sampling_stride(src_final, final_ch1)
    if len(src_final) != len(x):
        raise ValueError(
            f"Sample counts differ (sum={len(x)}, final={len(src_final)}). "
            "Ensure CSVFINAL was captured at matching resolution."
        )
    if not np.allclose(src_final, x):
        raise ValueError("CSVFINAL axis does not match summed traces:"+str(len(src_final))+" "+str(len(x)))
    diff = final_ch1 - summed

    plt.figure()
    plt.plot(x, diff, label="CSVFINAL CH1 - sum")
    # If peak positions were provided, overlay markers at the detected peaks
    if peak_x is not None and len(peak_x) > 0:
        # Interpolate diff values at the requested x positions to place markers
        y_peaks = np.interp(peak_x, x, diff)
        plt.plot(peak_x, y_peaks, "o", color="red", markersize=6, label="Detected peaks")
    # If minima positions were provided, overlay markers for minima
    if min_x is not None and len(min_x) > 0:
        y_mins = np.interp(min_x, x, diff)
        plt.plot(min_x, y_mins, "^", color="blue", markersize=6, label="Detected minima")
    plt.xlabel("Source")
    plt.ylabel("Voltage difference")
    plt.title("Difference between CSVFINAL CH1 and sum")
    plt.grid(True)
    plt.legend()


def detect_peaks_and_spacings(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    distance: Optional[float] = None,
    relative_amplitude_threshold: Optional[float] = None,
    max_relative_distance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect local maxima and minima in `y` and return their x positions and spacings.

        Returns `(peak_x, min_x, spacings, midpoints)` where:
            - `spacings` contains the x-axis differences between consecutive extrema
                (maxima and minima) sorted by x.
            - `midpoints` contains the mean x position between each consecutive
                pair of extrema and has the same length as `spacings`.
    The function prefers `scipy.signal.find_peaks` for robustness and falls
    back to a simple local-extrema detector when unavailable.
    """
    # Use module defaults when caller does not provide explicit filters
    if relative_amplitude_threshold is None:
        relative_amplitude_threshold = RELATIVE_AMPLITUDE_THRESHOLD
    if max_relative_distance is None:
        max_relative_distance = MAX_RELATIVE_DISTANCE

    try:
        from scipy.signal import find_peaks  # type: ignore

        find_kwargs = {}
        if prominence is not None:
            find_kwargs["prominence"] = prominence
        if height is not None:
            find_kwargs["height"] = height
        if distance is not None:
            if not np.isscalar(distance):
                find_kwargs["distance"] = distance
            else:
                dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
                find_kwargs["distance"] = max(1, int(round(distance / dx)))

        peaks_idx, _ = find_peaks(y, **find_kwargs)
        mins_idx, _ = find_peaks(-y, **find_kwargs)
    except Exception:
        # Fallback: simple local maxima/minima detector
        if len(y) < 3:
            peaks_idx = np.array([], dtype=int)
            mins_idx = np.array([], dtype=int)
        else:
            inner_max = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
            inner_min = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
            peaks_idx = np.nonzero(inner_max)[0] + 1
            mins_idx = np.nonzero(inner_min)[0] + 1

    peak_x = np.asarray(x)[peaks_idx]
    min_x = np.asarray(x)[mins_idx]

    # Optionally filter extrema by amplitude relative to global extrema
    if relative_amplitude_threshold is not None and (len(y) > 0):
        y_max = float(np.max(y))
        y_min = float(np.min(y))
        y_range = y_max - y_min if y_max != y_min else 0.0

        if y_range > 0.0:
            if peak_x.size > 0:
                y_peaks = np.interp(peak_x, x, y)
                keep_peak = y_peaks >= (y_max - relative_amplitude_threshold * y_range)
                peak_x = peak_x[keep_peak]
            if min_x.size > 0:
                y_mins = np.interp(min_x, x, y)
                keep_min = y_mins <= (y_min + relative_amplitude_threshold * y_range)
                min_x = min_x[keep_min]

    # Optionally filter extrema by distance (fraction of x-range) from global extrema
    if max_relative_distance is not None and (len(x) > 1):
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_range = x_max - x_min if x_max != x_min else 0.0
        if x_range > 0.0:
            # location of global extrema
            x_of_global_max = float(x[np.argmax(y)])
            x_of_global_min = float(x[np.argmin(y)])
            allowed = float(max_relative_distance) * x_range
            if peak_x.size > 0:
                keep_peak = np.abs(peak_x - x_of_global_max) <= allowed
                peak_x = peak_x[keep_peak]
            if min_x.size > 0:
                keep_min = np.abs(min_x - x_of_global_min) <= allowed
                min_x = min_x[keep_min]

    # Combine extrema, sort by x, and compute spacings and midpoints between consecutive extrema
    if peak_x.size == 0 and min_x.size == 0:
        spacings = np.array([], dtype=float)
        midpoints = np.array([], dtype=float)
    else:
        all_extrema = np.sort(np.concatenate([peak_x, min_x]))
        if all_extrema.size > 1:
            spacings = np.diff(all_extrema)
            midpoints = (all_extrema[:-1] + all_extrema[1:]) / 2.0
        else:
            spacings = np.array([], dtype=float)
            midpoints = np.array([], dtype=float)

    return peak_x, min_x, spacings, midpoints


def gaussian(x: np.ndarray, A: float, mu: float, sigma: float, c: float) -> np.ndarray:
    """Simple Gaussian with additive baseline: A * exp(-0.5*((x-mu)/sigma)**2) + c"""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def asym_gaussian(x: np.ndarray, A: float, mu: float, sigma_l: float, sigma_r: float, c: float) -> np.ndarray:
    """Asymmetric (split) Gaussian: left/right sigma.

    For x <= mu uses sigma_l, for x > mu uses sigma_r. This models longer tails on one side.
    """
    x = np.asarray(x)
    sigma = np.where(x <= mu, sigma_l, sigma_r)
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def find_and_fit_gaussians(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    distance: Optional[float] = None,
    max_peaks: int = 10,
    prefer_detected_center: bool = True,
    center_tolerance: Optional[float] = None,
) -> List[Tuple[float, float, float, float, float]]:
    """Detect peaks in `y` and fit a Gaussian to each peak.

    Returns a list of tuples `(A, mu, sigma, c)` for each fitted peak.
    """
    # Adaptive detection defaults
    n = len(x)
    dx = np.mean(np.diff(x)) if n > 1 else 1.0
    y_med = float(np.median(y)) if len(y) > 0 else 0.0
    y_max = float(np.max(y)) if len(y) > 0 else 0.0
    y_min = float(np.min(y)) if len(y) > 0 else 0.0
    y_range = y_max - y_min if y_max != y_min else 0.0

    # estimate noise with MAD (robust) and derive a sensible prominence
    mad = float(np.median(np.abs(y - y_med))) if len(y) > 0 else 0.0
    if prominence is None:
        # choose either a small multiple of the MAD or a fraction of the full range
        prominence = max(mad * 3.0, FIND_PEAKS_PROMINENCE_FACTOR * y_range)

    # choose a minimum inter-peak distance in samples if not provided
    if distance is None:
        distance_samples = max(1, int(round(FIND_PEAKS_DISTANCE_FRACTION * n)))
    else:
        # if user provided a scalar, treat it as x-units and convert to samples
        if np.isscalar(distance):
            distance_samples = max(1, int(round(float(distance) / dx)))
        else:
            # assume it's already in samples or an index-like value
            distance_samples = int(distance)

    # detect candidate peaks using scipy when available
    try:
        from scipy.signal import find_peaks  # type: ignore

        find_kwargs = {"prominence": prominence, "distance": distance_samples}
        if height is not None:
            find_kwargs["height"] = height

        peaks_idx, props = find_peaks(y, **find_kwargs)
        # report chosen detection params for debugging/tuning
        if PRINT_OPTIONS.get("print_detected_extrema", True):
            print(f"find_peaks used: prominence={prominence:.3g}, distance_samples={distance_samples}")
    except Exception:
        # fallback simple local maxima
        if len(y) < 3:
            peaks_idx = np.array([], dtype=int)
        else:
            inner = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
            peaks_idx = np.nonzero(inner)[0] + 1

    if peaks_idx.size == 0:
        return []

    # take strongest peaks first (by amplitude) up to max_peaks
    peak_vals = y[peaks_idx]
    order = np.argsort(peak_vals)[::-1]
    # try to compute widths for better initial sigma estimates
    try:
        from scipy.signal import peak_widths  # type: ignore

        widths_res = peak_widths(y, peaks_idx, rel_height=0.5)
        widths_samples = np.asarray(widths_res[0])  # widths in samples aligned with peaks_idx
    except Exception:
        widths_samples = np.full(peaks_idx.shape, fill_value=5.0)

    # reorder peaks and widths together
    peaks_idx = peaks_idx[order]
    widths_samples = widths_samples[order]
    peaks_idx = peaks_idx[:max_peaks]
    widths_samples = widths_samples[:max_peaks]

    # return list of tuples: (detected_x, A, mu, sigma, c)
    fits: List[Tuple[float, float, float, float, float]] = []
    n = len(x)
    dx = np.mean(np.diff(x)) if n > 1 else 1.0
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_range = x_max - x_min if x_max != x_min else 1.0

    for i, pk in enumerate(peaks_idx):
        width_samples = float(widths_samples[i]) if i < len(widths_samples) else 5.0
        # determine window: half distance to nearest other selected peak or default
        if len(peaks_idx) > 1:
            sorted_peaks = np.sort(peaks_idx)
            pos = np.searchsorted(sorted_peaks, pk)
            if pos == 0:
                right = sorted_peaks[1]
                left = max(0, pk - (right - pk))
            elif pos == len(sorted_peaks) - 1:
                left = sorted_peaks[-2]
                right = min(n - 1, pk + (pk - left))
            else:
                left = sorted_peaks[pos - 1]
                right = sorted_peaks[pos + 1]
            half_window = max(3, int(round((pk - left) / 2)))
            l_idx = max(0, pk - half_window * 2)
            r_idx = min(n, pk + half_window * 2 + 1)
        else:
            l_idx = max(0, pk - 10)
            r_idx = min(n, pk + 11)

        x_win = x[l_idx:r_idx]
        y_win = y[l_idx:r_idx]
        if x_win.size < 5:
            continue

        # initial guesses
        baseline = float(np.median(y_win))
        A0 = float(y[pk] - baseline)
        mu0 = float(x[pk])
        # use peak width estimate if available to set sigma0 (FWHM -> sigma)
        sigma0 = max(1e-6, (width_samples * dx) / 2.355)
        # ensure sigma0 not unreasonably large relative to window
        sigma0 = min(sigma0, max(1e-6, (x_win[-1] - x_win[0]) / 2.0))
        c0 = baseline

        p0 = [A0, mu0, sigma0, c0]

        # bounds: A can be negative or positive; sigma positive; mu within a tight window
        mu_tol = max((width_samples * dx) / 2.0, dx)
        mu_lower = max(x_win[0], mu0 - mu_tol)
        mu_upper = min(x_win[-1], mu0 + mu_tol)
        sigma_lower = max(1e-9, sigma0 * 0.2)
        sigma_upper = max(sigma0 * 5.0, (x_win[-1] - x_win[0]))

        lower = [-np.inf, mu_lower, sigma_lower, -np.inf]
        upper = [np.inf, mu_upper, sigma_upper, np.inf]

        # decide center tolerance (how far mu may move and still be considered matching)
        if center_tolerance is None:
            # default: a fraction of the estimated width in x-units (quarter of FWHM)
            center_tolerance = max(dx, (width_samples * dx) * 0.001)

        # --- Always attempt BOTH free symmetric and free asymmetric fits ---
        sym_result = None
        asym_result = None

        # Symmetric free fit (mu allowed to vary within bounds)
        try:
            l_sym, u_sym = _ensure_strict_bounds(lower, upper)
            popt_sym, _ = curve_fit(gaussian, x_win, y_win, p0=p0, bounds=(l_sym, u_sym), maxfev=20000)
            A_sym, mu_sym, sigma_sym, c_sym = float(popt_sym[0]), float(popt_sym[1]), float(abs(popt_sym[2])), float(popt_sym[3])
            resid_sym = y_win - gaussian(x_win, A_sym, mu_sym, sigma_sym, c_sym)
            ssr_sym = float(np.sum(resid_sym ** 2))
            sym_result = ("sym", A_sym, mu_sym, sigma_sym, c_sym, ssr_sym)
        except Exception as e:
            sym_result = None
            if PRINT_OPTIONS.get("print_ch2_fits", True):
                print(f"curve_fit (free symmetric) failed at peak mu0={mu0}: {e}")
                traceback.print_exc()

        # Asymmetric free fit (mu allowed to vary)
        try:
            sigma_l0 = sigma0
            sigma_r0 = max(sigma0 * 1.5, sigma0 + dx)
            p0_asym = [A0, mu0, sigma_l0, sigma_r0, c0]
            lower_asym = [-np.inf, mu_lower, 1e-9, 1e-9, -np.inf]
            upper_asym = [np.inf, mu_upper, sigma_upper, sigma_upper * 5.0, np.inf]
            l_asym, u_asym = _ensure_strict_bounds(lower_asym, upper_asym)
            popt_asym, _ = curve_fit(asym_gaussian, x_win, y_win, p0=p0_asym, bounds=(l_asym, u_asym), maxfev=20000)
            A_asym = float(popt_asym[0])
            mu_asym = float(popt_asym[1])
            sigma_l_asym = float(abs(popt_asym[2]))
            sigma_r_asym = float(abs(popt_asym[3]))
            c_asym = float(popt_asym[4])
            resid_asym = y_win - asym_gaussian(x_win, A_asym, mu_asym, sigma_l_asym, sigma_r_asym, c_asym)
            ssr_asym = float(np.sum(resid_asym ** 2))
            asym_result = ("asym", A_asym, mu_asym, sigma_l_asym, sigma_r_asym, c_asym, ssr_asym)
        except Exception as e:
            asym_result = None
            if PRINT_OPTIONS.get("print_ch2_fits", True):
                print(f"curve_fit (free asymmetric) failed at peak mu0={mu0}: {e}")
                traceback.print_exc()

        chosen = None
        chosen_desc = None

        # If both fits succeeded, choose the one with smaller SSR
        if sym_result is not None and asym_result is not None:
            ssr_sym = sym_result[-1]
            ssr_asym = asym_result[-1]
            if ssr_sym <= ssr_asym:
                chosen = ("sym",) + tuple(sym_result[1:-1])
                chosen_desc = f"free_sym (ssr={ssr_sym:.3g} <= {ssr_asym:.3g})"
            else:
                chosen = ("asym",) + tuple(asym_result[1:-1])
                chosen_desc = f"free_asym (ssr={ssr_asym:.3g} < {ssr_sym:.3g})"
        elif sym_result is not None:
            chosen = ("sym",) + tuple(sym_result[1:-1])
            chosen_desc = "free_sym (asym failed)"
        elif asym_result is not None:
            chosen = ("asym",) + tuple(asym_result[1:-1])
            chosen_desc = "free_asym (sym failed)"
        else:
            # Both free fits failed: fall back to previous fixed-center attempts
            # try fixed symmetric
            try:
                lower_fixed = [-np.inf, mu0, sigma_lower, -np.inf]
                upper_fixed = [np.inf, mu0, sigma_upper, np.inf]
                l2, u2 = _ensure_strict_bounds(lower_fixed, upper_fixed)
                popt_f, _ = curve_fit(gaussian, x_win, y_win, p0=p0, bounds=(l2, u2), maxfev=10000)
                A_f, mu_f, sigma_f, c_f = float(popt_f[0]), float(popt_f[1]), float(abs(popt_f[2])), float(popt_f[3])
                chosen = ("sym", A_f, mu_f, sigma_f, c_f)
                chosen_desc = "fixed_sym (fallback)"
            except Exception as e:
                if PRINT_OPTIONS.get("print_ch2_fits", True):
                    print(f"curve_fit (fixed symmetric) failed at peak mu0={mu0}: {e}")
                    traceback.print_exc()
                # try fixed asymmetric
                try:
                    p0_asym = [A0, mu0, sigma_l0, sigma_r0, c0]
                    lower_asym_fix = [-np.inf, mu0, 1e-9, 1e-9, -np.inf]
                    upper_asym_fix = [np.inf, mu0, sigma_upper, sigma_upper * 5.0, np.inf]
                    l2a, u2a = _ensure_strict_bounds(lower_asym_fix, upper_asym_fix)
                    popt_a, _ = curve_fit(asym_gaussian, x_win, y_win, p0=p0_asym, bounds=(l2a, u2a), maxfev=10000)
                    A_a = float(popt_a[0])
                    mu_a = float(popt_a[1])
                    sigma_la = float(abs(popt_a[2]))
                    sigma_ra = float(abs(popt_a[3]))
                    c_a = float(popt_a[4])
                    chosen = ("asym", A_a, mu_a, sigma_la, sigma_ra, c_a)
                    chosen_desc = "fixed_asym (fallback)"
                except Exception as e2:
                    if PRINT_OPTIONS.get("print_ch2_fits", True):
                        print(f"curve_fit (fixed asymmetric) failed at peak mu0={mu0}: {e2}")
                        traceback.print_exc()

        if chosen is None:
            # no successful fit
            continue

        detected_x = float(x[pk])
        if chosen[0] == "sym":
            _, A, mu, sigma, c = chosen
            fits.append((detected_x, "sym", A, mu, sigma, c))
            try:
                offset = mu - detected_x
                if PRINT_OPTIONS.get("print_ch2_fits", True):
                    print(f"peak at {detected_x:.6g}: chosen fit={chosen_desc}, mu_offset={offset:.3g}")
            except Exception:
                pass
        elif chosen[0] == "asym":
            # chosen asym tuple may be ("asym", A, mu, sigma_l, sigma_r, c)
            _, A, mu, sigma_l, sigma_r, c = chosen
            fits.append((detected_x, "asym", A, mu, sigma_l, sigma_r, c))
            try:
                offset = mu - detected_x
                if PRINT_OPTIONS.get("print_ch2_fits", True):
                    print(
                        f"peak at {detected_x:.6g}: chosen fit={chosen_desc} (asym), mu_offset={offset:.3g}, sigma_l={sigma_l:.3g}, sigma_r={sigma_r:.3g}"
                    )
            except Exception:
                pass

    return fits


def plot_ch2_doppler_and_overlay(
    x: np.ndarray,
    summed_ch1: np.ndarray,
    final_path: Path,
    doppler_path: Path,
    exp_params: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Load CH2 from `final_path` and `doppler_path`, compute doppler - final, and plot.

    Also overlay the resulting CH2 difference with the CH1 difference (CSVFINAL CH1 - sum).
    """
    # Use the shared helper to compute aligned difference signals
    src_final, diff_ch1, diff_ch2 = compute_ch2_diffs(final_path, doppler_path, x, summed_ch1)

    # Plot CH2 doppler - final alone (use CSVFINAL axis `src_final`)
    plt.figure()
    # prepare rescale state variables before use
    used_u = False
    u_vals = None
    # choose plotting x: rescaled u (when available) or original src_final
    x_plot = src_final
    lbl = "FINAL_DOPPLER CH2 - CSVFINAL CH2"
    plt.plot(x_plot, diff_ch2, label=lbl)

    # If an exponential spacing fit is available, compute the rescaled
    # coordinate u(x) and perform Gaussian fitting in that coordinate.
    x_for_fit = src_final
    y_for_fit = diff_ch2
    if exp_params is not None:
        try:
            a_e, b_e, c_e = exp_params
            u_vals = compute_u(src_final, a_e, b_e, c_e)
            x_end_u = float(src_final[-1])
            scale_end_u = a_e * np.exp(b_e * x_end_u) + c_e
            u_vals = scale_end_u * u_vals
            u_vals = u_vals - float(u_vals[0])
            x_for_fit = u_vals
            y_for_fit = diff_ch2
            used_u = True
            if PRINT_OPTIONS.get("print_ch2_fits", True):
                print(f"Using rescaled coordinate u(x) for CH2 fitting (scale_end={scale_end_u:.6g})")
            # replot CH2 difference on the rescaled axis so fits align
            try:
                plt.cla()
                lbl = "FINAL_DOPPLER CH2 - CSVFINAL CH2 (rescaled)"
                plt.plot(u_vals, diff_ch2, label=lbl)
            except Exception:
                # best-effort: ignore plotting errors here
                pass
        except Exception:
            used_u = False

    # Fit Gaussians to the prominent CH2 peaks and overlay the fits
    fits = find_and_fit_gaussians(x_for_fit, y_for_fit)
    if fits:
        if PRINT_OPTIONS.get("print_ch2_fits", True):
            print("Fitted Gaussian parameters for CH2 peaks (detected_x, ...):")
        first = True
        for idx, fit in enumerate(fits, start=1):
            detected_x = fit[0]
            ftype = fit[1]
            if ftype == "sym":
                _, _, A, mu, sigma, c = fit
                if PRINT_OPTIONS.get("print_ch2_fits", True):
                    print(
                        f"  peak {idx}: model=sym, detected_x={detected_x:.6g}, A={A:.6g}, mu={mu:.6g}, sigma={sigma:.6g}, c={c:.6g}"
                    )
                xf = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
                yf = gaussian(xf, A, mu, sigma, c)
                lbl = f"Gaussian fit #{idx}" if first else None
                # Plot fit in the same coordinate used for fitting
                plt.plot(xf, yf, "--", color="magenta", alpha=0.9, label=lbl)
                first = False
            elif ftype == "asym":
                # asymmetric fit tuple: (detected_x, 'asym', A, mu, sigma_l, sigma_r, c)
                _, _, A, mu, sigma_l, sigma_r, c = fit
                if PRINT_OPTIONS.get("print_ch2_fits", True):
                    print(
                        f"  peak {idx}: model=asym, detected_x={detected_x:.6g}, A={A:.6g}, mu={mu:.6g}, sigma_left={sigma_l:.6g}, sigma_right={sigma_r:.6g}, c={c:.6g}"
                    )
                xf = np.linspace(mu - 4 * max(sigma_l, sigma_r), mu + 4 * max(sigma_l, sigma_r), 300)
                yf = asym_gaussian(xf, A, mu, sigma_l, sigma_r, c)
                lbl = f"Asym Gaussian fit #{idx}" if first else None
                plt.plot(xf, yf, "--", color="orange", alpha=0.9, label=lbl)
                # mark detected center and fitted center in the fit coordinate
                plt.plot(detected_x, np.interp(detected_x, x_for_fit, y_for_fit), "x", color="black")
                plt.plot(mu, np.interp(mu, x_for_fit, y_for_fit), "o", color="orange")
                first = False
        # re-show legend so fitted curves appear
        plt.legend()
    plt.xlabel("Source")
    plt.ylabel("Voltage difference (CH2)")
    plt.title("CH2 Doppler difference (FINAL_DOPPLER - CSVFINAL)")
    plt.grid(True)
    plt.legend()

    # Plot overlay of CH1 and CH2 differences
    # Interpolate signals locally for plotting if axes differ. Do NOT modify
    # the original data arrays; interpolation is used only for visualization.
    plt.figure()
    # axes for the two difference signals (both diffs are aligned to `src_final`)
    axis_ch1 = src_final
    axis_ch2 = src_final

    # build a common x for plotting that contains all sample points (sorted unique)
    try:
        common_x = np.unique(np.concatenate([axis_ch1, axis_ch2, x]))
    except Exception:
        # fallback: use the provided x if concatenation fails
        common_x = x

    # interpolate each difference onto the common axis for plotting only
    try:
        plot_ch1 = np.interp(common_x, axis_ch1, diff_ch1)
    except Exception:
        # as a safe fallback, attempt interpolation using 'x' as axis
        plot_ch1 = np.interp(common_x, x, diff_ch1)

    try:
        plot_ch2 = np.interp(common_x, axis_ch2, diff_ch2)
    except Exception:
        plot_ch2 = np.interp(common_x, x, diff_ch2)

    plt.plot(common_x, plot_ch1, label="CSVFINAL CH1 - sum (CH1 diff)")
    plt.plot(common_x, plot_ch2, label="FINAL_DOPPLER CH2 - CSVFINAL CH2 (CH2 diff)")
    plt.xlabel("Source")
    plt.ylabel("Voltage difference")
    plt.title("Comparison: CH1 difference vs CH2 Doppler difference")
    plt.grid(True)
    plt.legend()


def main() -> None:
    covlon = DATA_DIR / "CSVCOVLON.csv"
    covshort = DATA_DIR / "CSVCOVSHORT.csv"
    csv_final = DATA_DIR / "CSVFINAL.csv"
    final_doppler = DATA_DIR / "FINAL_DOPPLER.csv"

    if not (covlon.exists() and covshort.exists() and csv_final.exists() and final_doppler.exists()):
        raise FileNotFoundError("One or more required CSV files are missing.")

    x_values, summed_ch1 = sum_ch1_signals(covlon, covshort)
    x_values, summed_ch1 = restrict_time_window(x_values, summed_ch1)
    x_values, summed_ch1 = apply_sampling_stride(x_values, summed_ch1)

    # Compute CSVFINAL CH1 - summed now so we can detect peaks and spacings
    src_final, final_ch1 = load_channel(csv_final, CHANNEL)
    src_final, final_ch1 = restrict_time_window(src_final, final_ch1)
    src_final, final_ch1 = src_final[::BASE_DOWNSAMPLE], final_ch1[::BASE_DOWNSAMPLE]
    src_final, final_ch1 = apply_sampling_stride(src_final, final_ch1)
    if len(src_final) != len(x_values) or not np.allclose(src_final, x_values):
        raise ValueError("CSVFINAL axis does not match summed traces (pre-plot peak check).")
    diff = final_ch1 - summed_ch1

    # Detect peaks and minima, then print spacings between consecutive extrema
    peak_positions, min_positions, spacings, midpoints = detect_peaks_and_spacings(x_values, diff)
    if PRINT_OPTIONS.get("print_detected_extrema", True):
        print(f"Detected {len(peak_positions)} peak(s):", peak_positions)
        print(f"Detected {len(min_positions)} minima(s):", min_positions)
    if PRINT_OPTIONS.get("print_spacing_fits", True):
        print(f"Spacings between consecutive extrema (count={len(spacings)}):", spacings)

    # Plot spacings vs mean position of each consecutive extrema pair
    # We'll compute rescaled spacings and store them; plotting is controlled by PLOT_OPTIONS
    rescaled_spacings_plot = None
    if len(spacings) > 0:
        # exclude the first spacing value as requested
        if len(spacings) > 1:
            spacings_plot = spacings[1:]
            midpoints_plot = midpoints[1:]
        else:
            # nothing to plot after exclusion
            spacings_plot = np.array([], dtype=float)
            midpoints_plot = np.array([], dtype=float)

        if PLOT_OPTIONS.get("show_spacings", True):
            plt.figure()
            plt.plot(midpoints_plot, spacings_plot, "o-", color="green", label="measured spacings")

        # Only attempt fits if there are sufficient points. Prefer the
        # user-requested excluded-first selection, but if that leaves too few
        # points, fall back to fitting the full spacings so we can still
        # compute a rescaling when possible.
        exp_params = None
        fit_used = "excluded"
        fit_x = midpoints_plot
        fit_y = spacings_plot
        if len(fit_x) < 2 and len(midpoints) >= 2:
            # fallback to full spacings
            fit_used = "full"
            fit_x = midpoints
            fit_y = spacings

        if len(fit_x) >= 2:
            # Exponential fit only: fit a*exp(b*x)+c
            x_fit = np.linspace(float(np.min(fit_x)), float(np.max(fit_x)), 200)
            try:
                a_exp, b_exp, c_exp = fit_exponential(fit_x, fit_y)
                y_exp = a_exp * np.exp(b_exp * x_fit) + c_exp
                exp_params = (a_exp, b_exp, c_exp)
                if PLOT_OPTIONS.get("show_spacings", True):
                    # plot the fit curve clipped to the plotted midpoints range
                    if len(midpoints_plot) > 0:
                        x_min_plot = float(np.min(midpoints_plot))
                        x_max_plot = float(np.max(midpoints_plot))
                        mask = (x_fit >= x_min_plot) & (x_fit <= x_max_plot)
                        if np.any(mask):
                            plt.plot(x_fit[mask], y_exp[mask], "--", color="red", alpha=0.9, label="exp fit")
                    else:
                        plt.plot(x_fit, y_exp, "--", color="red", alpha=0.9, label="exp fit")
            except Exception as e:
                if PRINT_OPTIONS.get("print_spacing_fits", True):
                    print(f"Exponential fit failed (used={fit_used}): {e}")
                exp_params = None

        if PLOT_OPTIONS.get("show_spacings", True):
            plt.xlabel("Source (mean of consecutive extrema)")
            plt.ylabel("Spacing between consecutive extrema")
            plt.title("CH1 extrema spacings vs mean position (first spacing excluded)")
            plt.grid(True)
            plt.legend()

        # Print fit parameters
        if PRINT_OPTIONS.get("print_spacing_fits", True):
            print("Spacing exponential fit (first spacing excluded unless fallback used):")
            if exp_params is not None:
                a_exp, b_exp, c_exp = exp_params
                print(f"  exp fit: a={a_exp:.6g}, b={b_exp:.6g}, c={c_exp:.6g} (fit_used={fit_used})")

        # Apply x-rescaling so extrema become equi-spaced in the new coordinate u(x).
        if exp_params is not None:
            try:
                # reconstruct full extrema positions from detected peaks/minima
                all_extrema = np.sort(np.concatenate([peak_positions, min_positions]))
                if all_extrema.size >= 2:
                    u_all = compute_u(all_extrema, a_exp, b_exp, c_exp)
                    # Multiply the rescaled coordinate by the fitted spacing value
                    # at the end of the extrema range so the u scale matches the
                    # spacing magnitude at the upper bound of the data.
                    x_end = float(all_extrema[-1])
                    scale_end = a_exp * np.exp(b_exp * x_end) + c_exp
                    u_all = scale_end * u_all
                    # normalize so first value is zero (constant offset irrelevant)
                    u_all = u_all - float(u_all[0])
                    spacings_u_all = np.diff(u_all)
                    # match the previously excluded-first spacing selection
                    if len(spacings_u_all) > 1:
                        spacings_u_plot = spacings_u_all[1:]
                    else:
                        spacings_u_plot = np.array([], dtype=float)
                    # store rescaled spacings and overlay them on the spacing
                    # plot (no y-rescaling) if spacing plotting is enabled.
                    rescaled_spacings_plot = (midpoints_plot.copy(), np.asarray(spacings_u_plot))
                    if PRINT_OPTIONS.get("print_spacing_fits", True):
                        print(f"  rescaled spacings (mean,std) = ({np.mean(spacings_u_plot):.6g}, {np.std(spacings_u_plot):.6g})")
                    if PLOT_OPTIONS.get("show_spacings", True):
                        try:
                            if len(spacings_u_plot) == len(midpoints_plot):
                                plt.plot(midpoints_plot, spacings_u_plot, "s-", color="orange", label="rescaled spacings (u)")
                            else:
                                n = min(len(spacings_u_plot), len(midpoints_plot))
                                if n > 0:
                                    plt.plot(midpoints_plot[:n], spacings_u_plot[:n], "s-", color="orange", label="rescaled spacings (u)")
                            plt.legend()
                        except Exception:
                            if PRINT_OPTIONS.get("print_spacing_fits", True):
                                print("Failed plotting rescaled spacings overlay")
            except Exception as e:
                if PRINT_OPTIONS.get("print_spacing_fits", True):
                    print(f"Rescaling of x-axis failed: {e}")
                    traceback.print_exc()

            # Additionally: plot CH1 and CH2 differences after applying
            # the exponential rescaling u(x) in a separate figure. This
            # plotting is controlled by PLOT_OPTIONS, not PRINT_OPTIONS.
            if PLOT_OPTIONS.get("show_rescaled_diffs", True):
                try:
                    a_e, b_e, c_e = exp_params
                    # compute u on the CSVFINAL axis and shift to zero
                    src_final_u, diff_ch1_u, diff_ch2_u = compute_ch2_diffs(csv_final, final_doppler, x_values, summed_ch1)
                    u_vals = compute_u(src_final_u, a_e, b_e, c_e)
                    # scale u by the fitted spacing value at the end of the
                    # CSVFINAL axis so the rescaled coordinate reflects the
                    # local spacing magnitude at the upper bound.
                    x_end_u = float(src_final_u[-1])
                    scale_end_u = a_e * np.exp(b_e * x_end_u) + c_e
                    u_vals = scale_end_u * u_vals
                    u_vals = u_vals - float(u_vals[0])
                    plt.figure()
                    plt.plot(u_vals, diff_ch1_u, label="CH1 diff (rescaled)")
                    plt.plot(u_vals, diff_ch2_u, label="CH2 diff (rescaled)")
                    plt.xlabel("u(x) (rescaled source)")
                    plt.ylabel("Voltage difference (rescaled)")
                    plt.title("CH1 and CH2 differences vs rescaled coordinate u(x)")
                    plt.grid(True)
                    plt.legend()
                except Exception as _e:
                    if PRINT_OPTIONS.get("print_spacing_fits", True):
                        print(f"Failed to plot rescaled diffs: {_e}")
        else:
            # Plot but no fits possible
            plt.xlabel("Source (mean of consecutive extrema)")
            plt.ylabel("Spacing between consecutive extrema")
            plt.title("CH1 extrema spacings vs mean position (insufficient points for fits)")
            plt.grid(True)

    # Plot results (controlled by PLOT_OPTIONS)
    if PLOT_OPTIONS.get("show_sum_and_fits", True):
        plot_sum_and_fits(x_values, summed_ch1)
    if PLOT_OPTIONS.get("show_difference", True):
        plot_difference(x_values, summed_ch1, csv_final, peak_positions, min_positions)
    # Plot CH2 doppler difference and overlay with CH1 difference
    if PLOT_OPTIONS.get("show_ch2", True):
        plot_ch2_doppler_and_overlay(x_values, summed_ch1, csv_final, final_doppler, exp_params=exp_params)

    # Note: rescaled spacings are overlaid on the spacing plot earlier when
    # `PLOT_OPTIONS['show_spacings']` is True; no separate figure is created.
    plt.show()


if __name__ == "__main__":
    main()
