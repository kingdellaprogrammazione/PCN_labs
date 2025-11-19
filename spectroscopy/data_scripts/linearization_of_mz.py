#!/usr/bin/env python3
"""Combine CH1 traces from two scope captures, fit, and compare."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit




DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHANNEL = "CH1"
MIN_TIME = -0.012
MAX_TIME = 0.0381
BASE_DOWNSAMPLE = 50  # Align high-resolution captures with CSVFINAL.
SAMPLE_STRIDE = 20  # Set >1 to average every N points after alignment/windowing.


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


def apply_sampling_stride(x: np.ndarray, *ys: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Average contiguous SAMPLE_STRIDE samples (no-op when stride == 1)."""
    if SAMPLE_STRIDE <= 1:
        return (x,) + tuple(ys)

    def block_mean(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        usable = len(array) - len(array) % SAMPLE_STRIDE
        if usable == 0:
            raise ValueError(
                f"Not enough samples ({len(array)}) for stride {SAMPLE_STRIDE}."
            )
        trimmed = array[:usable]
        return trimmed.reshape(-1, SAMPLE_STRIDE).mean(axis=1)

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

    plt.figure()
    plt.plot(x, summed, label="CH1 sum")
    plt.plot(x, slope * x + intercept, "--", label=f"Linear fit (y={slope:.3e}x+{intercept:.3e})")
    plt.plot(x, exp_model(x, a, b, c), ":", label=f"Exp fit (a={a:.3e}, b={b:.3e}, c={c:.3e})")
    plt.xlabel("Source")
    plt.ylabel("Voltage (scaled)")
    plt.title("Summed CH1 traces with fits")
    plt.legend()
    plt.grid(True)


def plot_difference(x: np.ndarray, summed: np.ndarray, final_path: Path) -> None:
    """Plot CSVFINAL CH1 minus the provided sum."""
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
        raise ValueError("CSVFINAL axis does not match summed traces.")
    diff = final_ch1 - summed

    plt.figure()
    plt.plot(x, diff, label="CSVFINAL CH1 - sum")
    plt.xlabel("Source")
    plt.ylabel("Voltage difference")
    plt.title("Difference between CSVFINAL CH1 and sum")
    plt.grid(True)
    plt.legend()


def main() -> None:
    covlon = DATA_DIR / "CSVCOVLON.csv"
    covshort = DATA_DIR / "CSVCOVSHORT.csv"
    csv_final = DATA_DIR / "CSVFINAL.csv"

    if not (covlon.exists() and covshort.exists() and csv_final.exists()):
        raise FileNotFoundError("One or more required CSV files are missing.")

    x_values, summed_ch1 = sum_ch1_signals(covlon, covshort)
    x_values, summed_ch1 = restrict_time_window(x_values, summed_ch1)
    x_values, summed_ch1 = apply_sampling_stride(x_values, summed_ch1)

    plot_sum_and_fits(x_values, summed_ch1)
    plot_difference(x_values, summed_ch1, csv_final)
    plt.show()


if __name__ == "__main__":
    main()
