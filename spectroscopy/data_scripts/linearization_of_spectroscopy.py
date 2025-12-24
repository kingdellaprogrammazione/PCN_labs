#!/usr/bin/env python3
"""Combine CH1 traces from two scope captures, fit, and compare."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np




DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHANNEL_SUM = "CH1"  # Channel used for CSVCOVLON/CSVCOVSHORT sum.
CHANNEL_REF = "CH2"  # Channel used for CSVFINAL/FINAL_DOPPLER comparison.
CHANNEL_REF_COMPARE = "CH3"  # Second reference channel to plot relative to CH2.
MIN_TIME = -0.012
MAX_TIME = 0.0381
BASE_DOWNSAMPLE = 50  # Align high-resolution captures with CSVFINAL.
SAMPLE_STRIDE = 20  # Set >1 to average every N points after alignment/windowing.
FINAL_CAPTURE_NAME = "CSVFINAL.csv"  # Toggle between CSVFINAL.csv and FINAL_DOPPLER.csv.


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
    src_a, ch1_a = load_channel(path_a, CHANNEL_SUM)
    src_b, ch1_b = load_channel(path_b, CHANNEL_SUM)

    if len(src_a) != len(src_b):
        raise ValueError("CSV files have different sample counts.")
    if not np.allclose(src_a, src_b):
        raise ValueError("Source axes differ; cannot sum traces safely.")

    return src_a[::2 * BASE_DOWNSAMPLE], (ch1_a + ch1_b)[::2 * BASE_DOWNSAMPLE]


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


def plot_sum_trace(x: np.ndarray, summed: np.ndarray) -> None:
    """Plot the summed CH1 trace for quick inspection."""
    plt.figure()
    plt.plot(x, summed, label="CSVCOVLON + CSVCOVSHORT")
    plt.xlabel("Source")
    plt.ylabel("Voltage (scaled)")
    plt.title("Summed CH1 trace (no fits)")
    plt.grid(True)
    plt.legend()


def plot_difference(x: np.ndarray, summed: np.ndarray, final_path: Path) -> None:
    """Plot (covlon + covshort) CH1 minus the chosen reference capture."""
    src_final, final_ref = load_channel(final_path, CHANNEL_REF)
    src_final, final_ref = restrict_time_window(src_final, final_ref)
    src_final, final_ref = src_final[::BASE_DOWNSAMPLE], final_ref[:: BASE_DOWNSAMPLE]
    src_final, final_ref = apply_sampling_stride(src_final, final_ref)
    if len(src_final) != len(x):
        raise ValueError(
            f"Sample counts differ (sum={len(x)}, final={len(src_final)}). "
            "Ensure the reference capture was recorded at matching resolution."
        )
    if not np.allclose(src_final, x):
        raise ValueError(f"{final_path.name} axis does not match summed traces.")
    diff = summed - final_ref

    plt.figure()
    plt.plot(x, diff, label=f"(covlon + covshort) - {final_path.stem}")
    plt.xlabel("Source")
    plt.ylabel("Voltage difference")
    plt.title(f"Difference between sum and {final_path.stem}")
    plt.grid(True)
    plt.legend()


def plot_reference_channel_difference(final_path: Path) -> None:
    """Plot the difference between two channels (CH2 - CH3) of the reference capture."""
    x_ref, ref_ch2 = load_channel(final_path, CHANNEL_REF)
    x_ref_ch3, ref_ch3 = load_channel(final_path, CHANNEL_REF_COMPARE)
    if len(x_ref) != len(x_ref_ch3) or not np.allclose(x_ref, x_ref_ch3):
        raise ValueError(f"{final_path.name} channels do not share the same axis.")

    x_ref, ref_ch2, ref_ch3 = restrict_time_window(x_ref, ref_ch2, ref_ch3)
    x_ref = x_ref[::BASE_DOWNSAMPLE]
    ref_ch2 = ref_ch2[::BASE_DOWNSAMPLE]
    ref_ch3 = ref_ch3[::BASE_DOWNSAMPLE]
    x_ref, ref_ch2, ref_ch3 = apply_sampling_stride(x_ref, ref_ch2, ref_ch3)

    plt.figure()
    plt.plot(x_ref, ref_ch2 - 0.170*ref_ch3, label=f"{CHANNEL_REF} - {CHANNEL_REF_COMPARE}")
    plt.xlabel("Source")
    plt.ylabel("Voltage difference")
    plt.title(f"{final_path.stem}: {CHANNEL_REF} minus {CHANNEL_REF_COMPARE}")
    plt.grid(True)
    plt.legend()


def main() -> None:
    covlon = DATA_DIR / "CSVCOVLON.csv"
    covshort = DATA_DIR / "CSVCOVSHORT.csv"
    final_capture = DATA_DIR / FINAL_CAPTURE_NAME

    if not (covlon.exists() and covshort.exists() and final_capture.exists()):
        raise FileNotFoundError("One or more required CSV files are missing.")

    x_values, summed_ch1 = sum_ch1_signals(covlon, covshort)
    x_values, summed_ch1 = restrict_time_window(x_values, summed_ch1)
    x_values, summed_ch1 = apply_sampling_stride(x_values, summed_ch1)

    plot_sum_trace(x_values, summed_ch1)
    plot_difference(x_values, summed_ch1, final_capture)
    plot_reference_channel_difference(final_capture)
    plt.show()


if __name__ == "__main__":
    main()
