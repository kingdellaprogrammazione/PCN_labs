"""
Convert Dektak Vision64 .dat profilometer files into matplotlib figures.

The .dat file is Bruker/Veeco's undocumented proprietary binary format
(identified by the embedded "FABIO" / "$$ Saved with: 7.21.08 $$" header).
The scan/session metadata region has a fixed, scan-independent layout, but
the actual height trace lives in a single contiguous block near the end of
the file: byte offset 0x50ee, as little-endian float32, terminated by a
zero-padded footer. This was found by diffing two .dat files against each
other (identical bytes = shared fixed-format metadata, differing bytes =
per-scan data) and confirming the resulting values form a smooth, glitch
-free curve.

No calibration constants for the vertical/horizontal scale could be
recovered, so values are plotted in arbitrary units (raw stored values vs.
sample index).

Usage:
    python3 dat_to_figure.py gruppo1.dat gruppo1smaller.dat
"""

import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_OFFSET = 0x50EE


def read_dat_profile(path):
    data = Path(path).read_bytes()
    tail = data[DATA_OFFSET:]
    # trailing zero-padded footer: strip it so the plot doesn't flatline at 0
    trimmed = tail.rstrip(b"\x00")
    n = len(trimmed) - (len(trimmed) % 4)
    return np.frombuffer(trimmed[:n], dtype="<f4")


def save_figure(profile, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(profile, color="tab:red", linewidth=1.2)
    ax.set_xlabel("Sample index (a.u.)")
    ax.set_ylabel("Height (a.u.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def main(paths):
    for path in paths:
        path = Path(path)
        profile = read_dat_profile(path)
        out_path = path.with_suffix(".png")
        save_figure(profile, out_path, path.stem)


if __name__ == "__main__":
    args = sys.argv[1:] or ["gruppo1.dat", "gruppo1smaller.dat"]
    main(args)
