#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Per-channel (scale, offset) to align traces visually.
CHANNEL_TRANSFORMS: dict[str, tuple[float, float]] = {
    "CH1": (10.0, 0.0),
    "CH2": (10.0, -13.5),
}


def load_scope_csv(csv_path: Path):
    """Return source axis label/values plus a dict of channel traces."""
    with csv_path.open() as handle:
        reader = csv.reader(handle)
        header = units = None

        for row in reader:
            if row and row[0].strip() == "Source":
                header = [col.strip() for col in row]
                units = [col.strip() for col in next(reader, [])]
                break
        if header is None:
            raise ValueError("Could not find the 'Source' header row in the CSV.")

        data_rows = [row for row in reader if any(field.strip() for field in row)]

    source_label = header[0]
    source_values = [float(row[0]) for row in data_rows]

    channels = {}
    for idx, name in enumerate(header[1:], start=1):
        name = name.strip()
        if not name:
            continue
        channels[name] = [float(row[idx]) for row in data_rows]

    return source_label, units, source_values, channels


def transform_trace(channel: str, values: list[float]) -> list[float]:
    """Apply optional scale/offset per channel."""
    scale, offset = CHANNEL_TRANSFORMS.get(channel, (1.0, 0.0))
    return [value * scale + offset for value in values]


def plot_channels(csv_path: Path):
    source_label, units, source_values, channels = load_scope_csv(csv_path)

    fig, ax = plt.subplots()
    for channel, values in channels.items():
        ax.plot(source_values, transform_trace(channel, values), label=channel)

    ax.set_xlabel(source_label or "Source")
    y_unit = units[1] if units and len(units) > 1 else ""
    ax.set_ylabel(f"Value ({y_unit})" if y_unit else "Value")
    ax.set_title(f"{csv_path.stem} traces")
    ax.grid(True)
    ax.legend()
    plt.show()


def find_csv_files(base_dir: Path) -> list[Path]:
    """Find all CSV files in spectroscopy/ and spectroscopy/data."""
    csv_paths: list[Path] = []
    for directory in {base_dir, base_dir / "data"}:
        if directory.is_dir():
            csv_paths.extend(sorted(directory.glob("*.csv")))
    unique = []
    seen = set()
    for path in csv_paths:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        unique.append(path)
    if not unique:
        raise FileNotFoundError("No CSV files found in spectroscopy or spectroscopy/data.")
    return unique


def main():
    data_root = Path(__file__).resolve().parents[1]
    for csv_file in find_csv_files(data_root):
        print(f"Plotting {csv_file.name}")
        plot_channels(csv_file)


if __name__ == "__main__":
    main()
