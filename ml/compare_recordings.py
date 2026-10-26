
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

# Confirm these labels against the CWRU metadata.
RECORDINGS = {
    "97.mat": "normal",
    "105.mat": "inner_race_fault",
}

WINDOW_SIZE = 1024


def extract_features(signal):
    signal = np.asarray(signal, dtype=float).ravel()

    return {
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "std_dev": float(np.std(signal)),
        "peak_to_peak": float(np.ptp(signal)),
    }


def process_recording(filename, label):
    file_path = DATA_FOLDER / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Download it before continuing."
        )

    data = loadmat(file_path)

    # Find the drive-end vibration channel.
    channel_names = [
        key for key in data
        if key.endswith("_DE_time")
    ]

    if len(channel_names) != 1:
        raise ValueError(
            f"Expected one drive-end channel in {filename}; "
            f"found {channel_names}"
        )

    signal = np.asarray(data[channel_names[0]]).ravel()
    rows = []

    for start in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = signal[start : start + WINDOW_SIZE]
        row = extract_features(window)
        row["recording"] = filename
        row["label"] = label
        row["start_sample"] = start
        rows.append(row)

    print(f"{filename}: {len(rows)} complete windows")
    return rows


def main():
    all_rows = []

    for filename, label in RECORDINGS.items():
        all_rows.extend(process_recording(filename, label))

    df = pd.DataFrame(all_rows)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / "comparison_features.csv"
    df.to_csv(output_file, index=False)

    print("\nFeature summary by recording:")
    print(
        df.groupby(["recording", "label"])[
            ["rms", "std_dev", "peak_to_peak"]
        ].mean().round(5)
    )

    print(f"\nSaved results to: {output_file}")

    # Compare RMS distributions, not just averages.
    df.boxplot(column="rms", by="label")
    plt.title("Vibration RMS by Recording Label")
    plt.suptitle("")
    plt.xlabel("Recording label")
    plt.ylabel("RMS amplitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
