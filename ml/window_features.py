
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

WINDOW_SIZE = 1024


def extract_features(signal: np.ndarray) -> dict:
    """Calculate statistical features for one signal window."""
    signal = np.asarray(signal, dtype=float).ravel()

    if signal.size == 0:
        raise ValueError("Signal window cannot be empty.")

    return {
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "std_dev": float(np.std(signal)),
        "peak_to_peak": float(np.ptp(signal)),
    }


def extract_window_features(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Extract features from consecutive, non-overlapping windows."""
    signal = np.asarray(signal, dtype=float).ravel()

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if signal.size < window_size:
        raise ValueError("Signal is shorter than one complete window.")

    rows = []

    for start in range(0, signal.size - window_size + 1, window_size):
        window = signal[start : start + window_size]
        features = extract_features(window)
        features["start_sample"] = start
        rows.append(features)

    return pd.DataFrame(rows)


def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "raw" / "97.mat"

    if not file_path.exists():
        raise FileNotFoundError(f"Recording not found: {file_path}")

    data = loadmat(file_path)
    signal = np.asarray(data["X097_DE_time"]).ravel()

    features_df = extract_window_features(signal)

    print("Input samples:", signal.size)
    print("Window size:", WINDOW_SIZE)
    print("Complete windows:", len(features_df))
    print("\nFirst five rows:")
    print(features_df.head())

    output_folder = project_root / "data" / "processed"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "97_de_features.csv"
    features_df.to_csv(output_file, index=False)

    print(f"\nSaved features to: {output_file}")


if __name__ == "__main__":
    main()
