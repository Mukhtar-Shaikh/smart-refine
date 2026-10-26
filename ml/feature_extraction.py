
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def extract_features(signal: np.ndarray) -> dict:
    """Calculate basic features from a vibration signal."""
    signal = np.asarray(signal, dtype=float).ravel()

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    rms = np.sqrt(np.mean(signal ** 2))
    std_dev = np.std(signal)
    peak_to_peak = np.ptp(signal)

    return {
        "rms": float(rms),
        "std_dev": float(std_dev),
        "peak_to_peak": float(peak_to_peak),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "raw" / "97.mat"

    if not file_path.exists():
        raise FileNotFoundError(f"Recording not found: {file_path}")

    data = loadmat(file_path)

    # Load the drive-end vibration channel.
    signal = np.asarray(data["X097_DE_time"]).ravel()

    print(f"File: {file_path.name}")
    print(f"Number of samples: {signal.size}")

    features = extract_features(signal)

    print("\nFeatures for the complete recording:")
    for name, value in features.items():
        print(f"{name}: {value:.6f}")

    # Plot a short section of the original signal.
    count = min(2000, signal.size)
    sample_indices = np.arange(count)

    plt.figure(figsize=(12, 4))
    plt.plot(sample_indices, signal[:count], linewidth=0.8)
    plt.title("CWRU Bearing Vibration — Drive-End Channel")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
