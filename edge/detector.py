
from pathlib import Path
import pandas as pd 
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "model" / "isolation_forest.joblib"

WINDOW_SIZE = 1024

# Load the trained model and the exact feature order
saved = joblib.load(MODEL_PATH)
model = saved["model"]
FEATURES = saved["features"]


def extract_features(signal):
    """Extract the same features used during model training."""
    signal = np.asarray(signal, dtype=float)

    if len(signal) != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} samples.")

    return {
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "std_dev": float(np.std(signal)),
        "peak_to_peak": float(np.ptp(signal)),
    }


def detect_window(signal):
    """Return extracted features and the model's anomaly decision."""
    features = extract_features(signal)

    # row = [[features[name] for name in FEATURES]]
    # prediction = int(model.predict(row)[0])

    # Create a DataFrame with the features in the correct order
    row = pd.DataFrame(
    [[features[name] for name in FEATURES]],
    columns=FEATURES,
    )
    prediction = int(model.predict(row)[0])

    status = "ANOMALY" if prediction == -1 else "NORMAL"

    return {
        "features": features,
        "status": status,
    }



if __name__ == "__main__":
    from scipy.io import loadmat

    recordings = {
        "97.mat": "normal",
        "105.mat": "inner_race_fault",
    }

    for filename, expected_label in recordings.items():
        data_path = PROJECT_ROOT / "data" / "raw" / filename

        mat = loadmat(data_path)
        channel_name = next(
            key for key in mat
            if key.endswith("_DE_time")
        )
        signal = mat[channel_name].ravel()

        # Test consecutive, non-overlapping windows
        window_count = len(signal) // WINDOW_SIZE
        anomaly_count = 0

        for i in range(window_count):
            start = i * WINDOW_SIZE
            window = signal[start : start + WINDOW_SIZE]

            result = detect_window(window)

            if result["status"] == "ANOMALY":
                anomaly_count += 1

        anomaly_rate = anomaly_count / window_count * 100

        print(f"\nRecording: {filename}")
        print(f"Expected condition: {expected_label}")
        print(f"Windows tested: {window_count}")
        print(f"Anomalies detected: {anomaly_count}")
        print(f"Anomaly rate: {anomaly_rate:.1f}%")
