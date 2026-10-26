
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "model" / "isolation_forest.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "105.mat"

BROKER = "127.0.0.1"
PORT = 1883
TOPIC = "smart-refine/telemetry"
WINDOW_SIZE = 1024
DEVICE_ID = "pump-01"


def extract_features(signal):
    signal = np.asarray(signal, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "std_dev": float(np.std(signal)),
        "peak_to_peak": float(np.ptp(signal)),
    }


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["features"]

    mat = loadmat(DATA_PATH)
    channel = next(k for k in mat if k.endswith("_DE_time"))
    signal = mat[channel].ravel()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="smart-refine-edge",
    )

    print(f"Connecting to MQTT broker at {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()

    try:
        window_count = min(10, len(signal) // WINDOW_SIZE)

        for index in range(window_count):
            start = index * WINDOW_SIZE
            window = signal[start : start + WINDOW_SIZE]
            features = extract_features(window)

            row = pd.DataFrame(
                [[features[name] for name in feature_names]],
                columns=feature_names,
            )
            prediction = int(model.predict(row)[0])
            status = "ANOMALY" if prediction == -1 else "NORMAL"

            payload = {
                "event_id": str(uuid.uuid4()),
                "device_id": DEVICE_ID,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_file": DATA_PATH.name,
                "window_index": index,
                "features": features,
                "status": status,
            }

            info = client.publish(
                TOPIC,
                json.dumps(payload),
                qos=1,
            )
            info.wait_for_publish(timeout=5)

            print(f"Published window {index + 1}: {status}")

            time.sleep(1)

    finally:
        client.loop_stop()
        client.disconnect()

    print("Finished publishing test telemetry.")


if __name__ == "__main__":
    main()
