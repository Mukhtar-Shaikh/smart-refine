
from pathlib import Path
from scipy.io import loadmat
import numpy as np

# Find MATLAB recordings in the raw-data folder
data_folder = Path("data/raw")
mat_files = list(data_folder.glob("*.mat"))

if not mat_files:
    raise FileNotFoundError(
        "No .mat files found in data/raw. "
        "Download a CWRU recording first."
    )

# Inspect the first recording
file_path = mat_files[0]
print(f"Inspecting: {file_path.name}")

data = loadmat(file_path)

print("\nVariables stored in the file:")
for key, value in data.items():
    if not key.startswith("__"):
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value)}")

print("\nPossible drive-end vibration channels:")
for key, value in data.items():
    if key.endswith("_DE_time"):
        signal = np.asarray(value).squeeze()
        print(f"Channel: {key}")
        print(f"Number of samples: {signal.size}")
        print(f"First 10 samples: {signal[:10]}")
