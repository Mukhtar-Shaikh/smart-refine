
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "comparison_features.csv"
MODEL_DIR = PROJECT_ROOT / "ml" / "model"
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"

FEATURES = ["rms", "std_dev", "peak_to_peak"]

# 1. Load the extracted window features
df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + ["label"]
missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"CSV is missing columns: {sorted(missing)}")

df = df.dropna(subset=required_columns)

# 2. Separate normal and faulty windows
normal = df[df["label"] == "normal"].copy()
faulty = df[df["label"] == "inner_race_fault"].copy()

if len(normal) < 2 or faulty.empty:
    raise ValueError(
        "Need at least 2 normal windows and 1 faulty window."
    )

# 3. Hold out some normal windows for testing
normal_train, normal_test = train_test_split(
    normal,
    test_size=0.30,
    random_state=42,
)

# 4. Train only on normal vibration
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)

model.fit(normal_train[FEATURES])

# 5. Test on unseen normal windows and faulty windows
test = pd.concat([normal_test, faulty], ignore_index=True)

# IsolationForest returns 1 for normal and -1 for anomaly
predictions = model.predict(test[FEATURES])

y_true = (test["label"] != "normal").astype(int)
y_pred = (predictions == -1).astype(int)

print("\nConfusion matrix")
print("Rows = actual [normal, fault]")
print("Columns = predicted [normal, anomaly]")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "fault"],
        zero_division=0,
    )
)

# 6. Save the trained model and its feature configuration
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
    },
    MODEL_PATH,
)

print(f"\nModel saved to: {MODEL_PATH}")
