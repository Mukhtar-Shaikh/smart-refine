
from datetime import datetime
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Smart-Refine API",
    description="Telemetry ingestion API for predictive maintenance",
    version="0.1.0",
)


class VibrationFeatures(BaseModel):
    rms: float = Field(ge=0)
    std_dev: float = Field(ge=0)
    peak_to_peak: float = Field(ge=0)


class TelemetryEvent(BaseModel):
    event_id: str
    device_id: str
    timestamp: datetime
    source_file: str
    window_index: int = Field(ge=0)
    features: VibrationFeatures
    status: Literal["NORMAL", "ANOMALY"]


@app.get("/health")
def health():
    return {"status": "ok", "service": "smart-refine-api"}


@app.post("/telemetry")
def receive_telemetry(event: TelemetryEvent):
    return {
        "accepted": True,
        "event_id": event.event_id,
        "device_id": event.device_id,
        "status": event.status,
    }
