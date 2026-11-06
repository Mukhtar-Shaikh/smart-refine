
from datetime import datetime
from typing import Literal


from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, HTTPException  # Added Depends and HTTPException
from sqlalchemy.orm import Session                   # Added
from sqlalchemy.exc import IntegrityError           # Added

from backend.database import SessionLocal           # Added
from backend.models import Telemetry                # Added

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
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/telemetry")
def receive_telemetry(event: TelemetryEvent,
    db: Session = Depends(get_db)):
    record = Telemetry(
        event_id=event.event_id,
        device_id=event.device_id,
        timestamp=event.timestamp.replace(tzinfo=None),
        source_file=event.source_file,
        window_index=event.window_index,
        rms=event.features.rms,
        std_dev=event.features.std_dev,
        peak_to_peak=event.features.peak_to_peak,
        status=event.status
    )

    # <-- Added database transaction block with error handling
    try:
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "accepted": True,
            "message": "Telemetry saved to MySQL",
            "database_id": record.id,
            "event_id": record.event_id
        }

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Duplicate event_id"
        )

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save telemetry"
        )