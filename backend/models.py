from sqlalchemy import Column, BigInteger, String, DateTime, Integer, Float, TIMESTAMP, text
from backend.database import Base


class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_id = Column(String(100), unique=True, nullable=False)
    device_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    source_file = Column(String(100))
    window_index = Column(Integer, nullable=False)
    rms = Column(Float, nullable=False)
    std_dev = Column(Float, nullable=False)
    peak_to_peak = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)
    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )