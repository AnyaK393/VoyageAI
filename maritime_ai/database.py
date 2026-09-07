"""Persistence for the API: SQLite locally, PostgreSQL/Supabase in deployment."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DEFAULT_DATABASE_URL = f"sqlite:///{Path(__file__).resolve().parent.parent / 'data' / 'voyageai.db'}"


class Base(DeclarativeBase):
    pass


class MarketObservation(Base):
    __tablename__ = "market_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_date: Mapped[datetime] = mapped_column(DateTime, index=True, unique=True)
    freight_proxy_rate: Mapped[float] = mapped_column(Float)
    bunker_proxy: Mapped[float] = mapped_column(Float)
    coal_price: Mapped[float] = mapped_column(Float)
    usd_inr: Mapped[float] = mapped_column(Float)
    congestion_days: Mapped[float] = mapped_column(Float)
    provenance: Mapped[str] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(60))
    version: Mapped[str] = mapped_column(String(60))
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="PENDING")
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    destination_port: Mapped[str] = mapped_column(String(80))
    cargo_tonnes: Mapped[float] = mapped_column(Float)
    horizon_weeks: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(60))
    vessel: Mapped[str] = mapped_column(String(80))
    contract: Mapped[str] = mapped_column(String(80))
    all_in_usd_per_tonne: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer)
    risk_label: Mapped[str] = mapped_column(String(20))
    scenario_json: Mapped[str] = mapped_column(Text)


class BrokerTender(Base):
    """A broker's comparable all-in tender for a particular voyage brief."""

    __tablename__ = "broker_tenders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    broker_name: Mapped[str] = mapped_column(String(120), index=True)
    origin: Mapped[str] = mapped_column(String(120))
    destination_port: Mapped[str] = mapped_column(String(80), index=True)
    cargo_tonnes: Mapped[float] = mapped_column(Float)
    cargo_type: Mapped[str] = mapped_column(String(80), default="Thermal coal")
    laycan_start: Mapped[datetime] = mapped_column(DateTime)
    vessel: Mapped[str] = mapped_column(String(80))
    contract: Mapped[str] = mapped_column(String(80))
    voyages: Mapped[int] = mapped_column(Integer)
    all_in_usd_per_tonne: Mapped[float] = mapped_column(Float)
    valid_until: Mapped[datetime] = mapped_column(DateTime, index=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="SUBMITTED", index=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    category: Mapped[str] = mapped_column(String(60))
    severity: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)


class Database:
    def __init__(self, url: str | None = None):
        resolved_url = url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
        # Supabase URLs are often provided as postgres://; SQLAlchemy needs a driver.
        if resolved_url.startswith("postgres://"):
            resolved_url = "postgresql+psycopg://" + resolved_url.removeprefix("postgres://")
        elif resolved_url.startswith("postgresql://"):
            resolved_url = "postgresql+psycopg://" + resolved_url.removeprefix("postgresql://")
        self.url = resolved_url
        self.engine = create_engine(self.url, connect_args={"check_same_thread": False} if self.url.startswith("sqlite") else {})
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        # Small additive migration so existing local demo databases keep working.
        inspector = inspect(self.engine)
        if "broker_tenders" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("broker_tenders")}
            if "cargo_type" not in columns:
                with self.engine.begin() as connection:
                    connection.execute(text("ALTER TABLE broker_tenders ADD COLUMN cargo_type VARCHAR(80) NOT NULL DEFAULT 'Thermal coal'"))

    def session(self) -> Session:
        return self.session_factory()

    def market_count(self) -> int:
        with self.session() as session:
            return len(session.scalars(select(MarketObservation.id)).all())
