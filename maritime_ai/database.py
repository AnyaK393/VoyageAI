"""Persistence for VoyageAI: SQLite locally, PostgreSQL/Supabase in deployment."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, inspect, select, text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

# Auto-load .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except Exception:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ and v:
                    os.environ[k] = v

DEFAULT_DATABASE_URL = f"sqlite:///{Path(__file__).resolve().parent.parent / 'data' / 'voyageai.db'}"


class Base(DeclarativeBase):
    pass


class MarketObservation(Base):
    __tablename__ = "market_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observation_date: Mapped[datetime] = mapped_column(DateTime, index=True, unique=True)
    freight_proxy_rate: Mapped[float] = mapped_column(Float)
    bunker_proxy: Mapped[float] = mapped_column(Float)
    coal_price: Mapped[float] = mapped_column(Float)
    usd_inr: Mapped[float] = mapped_column(Float)
    congestion_days: Mapped[float] = mapped_column(Float)
    provenance: Mapped[str] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class PortRecord(Base):
    __tablename__ = "ports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    port: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    max_draft_m: Mapped[float] = mapped_column(Float)
    max_loa_m: Mapped[float] = mapped_column(Float)
    max_beam_m: Mapped[float] = mapped_column(Float)
    max_dwt_tonnes: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class VesselRecord(Base):
    __tablename__ = "vessels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vessel: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    vessel_class: Mapped[str] = mapped_column(String(60))
    capacity_tonnes: Mapped[float] = mapped_column(Float)
    draft_m: Mapped[float] = mapped_column(Float)
    loa_m: Mapped[float] = mapped_column(Float)
    beam_m: Mapped[float] = mapped_column(Float)
    fuel_factor: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(60))
    version: Mapped[str] = mapped_column(String(60))
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="PENDING")
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    destination_port: Mapped[str] = mapped_column(String(80), index=True)
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
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


class TenderDecision(Base):
    """Immutable audit entry for tender triage, approval, and overrides."""

    __tablename__ = "tender_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    tender_id: Mapped[int] = mapped_column(Integer, index=True)
    action: Mapped[str] = mapped_column(String(30), index=True)
    actor_name: Mapped[str] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(Text, default="")
    system_recommended: Mapped[bool] = mapped_column(default=False)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    category: Mapped[str] = mapped_column(String(60))
    severity: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)


def normalize_database_url(url: str | None = None) -> str:
    """Resolve database URL and normalize Supabase/PostgreSQL connection schemes for SQLAlchemy."""
    raw_url = (url or os.getenv("DATABASE_URL") or "").strip()
    if not raw_url:
        return DEFAULT_DATABASE_URL

    driver_prefix = "postgresql+psycopg://"
    try:
        import psycopg  # noqa: F401
        driver_prefix = "postgresql+psycopg://"
    except ImportError:
        try:
            import psycopg2  # noqa: F401
            driver_prefix = "postgresql+psycopg2://"
        except ImportError:
            driver_prefix = "postgresql+psycopg://"

    if raw_url.startswith("postgres://"):
        return driver_prefix + raw_url.removeprefix("postgres://")
    elif raw_url.startswith("postgresql://"):
        return driver_prefix + raw_url.removeprefix("postgresql://")
    return raw_url


class Database:
    def __init__(self, url: str | None = None):
        self.raw_url = url or os.getenv("DATABASE_URL") or ""
        self.url = normalize_database_url(url)
        connect_args = {"check_same_thread": False} if self.url.startswith("sqlite") else {}
        try:
            self.engine = create_engine(self.url, connect_args=connect_args)
            self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        except Exception as e:
            if "psycopg" in str(e).lower() or "no module named" in str(e).lower():
                raise RuntimeError(
                    "PostgreSQL driver 'psycopg' is missing in your current Python environment.\n"
                    "Fix: Run `pip install 'psycopg[binary]>=3.2'`"
                ) from e
            raise e

    @property
    def is_supabase_or_postgres(self) -> bool:
        return self.url.startswith("postgresql")

    @property
    def database_type_display(self) -> str:
        if self.is_supabase_or_postgres:
            return "Supabase (PostgreSQL)"
        return "SQLite Demo (Local)"

    def initialize(self) -> None:
        """Create all tables in the target database and run additive migrations."""
        Base.metadata.create_all(self.engine)
        inspector = inspect(self.engine)
        if "broker_tenders" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("broker_tenders")}
            if "cargo_type" not in columns:
                with self.engine.begin() as connection:
                    connection.execute(text("ALTER TABLE broker_tenders ADD COLUMN cargo_type VARCHAR(80) NOT NULL DEFAULT 'Thermal coal'"))

    def session(self) -> Session:
        return self.session_factory()

    def get_table_counts(self) -> dict[str, int]:
        """Fetch row counts across all VoyageAI database tables."""
        with self.session() as session:
            def safe_count(cls):
                try:
                    return session.scalar(select(func.count(cls.id))) or 0
                except Exception:
                    return 0

            return {
                "market_observations": safe_count(MarketObservation),
                "ports": safe_count(PortRecord),
                "vessels": safe_count(VesselRecord),
                "model_runs": safe_count(ModelRun),
                "recommendations": safe_count(Recommendation),
                "broker_tenders": safe_count(BrokerTender),
                "tender_decisions": safe_count(TenderDecision),
                "alerts": safe_count(Alert),
            }

    def market_count(self) -> int:
        with self.session() as session:
            try:
                return session.scalar(select(func.count(MarketObservation.id))) or 0
            except Exception:
                return 0

    def seed_master_data(self, frame=None, provenance: str = "") -> int:
        """Seed market observations into Supabase / target database if empty."""
        from maritime_ai.data import load_freight_data

        if frame is None:
            frame, prov = load_freight_data()
            if not provenance:
                provenance = prov

        inserted = 0
        with self.session() as session:
            existing_dates = set(session.scalars(select(MarketObservation.observation_date)).all())
            for row in frame.itertuples(index=False):
                dt = row.date.to_pydatetime() if hasattr(row.date, "to_pydatetime") else row.date
                if dt in existing_dates:
                    continue
                session.add(
                    MarketObservation(
                        observation_date=dt,
                        freight_proxy_rate=float(row.rate_usd_per_tonne),
                        bunker_proxy=float(row.bunker_usd_per_tonne),
                        coal_price=float(row.coal_price_usd_per_tonne),
                        usd_inr=float(row.usd_inr),
                        congestion_days=float(row.congestion_days),
                        provenance=provenance or "VoyageAI master dataset",
                    )
                )
                inserted += 1
            session.commit()
        return inserted

    def seed_ports_and_vessels(self) -> tuple[int, int]:
        """Seed reference ports and vessel fleet specifications into database."""
        from maritime_ai.data import ports, vessels

        port_df = ports()
        vessel_df = vessels()
        ports_added = 0
        vessels_added = 0

        with self.session() as session:
            existing_ports = set(session.scalars(select(PortRecord.port)).all())
            for _, row in port_df.iterrows():
                if row["port"] in existing_ports:
                    continue
                session.add(
                    PortRecord(
                        port=row["port"],
                        max_draft_m=float(row["max_draft_m"]),
                        max_loa_m=float(row["max_loa_m"]),
                        max_beam_m=float(row["max_beam_m"]),
                        max_dwt_tonnes=float(row["max_dwt_tonnes"]),
                    )
                )
                ports_added += 1

            existing_vessels = set(session.scalars(select(VesselRecord.vessel)).all())
            for _, row in vessel_df.iterrows():
                if row["vessel"] in existing_vessels:
                    continue
                session.add(
                    VesselRecord(
                        vessel=row["vessel"],
                        vessel_class=row["class"],
                        capacity_tonnes=float(row["capacity_tonnes"]),
                        draft_m=float(row["draft_m"]),
                        loa_m=float(row["loa_m"]),
                        beam_m=float(row["beam_m"]),
                        fuel_factor=float(row["fuel_factor"]),
                    )
                )
                vessels_added += 1

            session.commit()
        return ports_added, vessels_added

    def seed_baseline_models(self) -> int:
        """Seed model evaluation benchmark runs into database."""
        from maritime_ai.data import load_freight_data
        from maritime_ai.forecast import backtest

        data, _ = load_freight_data()
        metrics = backtest(data)
        selected_model = metrics["selected_model"]

        added = 0
        with self.session() as session:
            for row in metrics["comparison"].to_dict("records"):
                session.add(
                    ModelRun(
                        model_name=row["Model"],
                        version=row["Version"],
                        mae=float(row["MAE"]),
                        rmse=float(row["RMSE"]),
                        training_rows=metrics["n_predictions"],
                        status="ACTIVE" if row["Model"] == selected_model else "TESTED",
                    )
                )
                added += 1
            session.commit()
        return added

    def seed_all_defaults(self) -> dict[str, int]:
        """Ensure all base tables (market data, ports, vessels, models) are seeded."""
        self.initialize()
        counts = self.get_table_counts()
        summary = {"market_observations": 0, "ports": 0, "vessels": 0, "model_runs": 0}

        if counts["market_observations"] == 0:
            summary["market_observations"] = self.seed_master_data()

        if counts["ports"] == 0 or counts["vessels"] == 0:
            p_add, v_add = self.seed_ports_and_vessels()
            summary["ports"] = p_add
            summary["vessels"] = v_add

        if counts["model_runs"] == 0:
            summary["model_runs"] = self.seed_baseline_models()

        return summary

    def save_recommendation(
        self,
        destination_port: str,
        cargo_tonnes: float,
        horizon_weeks: int,
        model_name: str,
        vessel: str,
        contract: str,
        all_in_usd_per_tonne: float,
        risk_score: int,
        risk_label: str,
        scenario: dict[str, Any] | str,
        origin_port: str = "Indonesia",
        saving_vs_spot_usd: float = 0.0,
        vessel_class: str = "",
        details: dict[str, Any] | str | None = None,
    ) -> Recommendation:
        """Persist an optimization recommendation to Supabase / target database."""
        payload = {}
        if isinstance(scenario, dict):
            payload.update(scenario)
        elif isinstance(scenario, str):
            try:
                payload = json.loads(scenario)
            except Exception:
                payload = {"raw_scenario": scenario}

        if origin_port:
            payload["origin_port"] = origin_port
        if saving_vs_spot_usd:
            payload["saving_vs_spot_usd"] = float(saving_vs_spot_usd)
        if vessel_class:
            payload["vessel_class"] = vessel_class
        if details:
            payload["details"] = details

        scenario_str = json.dumps(payload)

        with self.session() as session:
            rec = Recommendation(
                destination_port=destination_port,
                cargo_tonnes=float(cargo_tonnes),
                horizon_weeks=int(horizon_weeks),
                model_name=str(model_name),
                vessel=str(vessel),
                contract=str(contract),
                all_in_usd_per_tonne=float(all_in_usd_per_tonne),
                risk_score=int(risk_score),
                risk_label=str(risk_label),
                scenario_json=scenario_str,
            )
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec

    def save_alerts(self, alerts_list: list[tuple[str, str, str] | dict[str, str]]) -> int:
        """Persist system / risk / operational alerts to database."""
        added = 0
        with self.session() as session:
            for item in alerts_list:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    cat, sev, msg = item
                elif isinstance(item, dict):
                    cat = item.get("category", item.get("Category", "General"))
                    sev = item.get("severity", item.get("Severity", "Info"))
                    msg = item.get("message", item.get("Alert", item.get("msg", "")))
                else:
                    continue
                session.add(Alert(category=str(cat), severity=str(sev), message=str(msg)))
                added += 1
            session.commit()
        return added
