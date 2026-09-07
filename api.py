"""VoyageAI FastAPI service. Open /docs for the judge-friendly API console."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from maritime_ai.data import load_freight_data, ports, vessels
from maritime_ai.database import Database, MarketObservation, ModelRun, Recommendation
from maritime_ai.forecast import backtest
from maritime_ai.service import optimize_charter


class Scenario(BaseModel):
    bunker_pct: float = 0
    freight_pct: float = 0
    coal_pct: float = 0
    fx_pct: float = 0
    congestion_days: float = Field(default=0, ge=0)


class OptimizeRequest(BaseModel):
    destination_port: str = "Paradip"
    cargo_tonnes: int = Field(default=58000, ge=20000, le=180000)
    horizon_weeks: int = Field(default=12, ge=4, le=12)
    scenario: Scenario = Field(default_factory=Scenario)


def create_app(database_url: str | None = None) -> FastAPI:
    database = Database(database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database.initialize()
        frame, _ = load_freight_data()
        with database.session() as session:
            if database.market_count() == 0:
                for row in frame.itertuples(index=False):
                    session.add(MarketObservation(
                        observation_date=row.date.to_pydatetime(), freight_proxy_rate=float(row.rate_usd_per_tonne),
                        bunker_proxy=float(row.bunker_usd_per_tonne), coal_price=float(row.coal_price_usd_per_tonne),
                        usd_inr=float(row.usd_inr), congestion_days=float(row.congestion_days),
                        provenance="Supplied VoyageAI master dataset / prototype defaults",
                    ))
                session.commit()
        yield

    app = FastAPI(title="VoyageAI API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "database": "postgresql" if database.url.startswith("postgresql") else "sqlite-demo", "market_observations": database.market_count()}

    @app.get("/market/latest")
    def market_latest():
        with database.session() as session:
            item = session.scalar(select(MarketObservation).order_by(desc(MarketObservation.observation_date)))
            if item is None:
                raise HTTPException(503, "Market data has not been initialized.")
            return {"date": item.observation_date, "freight_proxy_rate": item.freight_proxy_rate, "bunker_proxy": item.bunker_proxy, "coal_price": item.coal_price, "usd_inr": item.usd_inr, "congestion_days": item.congestion_days, "provenance": item.provenance}

    @app.get("/data-health")
    def data_health():
        frame, provenance = load_freight_data()
        latest = frame["date"].max()
        return {"as_of": latest, "overall": "warning", "feeds": [
            {"name": "Freight", "status": "proxy", "latest": latest, "detail": "Market proxy, not route assessment"},
            {"name": "Coal", "status": "available", "latest": latest, "detail": "World Bank monthly input"},
            {"name": "USD/INR", "status": "available", "latest": latest, "detail": "Weekly close from raw feed"},
            {"name": "Congestion", "status": "missing", "latest": None, "detail": "Prototype default; AIS/port feed required"},
        ], "provenance": provenance}

    @app.get("/ports")
    def get_ports(): return ports().to_dict(orient="records")

    @app.get("/vessels")
    def get_vessels(): return vessels().to_dict(orient="records")

    @app.get("/model-status")
    def model_status():
        with database.session() as session:
            return [{"model": row.model_name, "version": row.version, "mae": row.mae, "rmse": row.rmse, "training_rows": row.training_rows, "status": row.status, "evaluated_at": row.evaluated_at} for row in session.scalars(select(ModelRun).order_by(desc(ModelRun.evaluated_at))).all()]

    @app.post("/models/evaluate")
    def evaluate_models():
        frame, _ = load_freight_data(); report = backtest(frame)
        with database.session() as session:
            for item in (report["ridge"], report["xgboost"]):
                session.add(ModelRun(model_name=item["model"], version=item["version"], mae=item["mae"], rmse=item["rmse"], training_rows=len(frame), status="ACTIVE" if item["model"] == report["selected_model"] else "TESTED"))
            session.commit()
        return {"selected_model": report["selected_model"], "comparison": report["comparison"].to_dict(orient="records")}

    @app.post("/optimize")
    def optimize(request: OptimizeRequest):
        try:
            result = optimize_charter(request.destination_port, request.cargo_tonnes, request.horizon_weeks, request.scenario.model_dump())
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        winner = result["recommendation"]
        with database.session() as session:
            session.add(Recommendation(destination_port=request.destination_port, cargo_tonnes=request.cargo_tonnes, horizon_weeks=request.horizon_weeks, model_name=result["model"], vessel=winner["vessel"], contract=winner["contract"], all_in_usd_per_tonne=winner["all_in_usd_per_tonne"], risk_score=winner["risk_score"], risk_label=winner["risk_label"], scenario_json=request.scenario.model_dump_json()))
            session.commit()
        return result

    @app.get("/decision-history")
    def decision_history():
        with database.session() as session:
            return [{"created_at": row.created_at, "destination_port": row.destination_port, "cargo_tonnes": row.cargo_tonnes, "model": row.model_name, "vessel": row.vessel, "contract": row.contract, "all_in_usd_per_tonne": row.all_in_usd_per_tonne, "risk_score": row.risk_score, "risk_label": row.risk_label, "scenario": json.loads(row.scenario_json)} for row in session.scalars(select(Recommendation).order_by(desc(Recommendation.created_at))).all()]

    return app


app = create_app()
