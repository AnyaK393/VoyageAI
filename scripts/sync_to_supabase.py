#!/usr/bin/env python3
"""Sync all VoyageAI data (master dataset, ports, fleet, model benchmarks, recommendations, tenders, alerts) to Supabase."""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from maritime_ai.data import load_freight_data, ports, vessels
from maritime_ai.database import Database, Recommendation, Alert, ModelRun, PortRecord, VesselRecord, MarketObservation, BrokerTender
from maritime_ai.service import optimize_charter
from maritime_ai.forecast import backtest
from maritime_ai.tenders import as_utc


def main():
    print("=" * 70, flush=True)
    print(" ⚓ VoyageAI • Supabase & Database Sync Utility", flush=True)
    print("=" * 70, flush=True)

    db = Database()
    print(f"\n📡 Target Database: {db.database_type_display}", flush=True)
    print(f"🔗 Normalized URL: {db.url.split('@')[-1] if '@' in db.url else db.url}", flush=True)

    if not db.is_supabase_or_postgres:
        print("\n⚠️  NOTE: DATABASE_URL is not currently pointing to Supabase/PostgreSQL.", flush=True)
        print("   To sync directly to your Supabase project, paste your connection string into `.env`:", flush=True)
        print("   DATABASE_URL=\"postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres\"\n", flush=True)
    else:
        print("\n✅ Connected to Supabase / PostgreSQL!\n", flush=True)

    print("Step 1: Initializing Database Schema (Tables)...", flush=True)
    db.initialize()
    print("   ✓ Tables ready: market_observations, ports, vessels, model_runs, recommendations, broker_tenders, alerts", flush=True)

    print("\nStep 2: Syncing Master Market Observations Dataset...", flush=True)
    data, provenance = load_freight_data()
    obs_count = db.seed_master_data(data, provenance)
    print(f"   ✓ Master dataset: {obs_count} new records added (Total in DB: {db.market_count()})", flush=True)

    print("\nStep 3: Syncing Reference Ports & Fleet Vessels...", flush=True)
    ports_added, vessels_added = db.seed_ports_and_vessels()
    print(f"   ✓ Ports added: {ports_added} | Fleet vessels added: {vessels_added}", flush=True)

    print("\nStep 4: Syncing AI Forecast Model Benchmark Runs...", flush=True)
    metrics = backtest(data)
    selected_model = metrics["selected_model"]
    models_added = 0
    with db.session() as session:
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
            models_added += 1
        session.commit()
    print(f"   ✓ Model evaluations saved: {models_added} models (Selected: {selected_model})", flush=True)

    print("\nStep 5: Generating and Uploading Optimization Decisions & Scenarios...", flush=True)
    test_ports = ["Paradip", "Vizag", "Gangavaram", "Gopalpur", "Dhamra", "Haldia"]
    test_scenarios = [
        {"name": "Baseline (Current Market)", "scenario": {}},
        {"name": "Bunker Surge (+15%)", "scenario": {"bunker_pct": 15}},
        {"name": "Freight Spike (+20%)", "scenario": {"freight_pct": 20}},
        {"name": "Monsoon Congestion (+3 days)", "scenario": {"congestion_days": 3}},
    ]

    recs_created = 0
    for p in test_ports:
        for sc in test_scenarios:
            try:
                cargo_val = 50000 if p == "Haldia" else 58000
                res = optimize_charter(
                    destination_port=p,
                    cargo_tonnes=cargo_val,
                    horizon_weeks=12,
                    scenario=sc["scenario"],
                    data=data,
                    metric=metrics,
                )
                winner = res["recommendation"]
                db.save_recommendation(
                    destination_port=p,
                    cargo_tonnes=cargo_val,
                    horizon_weeks=12,
                    model_name=res["model"],
                    vessel=winner["vessel"],
                    vessel_class=winner.get("vessel_class", ""),
                    contract=winner["contract"],
                    all_in_usd_per_tonne=winner["all_in_usd_per_tonne"],
                    saving_vs_spot_usd=winner.get("saving_vs_spot_usd", 0.0),
                    risk_score=winner["risk_score"],
                    risk_label=winner["risk_label"],
                    scenario={"scenario_name": sc["name"], **sc["scenario"]},
                    origin_port="Indonesia",
                    details={"eligible_options": res.get("eligible_options", [])},
                )
                recs_created += 1
            except Exception as e:
                print(f"   ✕ Skipped {p} ({sc['name']}): {e}", flush=True)

    print(f"   ✓ Generated and stored {recs_created} optimization decisions across destination ports & scenarios", flush=True)

    print("\nStep 6: Syncing Broker Tenders...", flush=True)
    today = date.today()
    tenders_data = [
        BrokerTender(
            broker_name="Meridian Shipbrokers", origin="Indonesia", destination_port="Paradip",
            cargo_tonnes=58000, cargo_type="Thermal coal", laycan_start=as_utc(today + timedelta(days=14)),
            vessel="Supramax-58", contract="Spot (1 voyage)", voyages=1, all_in_usd_per_tonne=23.75,
            valid_until=as_utc(today + timedelta(days=5)), notes="Firm prompt offer ex-Samarinda", status="SUBMITTED"
        ),
        BrokerTender(
            broker_name="Apex Chartering", origin="Indonesia", destination_port="Paradip",
            cargo_tonnes=58000, cargo_type="Thermal coal", laycan_start=as_utc(today + timedelta(days=14)),
            vessel="Ultramax-63", contract="3-voyage contract", voyages=3, all_in_usd_per_tonne=22.90,
            valid_until=as_utc(today + timedelta(days=4)), notes="Consecutive voyages with 25 days laytime", status="SUBMITTED"
        ),
        BrokerTender(
            broker_name="Oceanic Logistics", origin="Indonesia", destination_port="Vizag",
            cargo_tonnes=82000, cargo_type="Thermal coal", laycan_start=as_utc(today + timedelta(days=21)),
            vessel="Panamax-82", contract="6-voyage contract", voyages=6, all_in_usd_per_tonne=21.40,
            valid_until=as_utc(today + timedelta(days=7)), notes="Subject to stem and shipper approval", status="SUBMITTED"
        ),
    ]
    tenders_added = 0
    with db.session() as session:
        for t in tenders_data:
            session.add(t)
            tenders_added += 1
        session.commit()
    print(f"   ✓ Broker tenders submitted and stored: {tenders_added}", flush=True)

    print("\nStep 7: Syncing Active Operational & Risk Alerts...", flush=True)
    alerts = [
        ("Data Quality", "Medium", "Freight proxy rate is active; recommended to replace with daily Baltic/Platts assessment."),
        ("Operations", "High", "Live berth congestion feed is simulated; check port marine department notices before fixing."),
        ("Risk Governance", "Medium", "Monsoon laycan risk factor applies to eastern Indian discharge ports June–September."),
        ("Decision Rule", "Low", "Capesize-180 draft exceeds Haldia navigational depth constraint (max draft 8.5m)."),
    ]
    alerts_added = db.save_alerts(alerts)
    print(f"   ✓ Operational & risk alerts stored: {alerts_added}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(" 🎉 SYNC SUMMARY ACROSS SUPABASE DATABASE TABLES", flush=True)
    print("=" * 70, flush=True)
    counts = db.get_table_counts()
    print(f" │ 📊 Master Market Observations (`market_observations`): {counts['market_observations']:>6} rows", flush=True)
    print(f" │ 🚢 Destination Ports Master    (`ports`)              : {counts['ports']:>6} rows", flush=True)
    print(f" │ ⚓ Vessel Fleet Master        (`vessels`)            : {counts['vessels']:>6} rows", flush=True)
    print(f" │ 🤖 AI Model Registry Runs     (`model_runs`)         : {counts['model_runs']:>6} rows", flush=True)
    print(f" │ 📋 Recommendations & Decisions(`recommendations`)    : {counts['recommendations']:>6} rows", flush=True)
    print(f" │ 💼 Broker Tenders             (`broker_tenders`)     : {counts['broker_tenders']:>6} rows", flush=True)
    print(f" │ 🚨 Operational & Risk Alerts  (`alerts`)             : {counts['alerts']:>6} rows", flush=True)
    print("=" * 70, flush=True)
    print("✅ All website data has been printed and persisted to Supabase!", flush=True)


if __name__ == "__main__":
    main()
