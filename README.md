# Maritime AI — Charter Optimizer Prototype

A compact, runnable prototype for PS 26006. It chains together:

1. reproducible proxy freight data and walk-forward forecasting;
2. vessel–port feasibility checks;
3. spot vs. 3-voyage vs. 6-voyage cost comparison;
4. an explainable operational risk score; and
5. a Streamlit dashboard with a connected What-If Simulator and Decision Threshold Engine.

## Run

```bash
python3 -m unittest discover -s tests -v
streamlit run app.py
```

No external feed is required for the prototype. The supplied raw datasets can be joined into a reproducible weekly master file with `venv/bin/python scripts/build_master_dataset.py`; the app automatically uses it when `data/freight_rates.csv` is absent. Its freight and bunker values remain explicitly labelled market proxies, not route-specific freight or VLSFO quotes. See [the data catalogue](data/DATA_CATALOG.md) for lineage, gaps and recommended operational feeds. To use licensed or public route data, put a CSV in `data/freight_rates.csv` with `date` and `rate_usd_per_tonne` columns. Optional fields are `route`, `bunker_usd_per_tonne`, `coal_price_usd_per_tonne`, `usd_inr`, `congestion_days`, `route_distance_nm`, and `vessel_class_code`. The app supplies clearly labelled prototype defaults when optional fields are absent.

## Demo flow

Open **What-if scenario** in the sidebar and change bunker price, freight, coal, FX or congestion. The app re-runs the driver-based forecast, vessel × contract cost comparison, risk allowance and recommendation. The **Decision stability** section searches for the next bunker, freight and congestion level that flips the recommended contract.

The dashboard also includes a **System operations** tab. Use it to show data health, the active Ridge/XGBoost model registry, data-gap and decision-flip alerts, and the persistent recommendation audit trail. Click **Save current recommendation and alerts** to record the current run locally.

The **Tender desk** tab is a separate broker-quote workspace that does not alter the existing forecast dashboard. Add an all-in quote for Spot, 3-voyage or 6-voyage terms; VoyageAI ranks only offers that are still valid and feasible for the selected port and cargo. It is intentionally open in the prototype. Broker/admin permissions belong in the Supabase Auth deployment phase.

## Backend API and database

The existing Streamlit dashboard is unchanged. A separate FastAPI backend now exposes the same model and optimizer plus data health, model registry and saved decision history.

```bash
venv/bin/uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive judge demo. The default is a local SQLite database at `data/voyageai.db`, seeded from the master dataset on first start. For PostgreSQL or Supabase, set `DATABASE_URL` before starting, for example:

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/postgres'
venv/bin/uvicorn api:app --reload
```

Core endpoints: `/health`, `/market/latest`, `/data-health`, `/ports`, `/vessels`, `/model-status`, `POST /models/evaluate`, `POST /optimize`, `/decision-history`, `POST /tenders`, and `/tenders`.

## Important data note

The included rates are **synthetic proxy data** calibrated only to make the decision pipeline testable. They are not Baltic Exchange or SAIL historical rates. Production use should replace them with licensed freight/AIS feeds and validated port notices.
