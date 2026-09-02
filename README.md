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

No external feed is required for the prototype: the model generates a deterministic, clearly labelled proxy route-rate history. To use licensed or public route data, put a CSV in `data/freight_rates.csv` with `date` and `rate_usd_per_tonne` columns. Optional fields are `route`, `bunker_usd_per_tonne`, `coal_price_usd_per_tonne`, `usd_inr`, `congestion_days`, `route_distance_nm`, and `vessel_class_code`. The app supplies clearly labelled prototype defaults when optional fields are absent.

## Demo flow

Open **What-if scenario** in the sidebar and change bunker price, freight, coal, FX or congestion. The app re-runs the driver-based forecast, vessel × contract cost comparison, risk allowance and recommendation. The **Decision stability** section searches for the next bunker, freight and congestion level that flips the recommended contract.

## Important data note

The included rates are **synthetic proxy data** calibrated only to make the decision pipeline testable. They are not Baltic Exchange or SAIL historical rates. Production use should replace them with licensed freight/AIS feeds and validated port notices.
