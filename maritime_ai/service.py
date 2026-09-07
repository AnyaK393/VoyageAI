"""Shared application service used by FastAPI without changing Streamlit."""

from __future__ import annotations

import json

import pandas as pd

from maritime_ai.data import load_freight_data, ports, vessels
from maritime_ai.decision import contract_options, feasibility, risk_score
from maritime_ai.forecast import backtest, forecast


def optimize_charter(
    destination_port: str,
    cargo_tonnes: float,
    horizon_weeks: int = 12,
    scenario: dict | None = None,
) -> dict:
    """Run the same forecast → feasibility → optimizer path as the dashboard."""
    scenario = scenario or {}
    data, provenance = load_freight_data()
    metric = backtest(data)
    forecast_frame = forecast(data, horizon_weeks, scenario, metric["selected_model"])
    forecast_frame["rate_usd_per_tonne"] *= 1 + scenario.get("freight_pct", 0) / 100

    port_table = ports().set_index("port")
    if destination_port not in port_table.index:
        raise ValueError(f"Unknown destination port: {destination_port}")
    port = port_table.loc[destination_port]
    candidates = []
    for _, vessel in vessels().iterrows():
        gate = feasibility(port, vessel, cargo_tonnes)
        if not gate["feasible"]:
            continue
        options = contract_options(forecast_frame, vessel, cargo_tonnes, scenario=scenario)
        options["vessel"] = vessel["vessel"]
        options["vessel_class"] = vessel["class"]
        candidates.append(options)
    if not candidates:
        raise ValueError("No vessel meets the selected port and cargo constraints.")
    options = pd.concat(candidates, ignore_index=True).sort_values("all_in_usd_per_tonne").reset_index(drop=True)
    winner = options.iloc[0]
    vessel = vessels().set_index("vessel").loc[winner["vessel"]]
    gate = feasibility(port, vessel, cargo_tonnes)
    volatility = float(data["rate_usd_per_tonne"].tail(12).std())
    score, label = risk_score(volatility, float(forecast_frame["congestion_days"].mean()), gate)
    return {
        "provenance": provenance,
        "model": metric["selected_model"],
        "model_version": metric["selected_version"],
        "backtest": {"mae": metric["mae"], "rmse": metric["rmse"], "points": metric["n_predictions"]},
        "forecast_average_usd_per_tonne": float(forecast_frame["rate_usd_per_tonne"].mean()),
        "forecast": json.loads(forecast_frame[["date", "rate_usd_per_tonne"]].to_json(orient="records", date_format="iso")),
        "recommendation": {
            "vessel": str(winner["vessel"]), "vessel_class": str(winner["vessel_class"]),
            "contract": str(winner["contract"]), "all_in_usd_per_tonne": float(winner["all_in_usd_per_tonne"]),
            "saving_vs_spot_usd": float(winner["saving_vs_spot_usd"]), "risk_score": score, "risk_label": label,
        },
        "eligible_options": json.loads(options[["vessel", "vessel_class", "contract", "voyages", "all_in_usd_per_tonne", "saving_vs_spot_usd"]].to_json(orient="records")),
    }
