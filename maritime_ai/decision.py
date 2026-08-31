import pandas as pd


def feasibility(port: pd.Series, vessel: pd.Series, cargo_tonnes: float) -> dict:
    checks = {
        "Draft": vessel.draft_m <= port.max_draft_m,
        "LOA": vessel.loa_m <= port.max_loa_m,
        "Beam": vessel.beam_m <= port.max_beam_m,
        "Port DWT": vessel.capacity_tonnes <= port.max_dwt_tonnes,
        "Cargo capacity": vessel.capacity_tonnes >= cargo_tonnes,
    }
    return {"feasible": all(checks.values()), "checks": checks}


def risk_score(volatility: float, congestion_days: float, feasibility_result: dict) -> tuple[int, str]:
    score = min(100, round(volatility * 22 + congestion_days * 12 + (0 if feasibility_result["feasible"] else 50)))
    label = "Low" if score < 30 else "Medium" if score < 60 else "High"
    return score, label


def contract_options(forecast_rates: pd.DataFrame, vessel: pd.Series, cargo_tonnes: float, port_cost_usd: float = 85000) -> pd.DataFrame:
    """Cost model compares expected spot exposure against fixed multi-voyage offers.

    Term discounts model procurement leverage; users can replace them with received quotes.
    """
    expected_rate = float(forecast_rates.rate_usd_per_tonne.mean())
    bunker = float(forecast_rates.bunker_usd_per_tonne.mean())
    congestion = float(forecast_rates.congestion_days.mean())
    voyage_count = {"Spot (1 voyage)": 1, "3-voyage contract": 3, "6-voyage contract": 6}
    discount = {"Spot (1 voyage)": 0.0, "3-voyage contract": 0.035, "6-voyage contract": 0.065}
    rows = []
    for contract, voyages in voyage_count.items():
        contracted_rate = expected_rate * (1 - discount[contract])
        freight = contracted_rate * cargo_tonnes * voyages
        fuel = bunker * vessel.fuel_factor * 0.018 * cargo_tonnes * voyages
        delay = congestion * 18000 * voyages
        total = freight + fuel + port_cost_usd * voyages + delay
        rows.append([contract, voyages, contracted_rate, total, delay, total / (cargo_tonnes * voyages)])
    result = pd.DataFrame(rows, columns=["contract", "voyages", "rate_usd_per_tonne", "total_cost_usd", "delay_cost_usd", "all_in_usd_per_tonne"])
    spot_unit = float(result.iloc[0].all_in_usd_per_tonne)
    result["saving_vs_spot_usd"] = (spot_unit - result.all_in_usd_per_tonne) * cargo_tonnes * result.voyages
    return result.sort_values("all_in_usd_per_tonne").reset_index(drop=True)
