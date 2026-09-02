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


def contract_options(forecast_rates: pd.DataFrame, vessel: pd.Series, cargo_tonnes: float, port_cost_usd: float = 85000,
                     scenario: dict | None = None) -> pd.DataFrame:
    """Cost model compares expected spot exposure against fixed multi-voyage offers.

    Term discounts model procurement leverage; users can replace them with received quotes.
    """
    scenario = scenario or {}
    expected_rate = float(forecast_rates.rate_usd_per_tonne.mean())
    bunker = float(forecast_rates.bunker_usd_per_tonne.mean())
    congestion = float(forecast_rates.congestion_days.mean())
    voyage_count = {"Spot (1 voyage)": 1, "3-voyage contract": 3, "6-voyage contract": 6}
    discount = {"Spot (1 voyage)": 0.0, "3-voyage contract": 0.035, "6-voyage contract": 0.065}
    # Longer commitments receive a transparent contingency for fuel, congestion
    # and market uncertainty. This is what lets the optimizer make a genuine
    # cost-versus-commitment decision under a stress scenario.
    stress = (abs(scenario.get("bunker_pct", 0)) / 100 * 0.42
              + max(0, scenario.get("congestion_days", 0)) * 0.035
              + abs(scenario.get("freight_pct", 0)) / 100 * 0.18)
    exposure = {"Spot (1 voyage)": 0.0, "3-voyage contract": 0.85, "6-voyage contract": 2.35}
    rows = []
    for contract, voyages in voyage_count.items():
        contracted_rate = expected_rate * (1 - discount[contract])
        freight = contracted_rate * cargo_tonnes * voyages
        fuel = bunker * vessel.fuel_factor * 0.018 * cargo_tonnes * voyages
        delay = congestion * 18000 * voyages
        risk_allowance = (freight + fuel + delay) * stress * exposure[contract]
        total = freight + fuel + port_cost_usd * voyages + delay + risk_allowance
        rows.append([contract, voyages, contracted_rate, total, delay, risk_allowance, total / (cargo_tonnes * voyages)])
    result = pd.DataFrame(rows, columns=["contract", "voyages", "rate_usd_per_tonne", "total_cost_usd", "delay_cost_usd", "risk_allowance_usd", "all_in_usd_per_tonne"])
    spot_unit = float(result.iloc[0].all_in_usd_per_tonne)
    result["saving_vs_spot_usd"] = (spot_unit - result.all_in_usd_per_tonne) * cargo_tonnes * result.voyages
    return result.sort_values("all_in_usd_per_tonne").reset_index(drop=True)

def find_decision_flip(
    run_scenario,
    base_scenario: dict,
    variable: str,
    max_stress: float,
    step: float,
):
    """
    Find the first additional adverse stress level that changes
    the optimizer's decision.

    The supplied run_scenario function must return the winning
    option as a pandas Series containing 'contract' and 'vessel'.

    Returns:
        (threshold, new_winner)
        threshold is the approximate additional stress required.
        Returns (None, None) if no flip occurs within the tested range.
    """
    base_result = run_scenario(base_scenario)

    base_signature = (
        base_result["contract"],
        base_result["vessel"],
    )

    previous = 0.0
    stress = float(step)

    # First find a bracket where the decision changes.
    while stress <= max_stress + 1e-9:
        trial = dict(base_scenario)
        trial[variable] = trial.get(variable, 0) + stress

        result = run_scenario(trial)

        signature = (
            result["contract"],
            result["vessel"],
        )

        if signature != base_signature:
            # We know the flip is between previous and stress.
            low = previous
            high = stress

            # Refine the threshold.
            for _ in range(10):
                midpoint = (low + high) / 2

                trial = dict(base_scenario)
                trial[variable] = trial.get(variable, 0) + midpoint

                midpoint_result = run_scenario(trial)

                midpoint_signature = (
                    midpoint_result["contract"],
                    midpoint_result["vessel"],
                )

                if midpoint_signature == base_signature:
                    low = midpoint
                else:
                    high = midpoint

            # Final evaluation at the estimated threshold.
            final_scenario = dict(base_scenario)
            final_scenario[variable] = (
                final_scenario.get(variable, 0) + high
            )

            final_result = run_scenario(final_scenario)

            return high, final_result

        previous = stress
        stress += step

    return None, None