"""Tender validation and ranking shared by the dashboard and API."""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import pandas as pd

from maritime_ai.data import feasible_berths, vessels


CONTRACT_VOYAGES = {
    "Spot (1 voyage)": 1,
    "3-voyage contract": 3,
    "6-voyage contract": 6,
}


def as_utc(value: date | datetime) -> datetime:
    """Store date-only tender controls as an unambiguous UTC timestamp."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, time.max, tzinfo=timezone.utc)


def rank_tenders(tenders: list, destination_port: str, cargo_tonnes: float, cargo_type: str = "Thermal coal", now: datetime | None = None) -> pd.DataFrame:
    """Rank current tenders only when they are valid and vessel-port feasible.

    A tender's supplied rate is intentionally treated as all-in. VoyageAI does
    not invent discounts or add proxy costs to a broker's commercial offer.
    """
    now = now or datetime.now(timezone.utc)
    vessel_table = vessels().set_index("vessel")
    rows = []
    for tender in tenders:
        reasons = []
        vessel = vessel_table.loc[tender.vessel] if tender.vessel in vessel_table.index else None
        if tender.status != "SUBMITTED":
            reasons.append(f"Status: {tender.status.title()}")
        if as_utc(tender.valid_until) < now:
            reasons.append("Quote expired")
        if tender.destination_port != destination_port:
            reasons.append("Different discharge port")
        if abs(float(tender.cargo_tonnes) - float(cargo_tonnes)) > 0.01:
            reasons.append("Different cargo quantity")
        if tender.cargo_type != cargo_type:
            reasons.append("Different cargo type")
        if vessel is None:
            reasons.append("Unknown vessel")
        else:
            berth_matches = feasible_berths(destination_port, cargo_type, vessel, cargo_tonnes)
            passing_berths = berth_matches[berth_matches["feasible"]] if not berth_matches.empty else berth_matches
            if passing_berths.empty:
                reasons.append("No compatible cargo berth")
        eligible = not reasons
        rows.append({
            "id": tender.id, "broker": tender.broker_name, "vessel": tender.vessel,
            "imo": vessel["imo"] if vessel is not None else "—", "cargo_type": tender.cargo_type,
            "contract": tender.contract, "voyages": tender.voyages,
            "all_in_usd_per_tonne": tender.all_in_usd_per_tonne,
            "valid_until": tender.valid_until, "notes": tender.notes,
            "eligible": eligible, "decision": "Eligible" if eligible else "Not comparable",
            "berth": passing_berths.iloc[0]["berth"] if vessel is not None and not passing_berths.empty else "—",
            "reason": "All validations passed" if eligible else "; ".join(reasons),
        })
    return pd.DataFrame(rows).sort_values(["eligible", "all_in_usd_per_tonne"], ascending=[False, True]).reset_index(drop=True) if rows else pd.DataFrame()
