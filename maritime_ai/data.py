from pathlib import Path
import numpy as np
import pandas as pd
from maritime_ai.decision import feasibility

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CARGO_TYPES = ("Thermal coal", "Coking coal", "Iron ore")


def load_freight_data(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load supplied master data, an explicitly supplied CSV, or demo fallback.

    The generated master dataset remains labelled as a freight-market proxy. It
    is intentionally preferred over synthetic history without changing the
    dashboard's input/output contract.
    """
    source = path or DATA_DIR / "freight_rates.csv"
    if path is None and not source.exists():
        master = DATA_DIR / "processed" / "voyageai_master_weekly.csv"
        if master.exists():
            source = master
    if source.exists():
        frame = pd.read_csv(source, parse_dates=["date"]).sort_values("date")
        required = {"date", "rate_usd_per_tonne"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Freight CSV missing columns: {sorted(missing)}")
        # These defaults keep a minimal historic-rate CSV usable, while exposing
        # the market and operational inputs used by the prototype model.
        for name, default in {
            "bunker_usd_per_tonne": 620.0,
            "coal_price_usd_per_tonne": 105.0,
            "usd_inr": 83.0,
            "congestion_days": 1.8,
            "route_distance_nm": 2400.0,
            "vessel_class_code": 1.0,
        }.items():
            if name not in frame:
                frame[name] = default
        if source.name == "voyageai_master_weekly.csv":
            return frame.reset_index(drop=True), (
                "Supplied master dataset: Statistics Finland freight index "
                "proxy + World Bank coal/Brent proxies + USD/INR"
            )
        return frame.reset_index(drop=True), f"CSV: {source.name}"

    index = pd.date_range("2023-01-02", periods=156, freq="W-MON")
    rng = np.random.default_rng(26006)
    trend = np.linspace(0, 4.2, len(index))
    seasonal = 2.4 * np.sin(np.arange(len(index)) * 2 * np.pi / 26)
    shocks = rng.normal(0, 1.05, len(index)).cumsum() * 0.18 + rng.normal(0, 0.55, len(index))
    rate = 15.5 + trend + seasonal + shocks
    bunker = 590 + 24 * np.sin(np.arange(len(index)) * 2 * np.pi / 40) + rng.normal(0, 8, len(index))
    congestion = np.maximum(0.3, 1.4 + 0.45 * np.sin(np.arange(len(index)) * 2 * np.pi / 18) + rng.normal(0, .18, len(index)))
    coal = 102 + 6 * np.sin(np.arange(len(index)) * 2 * np.pi / 34) + rng.normal(0, 1.6, len(index))
    fx = 82.5 + np.linspace(0, 1.0, len(index)) + rng.normal(0, .18, len(index))
    month = index.month
    monsoon = ((month >= 6) & (month <= 9)).astype(int)
    return pd.DataFrame({"date": index, "route": "Indonesia–Paradip", "rate_usd_per_tonne": rate.round(2),
                         "bunker_usd_per_tonne": bunker.round(1), "coal_price_usd_per_tonne": coal.round(2),
                         "usd_inr": fx.round(2), "congestion_days": congestion.round(2),
                         "route_distance_nm": 2400.0, "vessel_class_code": 1.0,
                         "monsoon_indicator": monsoon}), "Deterministic proxy data (demo only)"


def ports() -> pd.DataFrame:
    return pd.DataFrame([
        ["Paradip", 17.1, 270, 45, 180000], ["Vizag", 16.5, 260, 43, 160000],
        ["Gangavaram", 21.0, 300, 50, 200000], ["Gopalpur", 14.5, 230, 38, 90000],
        ["Dhamra", 18.5, 290, 47, 180000], ["Haldia", 8.5, 225, 35, 60000],
    ], columns=["port", "max_draft_m", "max_loa_m", "max_beam_m", "max_dwt_tonnes"])


def berths() -> pd.DataFrame:
    """Reference berth constraints used for cargo-aware prototype gating.

    Paradip coal-berth geometry is transcribed from the Port Authority's berth
    specifications. DWT and non-Paradip entries remain explicit prototype
    references until dated terminal notices are connected.
    """
    return pd.DataFrame([
        ["PPT-NCB-03", "Paradip", "New Coal Import Berth 03", "Coking coal", 16.0, 300, 46, 180000, "Paradip Port Authority berth specification; DWT prototype reference", "Observed/reference"],
        ["PPT-CB-05", "Paradip", "Coal Berth 01", "Thermal coal", 14.5, 300, 46, 180000, "Paradip Port Authority berth specification; DWT prototype reference", "Observed/reference"],
        ["PPT-CB-06", "Paradip", "Coal Berth 02", "Thermal coal", 14.5, 300, 46, 180000, "Paradip Port Authority berth specification; DWT prototype reference", "Observed/reference"],
        ["PPT-IOB-02", "Paradip", "New Iron Ore Berth 02", "Iron ore", 16.0, 300, 46, 180000, "Paradip Port Authority berth specification; DWT prototype reference", "Observed/reference"],
        ["VIZ-REF-01", "Vizag", "Port reference berth", "Thermal coal|Coking coal|Iron ore", 16.5, 260, 43, 160000, "Prototype reference; replace with terminal notice", "Prototype"],
        ["GAN-REF-01", "Gangavaram", "Port reference berth", "Thermal coal|Coking coal|Iron ore", 21.0, 300, 50, 200000, "Prototype reference; replace with terminal notice", "Prototype"],
        ["GOP-REF-01", "Gopalpur", "Port reference berth", "Thermal coal|Coking coal|Iron ore", 14.5, 230, 38, 90000, "Prototype reference; replace with terminal notice", "Prototype"],
        ["DHA-REF-01", "Dhamra", "Port reference berth", "Thermal coal|Coking coal|Iron ore", 18.5, 290, 47, 180000, "Prototype reference; replace with terminal notice", "Prototype"],
        ["HAL-REF-01", "Haldia", "Port reference berth", "Thermal coal|Coking coal|Iron ore", 8.5, 225, 35, 60000, "Prototype reference; replace with terminal notice", "Prototype"],
    ], columns=["berth_id", "port", "berth", "cargo_types", "max_draft_m", "max_loa_m", "max_beam_m", "max_dwt_tonnes", "source", "data_status"])


def feasible_berths(port_name: str, cargo_type: str, vessel: pd.Series, cargo_tonnes: float) -> pd.DataFrame:
    """Return every compatible berth plus transparent pass/fail reasons."""
    berth_table = berths()
    candidates = berth_table[(berth_table["port"] == port_name) & berth_table["cargo_types"].str.split("|").apply(lambda values: cargo_type in values)]
    rows = []
    for _, berth in candidates.iterrows():
        gate = feasibility(berth, vessel, cargo_tonnes)
        failed = [name for name, passed in gate["checks"].items() if not passed]
        rows.append({**berth.to_dict(), "feasible": gate["feasible"], "constraint": ", ".join(failed) if failed else "All checks passed"})
    return pd.DataFrame(rows)


def vessels() -> pd.DataFrame:
    return pd.DataFrame([
        ["Supramax-58", "DEMO-IMO-9001001", "Supramax", 58000, 12.8, 199, 32.3, 0.72],
        ["Ultramax-63", "DEMO-IMO-9001002", "Ultramax", 63000, 13.2, 200, 32.4, 0.70],
        ["Panamax-82", "DEMO-IMO-9001003", "Panamax", 82000, 14.2, 229, 32.3, 0.61],
        ["Kamsarmax-84", "DEMO-IMO-9001004", "Kamsarmax", 84000, 14.5, 229, 32.3, 0.60],
        ["Capesize-180", "DEMO-IMO-9001005", "Capesize", 180000, 18.3, 289, 45.0, 0.46],
    ], columns=["vessel", "imo", "class", "capacity_tonnes", "draft_m", "loa_m", "beam_m", "fuel_factor"])
