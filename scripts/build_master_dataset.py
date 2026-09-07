"""Build VoyageAI's weekly modelling dataset from the checked-in raw sources.

This intentionally uses only the Python standard library plus pandas so it can
read the World Bank workbook even when ``openpyxl`` is not installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "voyageai_master_weekly.csv"
NAMESPACE = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_name(cell_ref: str) -> str:
    """Return the Excel column letters from a cell reference (for example A12)."""
    return "".join(character for character in cell_ref if character.isalpha())


def _workbook_strings(book: ZipFile) -> list[str]:
    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.findall(".//x:t", NAMESPACE))
            for item in root.findall("x:si", NAMESPACE)]


def _read_world_bank_monthly_prices(path: Path) -> pd.DataFrame:
    """Extract Brent and Australian coal monthly prices from the Pink Sheet XLSX."""
    with ZipFile(path) as book:
        strings = _workbook_strings(book)
        root = ET.fromstring(book.read("xl/worksheets/sheet3.xml"))

    rows: list[dict[str, str]] = []
    for row in root.findall(".//x:sheetData/x:row", NAMESPACE):
        values: dict[str, str] = {}
        for cell in row.findall("x:c", NAMESPACE):
            value = cell.find("x:v", NAMESPACE)
            text = "" if value is None else (value.text or "")
            if cell.get("t") == "s" and text:
                text = strings[int(text)]
            values[_column_name(cell.get("r", ""))] = text
        rows.append(values)

    headers = next(row for row in rows if row.get("A") == "" and row.get("C") == "Crude oil, Brent")
    reverse_headers = {name: column for column, name in headers.items()}
    required = {"Crude oil, Brent", "Coal, Australian"}
    missing = required - set(reverse_headers)
    if missing:
        raise ValueError(f"World Bank workbook is missing expected series: {sorted(missing)}")

    result = []
    for row in rows:
        period = row.get("A", "")
        if len(period) != 7 or period[4] != "M":
            continue
        result.append({
            "date": pd.to_datetime(period.replace("M", "-"), format="%Y-%m"),
            "brent_usd_per_bbl": pd.to_numeric(row.get(reverse_headers["Crude oil, Brent"]), errors="coerce"),
            "coal_australian_usd_per_tonne": pd.to_numeric(row.get(reverse_headers["Coal, Australian"]), errors="coerce"),
        })
    return pd.DataFrame(result).dropna().sort_values("date")


def build_master_dataset() -> pd.DataFrame:
    # Keep the YYYY.MM period as text: automatic float conversion would make
    # 2010.01 and 2010.10 both look like ``2010.1``.
    freight = pd.read_csv(
        RAW_DIR / "0_MB_HINNAT_Merirahtien_hinnat_kuukausidata_aikasarja_EN.csv",
        dtype={"Kuukausi": "string"},
    )
    freight = freight.rename(columns={"Kuukausi": "period", "Sea freight prices": "freight_proxy_index"})
    freight["date"] = pd.to_datetime(freight["period"].astype(str), format="%Y.%m")
    freight = freight[["date", "freight_proxy_index"]].sort_values("date")

    fx = pd.read_csv(RAW_DIR / "usdinr_d.csv", parse_dates=["Date"])
    fx = fx.rename(columns={"Date": "date", "Close": "usd_inr"})[["date", "usd_inr"]].sort_values("date")
    commodities = _read_world_bank_monthly_prices(RAW_DIR / "CMO-Historical-Data-Monthly.xlsx")

    # Use the common, observed period only; no future values are back-filled.
    latest = min(freight["date"].max(), fx["date"].max(), commodities["date"].max())
    earliest = max(freight["date"].min(), fx["date"].min(), commodities["date"].min())
    weekly_index = pd.date_range(earliest, latest, freq="W-FRI")

    monthly = freight.merge(commodities, on="date", how="inner").set_index("date").sort_index()
    weekly_market = monthly.reindex(monthly.index.union(weekly_index)).interpolate(method="time").reindex(weekly_index)
    weekly_fx = fx.set_index("date")["usd_inr"].resample("W-FRI").last().reindex(weekly_index).ffill()

    result = weekly_market.reset_index(names="date")
    result["usd_inr"] = weekly_fx.values
    # A scale-equivalent target preserves the existing prototype's $/t display,
    # but is explicitly documented as a freight-market proxy, not a route quote.
    result["rate_usd_per_tonne"] = result["freight_proxy_index"] / 100.0
    # Brent is an energy proxy; it is not a VLSFO bunker assessment.
    result["bunker_usd_per_tonne"] = result["brent_usd_per_bbl"] * 7.35
    result["coal_price_usd_per_tonne"] = result["coal_australian_usd_per_tonne"]
    result["congestion_days"] = 1.8
    result["route_distance_nm"] = 2400.0
    result["vessel_class_code"] = 1.0
    result["route"] = "Freight-market proxy → East Coast India planning"
    result["freight_source"] = "Statistics Finland sea-freight price index (proxy)"
    result["coal_source"] = "World Bank Pink Sheet: Coal, Australian"
    result["bunker_source"] = "World Bank Pink Sheet: Brent × 7.35 energy proxy"
    result["fx_source"] = "Raw USD/INR daily close, weekly last observation"
    result["congestion_source"] = "Prototype default; replace with AIS/port feed"
    ordered = [
        "date", "route", "rate_usd_per_tonne", "freight_proxy_index",
        "bunker_usd_per_tonne", "brent_usd_per_bbl", "coal_price_usd_per_tonne",
        "usd_inr", "congestion_days", "route_distance_nm", "vessel_class_code",
        "freight_source", "coal_source", "bunker_source", "fx_source", "congestion_source",
    ]
    return result[ordered].dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    master = build_master_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)
    print(f"Wrote {len(master)} weekly observations to {args.output}")
    print(f"Coverage: {master.date.min():%Y-%m-%d} to {master.date.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
