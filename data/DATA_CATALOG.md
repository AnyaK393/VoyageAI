# VoyageAI data catalogue

`processed/voyageai_master_weekly.csv` is the reproducible modelling layer. Build it with:

```bash
venv/bin/python scripts/build_master_dataset.py
```

It joins the three supplied sources over their common observed period and uses Friday as the modelling week.

| Model field | Current source | Treatment | Decision-use status |
| --- | --- | --- | --- |
| `rate_usd_per_tonne` | Statistics Finland monthly sea-freight price series | Index ÷ 100 for the existing prototype scale | Market proxy only; not an India route quote |
| `coal_price_usd_per_tonne` | World Bank Pink Sheet, Australian coal | Monthly series interpolated to week | Usable public market driver |
| `bunker_usd_per_tonne` | World Bank Pink Sheet, Brent | Brent × 7.35 energy-cost proxy | Proxy only; replace with a VLSFO assessment |
| `usd_inr` | Supplied daily USD/INR close | Last available daily close each Friday | Usable FX driver |
| `congestion_days` | No supplied observation | Existing prototype default (1.8) | Do not treat as observed data |

## Data still required for an operational model

The current master dataset makes the prototype data-backed, but it cannot make a route-specific charter decision accurate by itself. Add these as separate, dated feeds; do not fabricate values when a feed is unavailable.

| Priority | Dataset | Recommended source | Minimum fields |
| --- | --- | --- | --- |
| 1 | Route-specific dry-bulk freight / fixture data | Baltic Exchange route assessments or a licensed broker/fixture feed | assessment date, origin, destination, vessel class, rate, unit, source |
| 1 | Bunker VLSFO price at bunker ports | Ship & Bunker, S&P Global/Platts, or supplier quotes (licensed where required) | date, port, fuel grade, USD/mt, source |
| 1 | Port congestion and vessel activity | Port authority berth/anchorage notices plus AIS provider | timestamp, port/terminal, vessels at anchor, vessels berthed, waiting hours |
| 1 | Port restrictions and berth availability | Official port/terminal notices | effective date, draft, LOA, beam, DWT, berth status |
| 2 | Weather / monsoon risk | India Meteorological Department and route weather provider | timestamp, route/port, wind, wave, cyclone/monsoon alert |
| 2 | Vessel particulars and performance | Class/registry data and vetted owner/broker data | IMO, DWT, draft, LOA, beam, consumption, speed |
| 2 | Actual fixture outcomes | Internal chartering/ERP records | fixture date, route, vessel, cargo, laycan, agreed rate, demurrage, realized cost |

## Quality gates

Before training, reject or flag a row when: the timestamp is missing; a unit is unknown; a price is non-positive; an observation is duplicated; an effective port constraint has expired; or a route/vessel identifier cannot be mapped to master data. Track source, retrieval time, source timestamp, unit, and licence in every raw-ingestion record.

The required future target is a `freight_assessment_usd_per_tonne` or equivalent route-specific rate. Once it exists, replace `rate_usd_per_tonne` with that field rather than treating the current proxy backtest as operational accuracy.
