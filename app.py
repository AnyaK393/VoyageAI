import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta

from maritime_ai.data import load_freight_data, ports, vessels
from maritime_ai.decision import (
    contract_options,
    feasibility,
    risk_score,
    find_decision_flip,
)
from maritime_ai.forecast import backtest, forecast


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VoyageAI | Charter Control Room",
    page_icon="⚓",
    layout="wide",
)


# ============================================================
# EXISTING STYLING — UNCHANGED
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #07131f;
        color: #e7eef5;
    }

    [data-testid="stSidebar"] {
        background: #0b1c2c;
        border-right: 1px solid #183247;
    }

    h1, h2, h3 {
        color: #f6fbff !important;
    }

    .eyebrow {
        color: #65d8c2;
        font-weight: 700;
        letter-spacing: .13em;
        font-size: .76rem;
    }

    .hero {
        background: linear-gradient(115deg, #102b42, #0c2232);
        border: 1px solid #28546a;
        border-radius: 18px;
        padding: 1.35rem 1.55rem;
        margin: .4rem 0 1rem;
    }

    .hero h2 {
        margin: 0 0 .4rem;
        font-size: 1.65rem;
    }

    .hero p {
        color: #b9cbd7;
        margin: 0;
    }

    .signal {
        color: #65d8c2;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: #0d2131;
        border: 1px solid #1d4055;
        padding: .8rem;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a5bbc9;
    }

    div[data-testid="stMetricValue"] {
        color: #f6fbff;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.2rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #a9bfcb;
        padding: .65rem .2rem;
    }

    .stTabs [aria-selected="true"] {
        color: #65d8c2 !important;
        border-bottom-color: #65d8c2 !important;
    }

    .stAlert {
        border-radius: 12px;
    }

    .risk-ring {
        width: 148px;
        height: 148px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        margin: .4rem auto;
    }

    .risk-core {
        width: 112px;
        height: 112px;
        background: #07131f;
        border-radius: 50%;
        display: grid;
        place-items: center;
        text-align: center;
        color: #f6fbff;
    }

    .risk-core b {
        font-size: 1.65rem;
        display: block;
    }

    .risk-core span {
        color: #a5bbc9;
        font-size: .75rem;
    }

    .tiny {
        color: #91aab9;
        font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SOURCE / ROUTE DATA
# ============================================================

SOURCE_PORTS = {
    "Indonesia · Kalimantan": {
        "distance_nm": 2400,
        "season": "Monsoon watch: Jun–Sep",
    },
    "Australia · Newcastle": {
        "distance_nm": 5600,
        "season": "Cyclone watch: Nov–Apr",
    },
    "Mozambique · Beira": {
        "distance_nm": 4800,
        "season": "Cyclone watch: Nov–Apr",
    },
    "USA · Hampton Roads": {
        "distance_nm": 11300,
        "season": "Atlantic storm watch: Jun–Nov",
    },
}


# ============================================================
# LOAD DATA
# ============================================================

data, provenance = load_freight_data()
port_table = ports()
vessel_table = vessels()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚓ VoyageAI")
    st.caption("Charter intelligence control room")

    st.divider()

    st.markdown("### Voyage brief")

    origin = st.selectbox(
        "Load port / origin",
        list(SOURCE_PORTS),
    )

    port_name = st.selectbox(
        "Discharge port",
        port_table["port"].tolist(),
    )

    cargo = st.number_input(
        "Cargo per voyage (tonnes)",
        20000,
        180000,
        58000,
        1000,
    )

    laycan_start = st.date_input(
        "Laycan start",
        value=date.today() + timedelta(days=14),
    )

    horizon = st.select_slider(
        "Planning horizon",
        [4, 8, 12],
        value=12,
        format_func=lambda x: f"{x} weeks",
    )

    risk_appetite = st.select_slider(
        "Risk appetite",
        ["Conservative", "Balanced", "Aggressive"],
        value="Balanced",
    )

    # ========================================================
    # WHAT-IF CONTROLS
    # ========================================================

    with st.expander("What-if scenario"):

        st.caption(
            "Changes re-run the forecast, cost model and optimizer."
        )

        bunker_shock = st.slider(
            "Bunker price change",
            -20,
            40,
            0,
            format="%d%%",
        )

        freight_shock = st.slider(
            "Freight-market change",
            -20,
            30,
            0,
            format="%d%%",
        )

        coal_shock = st.slider(
            "Coal-price change",
            -20,
            30,
            0,
            format="%d%%",
        )

        fx_shock = st.slider(
            "USD/INR change",
            -10,
            15,
            0,
            format="%d%%",
        )

        congestion_shock = st.slider(
            "Extra congestion days",
            0,
            7,
            0,
        )

        cargo_change = st.slider(
            "Cargo quantity change",
            0,
            30,
            0,
            format="%d%%",
        )

    st.divider()

    st.caption("Prototype data status")

    st.warning(
        "Proxy freight history is active. Replace it with "
        "verified/licensed CSV data before operational use."
    )


# ============================================================
# CURRENT INPUTS
# ============================================================

port = (
    port_table
    .set_index("port")
    .loc[port_name]
)

analysis_cargo = int(
    cargo * (1 + cargo_change / 100)
)


scenario = {
    "bunker_pct": bunker_shock,
    "freight_pct": freight_shock,
    "coal_pct": coal_shock,
    "fx_pct": fx_shock,
    "congestion_days": congestion_shock,
}


# ============================================================
# BASE MODEL METRICS
# ============================================================

metrics = backtest(data)


# ============================================================
# VESSEL FEASIBILITY
# ============================================================

feasibility_rows = []

for _, candidate in vessel_table.iterrows():

    result = feasibility(
        port,
        candidate,
        analysis_cargo,
    )

    feasibility_rows.append(
        {
            "Vessel": candidate["vessel"],
            "Class": candidate["class"],
            "Capacity (t)": f"{candidate['capacity_tonnes']:,.0f}",
            "Decision": (
                "✓ Eligible"
                if result["feasible"]
                else "✕ Not eligible"
            ),
            "Constraint": (
                ", ".join(
                    k
                    for k, passed in result["checks"].items()
                    if not passed
                )
                or "All checks passed"
            ),
        }
    )


eligible_vessels = vessel_table[
    vessel_table.apply(
        lambda candidate: feasibility(
            port,
            candidate,
            analysis_cargo,
        )["feasible"],
        axis=1,
    )
].copy()


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">DECISION SUPPORT • PS 26006</div>',
    unsafe_allow_html=True,
)

st.title("Charter Control Room")

st.caption(
    f"{origin}  →  {port_name}  ·  "
    f"{analysis_cargo:,.0f} t per voyage  ·  "
    f"Laycan: {laycan_start:%d %b %Y}  ·  "
    f"{horizon}-week planning view"
)


if eligible_vessels.empty:

    st.error(
        "No vessel passes every port and cargo rule. "
        "Reduce cargo or choose another discharge port."
    )

    st.stop()


# ============================================================
# CORE SCENARIO OPTIMIZER
# ============================================================

def optimize_scenario(
    test_scenario: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    One connected scenario run:

        scenario drivers
            ↓
        freight forecast
            ↓
        vessel feasibility shortlist
            ↓
        contract cost model
            ↓
        optimizer
            ↓
        best decision
    """

    frame = forecast(
        data,
        horizon,
        driver_changes=test_scenario,
    )

    # Explicit market shock applied after the model forecast.
    # This represents a scenario change in the freight market.
    freight_change = (
        test_scenario.get("freight_pct", 0) / 100
    )

    frame["rate_usd_per_tonne"] *= (
        1 + freight_change
    )

    shortlist = []

    for _, candidate in eligible_vessels.iterrows():

        candidate_options = contract_options(
            frame,
            candidate,
            analysis_cargo,
            scenario=test_scenario,
        )

        candidate_options["vessel"] = candidate["vessel"]
        candidate_options["vessel_class"] = candidate["class"]

        shortlist.append(candidate_options)

    result = (
        pd.concat(
            shortlist,
            ignore_index=True,
        )
        .sort_values("all_in_usd_per_tonne")
        .reset_index(drop=True)
    )

    return frame, result


# ============================================================
# CURRENT SCENARIO
# ============================================================

forecast_frame, all_options = optimize_scenario(
    scenario
)

winner = all_options.iloc[0]


# ============================================================
# BASELINE SCENARIO
# ============================================================

# No market shock.
# This is used only as the comparison point for the
# What-If Decision Flip feature.

base_forecast_frame, base_options = optimize_scenario({})

base_winner = base_options.iloc[0]


# ============================================================
# WINNER DETAILS
# ============================================================

vessel_name = winner["vessel"]

vessel = (
    eligible_vessels
    .set_index("vessel")
    .loc[vessel_name]
)

feasible_result = feasibility(
    port,
    vessel,
    analysis_cargo,
)


# ============================================================
# RISK
# ============================================================

volatility = float(
    data["rate_usd_per_tonne"]
    .tail(12)
    .std()
)

congestion = float(
    forecast_frame["congestion_days"].mean()
)

score, label = risk_score(
    volatility,
    congestion,
    feasible_result,
)


# ============================================================
# CURRENT WINNER OPTIONS
# ============================================================

options = (
    all_options[
        all_options["vessel"] == vessel_name
    ]
    .drop(
        columns=["vessel", "vessel_class"]
    )
    .reset_index(drop=True)
)


fallbacks = (
    all_options
    .drop_duplicates("vessel")
    .iloc[1:3]
)


# Demo FX conversion.
# Keep this unchanged for the prototype.
inr_rate = 83.0


# ============================================================
# FORECAST CONFIDENCE BAND
# ============================================================

forecast_frame["lower_band"] = (
    forecast_frame["rate_usd_per_tonne"]
    - 1.96
    * metrics["rmse"]
    * np.sqrt(
        np.arange(1, horizon + 1) / horizon
    )
)

forecast_frame["upper_band"] = (
    forecast_frame["rate_usd_per_tonne"]
    + 1.96
    * metrics["rmse"]
    * np.sqrt(
        np.arange(1, horizon + 1) / horizon
    )
)


# ============================================================
# RECOMMENDATION TEXT
# ============================================================

contract_copy = {
    "Spot (1 voyage)": (
        "Use when flexibility matters more than price certainty."
    ),
    "3-voyage contract": (
        "Balances commitment with ability to react to market change."
    ),
    "6-voyage contract": (
        "Best unit cost in the current forecast; "
        "locks the most exposure."
    ),
}


rate_direction = (
    "rising"
    if forecast_frame["rate_usd_per_tonne"].iloc[-1]
    > forecast_frame["rate_usd_per_tonne"].iloc[0]
    else "falling"
)


recommended_window = (
    laycan_start
    if rate_direction == "rising"
    else forecast_frame.loc[
        forecast_frame["rate_usd_per_tonne"].idxmin(),
        "date",
    ].date()
)


# ============================================================
# HERO RECOMMENDATION
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">RECOMMENDATION</div>
        <h2>
            {winner["contract"]} with {vessel_name}
        </h2>
        <p>
            <span class="signal">
                ${winner["saving_vs_spot_usd"]:,.0f} estimated saving
            </span>
            versus repeated spot booking.
            <b>Recommended booking window:</b>
            {recommended_window:%d %b %Y}.
            {contract_copy[winner["contract"]]}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Forecast freight",
    f"${forecast_frame['rate_usd_per_tonne'].mean():.2f}/t",
    f"{horizon}-week average",
)

m2.metric(
    "All-in cost",
    f"${winner['all_in_usd_per_tonne']:.2f}/t",
    f"{winner['voyages']} voyage plan",
)

m3.metric(
    "Risk signal",
    f"{label} · {score}/100",
    f"{congestion:.1f} expected congestion days",
)

m4.metric(
    "Feasibility",
    "PASS",
    f"{vessel['class']} · {vessel['capacity_tonnes']:,.0f} t",
)

m5.metric(
    "Saving vs spot",
    f"₹{winner['saving_vs_spot_usd'] * inr_rate / 1_000_000:.2f}M",
    f"${winner['saving_vs_spot_usd']:,.0f}",
)


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    market_tab,
    fleet_tab,
    contract_tab,
    risk_tab,
) = st.tabs(
    [
        "Executive overview",
        "Market intelligence",
        "Fleet & port",
        "Contract optimizer",
        "Risk & explainability",
    ]
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

with overview_tab:

    left, right = st.columns([1.25, 1])

    with left:

        st.subheader("Decision summary")

        st.markdown(
            f"""
            - **Book:** {winner['contract']} with **{vessel_name}**.
            - **Window:** {recommended_window:%d %b %Y};
              the forecast trend is **{rate_direction}**.
            - **Feasibility:** vessel is compatible with
              **{port_name}** at {analysis_cargo:,.0f} t.
            - **Action:** seek two firm broker quotes,
              confirm berth availability, and validate vessel
              laycan before award.
            """
        )

        st.subheader("Contract choices")

        show_options = options[
            [
                "contract",
                "voyages",
                "rate_usd_per_tonne",
                "all_in_usd_per_tonne",
                "saving_vs_spot_usd",
            ]
        ].copy()

        show_options.columns = [
            "Contract",
            "Voyages",
            "Freight rate ($/t)",
            "All-in ($/t)",
            "Saving vs spot ($)",
        ]

        st.dataframe(
            show_options.style.format(
                {
                    "Freight rate ($/t)": "${:.2f}",
                    "All-in ($/t)": "${:.2f}",
                    "Saving vs spot ($)": "${:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The recommendation is selected from every feasible "
            "vessel × contract combination, ranked by expected "
            "all-in cost."
        )

    with right:

        st.subheader("Voyage context")

        st.metric(
            "Indicative sea distance",
            f"{SOURCE_PORTS[origin]['distance_nm']:,} nm",
        )

        st.info(
            f"**Seasonal watch:** "
            f"{SOURCE_PORTS[origin]['season']}"
        )

        st.info(
            f"**Port profile:** "
            f"draft {port['max_draft_m']:.1f} m · "
            f"LOA {port['max_loa_m']:.0f} m · "
            f"beam {port['max_beam_m']:.0f} m"
        )

        st.caption(
            "Planning context only; verify the live route, "
            "weather and berth situation."
        )

    st.subheader("Fallback recommendations")

    if fallbacks.empty:

        st.caption(
            "No second eligible vessel is available under "
            "the present port and cargo constraints."
        )

    else:

        fallback_view = fallbacks[
            [
                "vessel",
                "vessel_class",
                "contract",
                "all_in_usd_per_tonne",
                "saving_vs_spot_usd",
            ]
        ].copy()

        fallback_view.columns = [
            "Fallback vessel",
            "Class",
            "Best contract",
            "All-in ($/t)",
            "Saving vs spot ($)",
        ]

        st.dataframe(
            fallback_view.style.format(
                {
                    "All-in ($/t)": "${:.2f}",
                    "Saving vs spot ($)": "${:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "Data provenance — what each number means"
    ):

        source_table = pd.DataFrame(
            [
                [
                    "Freight forecast",
                    "Proxy",
                    provenance,
                ],
                [
                    "Bunker & congestion",
                    "Proxy",
                    "Deterministic demo inputs",
                ],
                [
                    "Port limits",
                    "Reference",
                    "Replace with official port/terminal notices",
                ],
                [
                    "Vessel specifications",
                    "Reference",
                    "Replace with validated vessel registry/AIS data",
                ],
                [
                    "FX rate",
                    "Proxy",
                    f"Fixed at ₹{inr_rate:.0f}/USD for demo",
                ],
                [
                    "Weather/season",
                    "Rule",
                    "Route seasonal advisory, not live weather",
                ],
            ],
            columns=[
                "Input",
                "Status",
                "Current source",
            ],
        )

        st.dataframe(
            source_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MARKET INTELLIGENCE
# ============================================================

with market_tab:

    st.subheader("Freight outlook")

    st.caption(
        f"Ridge Regression using rate lags, recent average, "
        f"bunker proxy and congestion proxy. Data: {provenance}."
    )

    chart_data = pd.concat(
        [
            data[
                [
                    "date",
                    "rate_usd_per_tonne",
                ]
            ].assign(series="Historical"),

            forecast_frame[
                [
                    "date",
                    "rate_usd_per_tonne",
                ]
            ].assign(series="Forecast"),
        ]
    ).pivot(
        index="date",
        columns="series",
        values="rate_usd_per_tonne",
    )

    st.line_chart(
        chart_data,
        height=340,
        x_label="Week",
        y_label="Freight rate (USD per tonne)",
    )

    st.markdown("#### Forecast confidence range")

    st.area_chart(
        forecast_frame.set_index("date")[
            [
                "lower_band",
                "upper_band",
            ]
        ],
        height=190,
        x_label="Forecast week",
        y_label="USD per tonne",
    )

    st.caption(
        "The shaded range is a prototype uncertainty estimate "
        "based on backtest RMSE; it is not a market guarantee."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Walk-forward MAE",
        f"${metrics['mae']:.2f}/t",
    )

    c2.metric(
        "Walk-forward RMSE",
        f"${metrics['rmse']:.2f}/t",
    )

    c3.metric(
        "Backtest points",
        metrics["n_predictions"],
    )

    backtest_frame = pd.DataFrame(
        {
            "Actual": metrics["actuals"],
            "Predicted": metrics["predictions"],
        },
        index=data["date"].tail(
            metrics["n_predictions"]
        ),
    )

    st.markdown(
        "#### Backtest — predicted vs actual"
    )

    st.line_chart(
        backtest_frame.tail(26),
        height=250,
        x_label="Week",
        y_label="USD per tonne",
    )

    st.caption(
        "Do not present proxy-data backtest scores as real "
        "market-performance claims."
    )


# ============================================================
# FLEET & PORT
# ============================================================

with fleet_tab:

    st.subheader("Vessel–port feasibility gate")

    st.caption(
        "Candidates must pass draft, LOA, beam, port DWT "
        "and cargo-capacity rules."
    )

    st.dataframe(
        pd.DataFrame(feasibility_rows),
        use_container_width=True,
        hide_index=True,
    )

    check_columns = st.columns(
        len(feasible_result["checks"])
    )

    for column, (rule, passed) in zip(
        check_columns,
        feasible_result["checks"].items(),
    ):

        column.metric(
            rule,
            "PASS" if passed else "FAIL",
        )


# ============================================================
# CONTRACT OPTIMIZER
# ============================================================

with contract_tab:

    st.subheader("Spot vs multi-voyage optimization")

    st.caption(
        "The engine compares expected freight, fuel, port, "
        "waiting and delay costs for each contract length."
    )

    graph = (
        options
        .set_index("contract")[
            [
                "total_cost_usd",
                "delay_cost_usd",
            ]
        ]
        .rename(
            columns={
                "total_cost_usd": "Total expected cost",
                "delay_cost_usd": "Delay allowance",
            }
        )
    )

    st.bar_chart(
        graph,
        height=310,
        y_label="USD",
    )

    total_freight = (
        winner["rate_usd_per_tonne"]
        * analysis_cargo
        * winner["voyages"]
    )

    total_port = (
        85000
        * winner["voyages"]
    )

    total_delay = winner[
        "delay_cost_usd"
    ]

    total_fuel = (
        winner["total_cost_usd"]
        - total_freight
        - total_port
        - total_delay
        - winner["risk_allowance_usd"]
    )

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Freight",
        f"${total_freight:,.0f}",
    )

    b2.metric(
        "Fuel estimate",
        f"${total_fuel:,.0f}",
    )

    b3.metric(
        "Port costs",
        f"${total_port:,.0f}",
    )

    b4.metric(
        "Delay allowance",
        f"${total_delay:,.0f}",
    )

    st.caption(
        f"Risk/commitment allowance: "
        f"${winner['risk_allowance_usd']:,.0f}. "
        f"It increases for longer commitments when the "
        f"active scenario raises fuel, congestion or market uncertainty."
    )

    st.info(
        f"**Decision rule:** {winner['contract']} wins because "
        f"it has the lowest expected all-in cost at "
        f"**${winner['all_in_usd_per_tonne']:.2f}/t**. "
        f"The ₹ value uses a demo conversion of "
        f"₹{inr_rate:.0f}/USD."
    )


    # ========================================================
    # EXISTING SCENARIO WARNING
    # ========================================================

    if any(
        float(value) != 0
        for value in scenario.values()
    ) or cargo_change:

        st.warning(
            f"Scenario recalculated: "
            f"bunker {bunker_shock:+d}%, "
            f"freight {freight_shock:+d}%, "
            f"coal {coal_shock:+d}%, "
            f"USD/INR {fx_shock:+d}%, "
            f"congestion +{congestion_shock} days, "
            f"cargo {cargo_change:+d}%."
        )


    # ========================================================
    # 🔄 WHAT-IF DECISION FLIP
    # ========================================================

    st.markdown("---")

    st.markdown("### 🔄 What-If Decision Flip")

    st.caption(
        "Change the scenario in the sidebar. VoyageAI "
        "re-runs the forecast → cost model → optimizer "
        "and checks whether the optimal decision changes."
    )

    scenario_active = any(
        float(value) != 0
        for value in scenario.values()
    )

    base_decision = (
        f"{base_winner['contract']} · "
        f"{base_winner['vessel']}"
    )

    scenario_decision = (
        f"{winner['contract']} · "
        f"{winner['vessel']}"
    )

    comparison = pd.DataFrame(
        [
            [
                "Baseline",
                base_winner["contract"],
                base_winner["vessel"],
                base_winner["all_in_usd_per_tonne"],
            ],
            [
                "What-if scenario",
                winner["contract"],
                winner["vessel"],
                winner["all_in_usd_per_tonne"],
            ],
        ],
        columns=[
            "Case",
            "Contract",
            "Vessel",
            "All-in cost ($/t)",
        ],
    )

    st.dataframe(
        comparison.style.format(
            {
                "All-in cost ($/t)": "${:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DECISION FLIPPED
    # ========================================================

    if scenario_active:

        if base_decision != scenario_decision:

            st.error(
                f"""
                🔄 **DECISION FLIPPED**

                **{base_decision}**
                →
                **{scenario_decision}**
                """
            )

            changed_factors = []

            if bunker_shock:
                changed_factors.append(
                    f"bunker price {bunker_shock:+d}%"
                )

            if freight_shock:
                changed_factors.append(
                    f"freight market {freight_shock:+d}%"
                )

            if coal_shock:
                changed_factors.append(
                    f"coal price {coal_shock:+d}%"
                )

            if fx_shock:
                changed_factors.append(
                    f"USD/INR {fx_shock:+d}%"
                )

            if congestion_shock:
                changed_factors.append(
                    f"port congestion +{congestion_shock} days"
                )

            reason_text = ", ".join(
                changed_factors
            )

            st.info(
                f"""
                **Why did it flip?**

                The scenario changed {reason_text}.

                VoyageAI recalculated the freight forecast,
                voyage costs, delay exposure and contract
                risk before selecting the new lowest-cost
                feasible option.
                """
            )

        else:

            cost_change = (
                winner["all_in_usd_per_tonne"]
                - base_winner["all_in_usd_per_tonne"]
            )

            st.success(
                f"🟢 **Decision remains stable:** "
                f"{scenario_decision}"
            )

            st.caption(
                f"All-in cost changed by "
                f"${cost_change:+.2f}/t, but the optimal "
                f"charter strategy did not change."
            )

    else:

        st.info(
            "Set a bunker, freight, coal, FX or congestion "
            "shock in the sidebar to see the recommendation react."
        )


# ============================================================
# RISK & EXPLAINABILITY
# ============================================================

with risk_tab:

    st.subheader("Risk and hedge guidance")

    gauge, risk_detail = st.columns(
        [.55, 1.45]
    )

    gauge_color = (
        "#42c5aa"
        if label == "Low"
        else "#f4b35d"
        if label == "Medium"
        else "#ef7582"
    )

    with gauge:

        st.markdown(
            f"""
            <div
                class="risk-ring"
                style="
                    background:
                    conic-gradient(
                        {gauge_color}
                        {score * 3.6}deg,
                        #193548 0deg
                    )
                "
            >
                <div class="risk-core">
                    <div>
                        <b>{score}</b>
                        <span>{label.upper()} RISK</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with risk_detail:

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Market volatility",
            f"${volatility:.2f}/t",
            "Recent 12 weeks",
        )

        r2.metric(
            "Port congestion",
            f"{congestion:.1f} days",
            "Proxy estimate",
        )

        r3.metric(
            "Seasonality",
            "WATCH",
            SOURCE_PORTS[origin]["season"],
        )


    # ========================================================
    # COMPONENT RISKS
    # ========================================================

    st.markdown("#### Component risks")

    market_component = min(
        100,
        round(volatility * 22),
    )

    port_component = min(
        100,
        round(congestion * 18),
    )

    weather_component = (
        40
        if "watch"
        in SOURCE_PORTS[origin]["season"].lower()
        else 15
    )

    vessel_component = (
        12
        if feasible_result["feasible"]
        else 85
    )

    contract_component = {
        "Spot (1 voyage)": 18,
        "3-voyage contract": 32,
        "6-voyage contract": 48,
    }[winner["contract"]]

    component_data = pd.DataFrame(
        {
            "Risk score": [
                market_component,
                port_component,
                weather_component,
                vessel_component,
                contract_component,
            ]
        },
        index=[
            "Market",
            "Port",
            "Weather / season",
            "Vessel",
            "Contract commitment",
        ],
    )

    st.bar_chart(
        component_data,
        height=230,
        y_label="Risk contribution (0–100)",
    )


    # ========================================================
    # HEDGE GUIDANCE
    # ========================================================

    hedge_message = {
        "Conservative": (
            "Protect price exposure: request a fixed "
            "multi-voyage quote and consider an FFA hedge "
            "after treasury review."
        ),
        "Balanced": (
            "Lock the recommended contract only after "
            "comparing two firm broker quotes and "
            "confirming berth availability."
        ),
        "Aggressive": (
            "Keep flexibility: use spot or a shorter "
            "contract only if procurement accepts "
            "forecast uncertainty."
        ),
    }[risk_appetite]

    st.success(
        f"**{risk_appetite} policy:** {hedge_message}"
    )


    # ========================================================
    # WHY VOYAGEAI RECOMMENDS THIS
    # ========================================================

    st.markdown("#### Why VoyageAI recommends this")

    st.markdown(
        f"""
        - **Cost:** {winner['contract']} gives the lowest
          all-in forecast cost.
        - **Feasibility:** {vessel_name} passes all draft,
          LOA, beam, DWT and capacity rules.
        - **Forecast:** expected freight level is
          ${forecast_frame['rate_usd_per_tonne'].mean():.2f}/t
          over the selected horizon.
        - **Risk:** market/port score is
          {label.lower()} at {score}/100.
        - **Decision could change if:** a broker quote
          materially changes, congestion worsens, or a new
          rate forecast shifts the price outlook.
        """
    )

    st.caption(
        "FFA/hedging guidance is educational, not financial advice. "
        "A real deployment needs approved limits and live market data."
    )


    # ========================================================
    # 🎯 DECISION THRESHOLD ENGINE
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 🎯 Decision stability — flip thresholds"
    )

    st.caption(
        "VoyageAI repeatedly re-runs the same "
        "forecast → cost → optimizer pipeline to find "
        "the stress level where the current decision changes."
    )


    # --------------------------------------------------------
    # Threshold runner
    # --------------------------------------------------------

    def run_for_threshold(test_scenario):

        _, test_options = optimize_scenario(
            test_scenario
        )

        return test_options.iloc[0]


    # --------------------------------------------------------
    # Find thresholds from BASELINE
    # --------------------------------------------------------

    bunker_flip, bunker_winner = find_decision_flip(
        run_for_threshold,
        {},
        "bunker_pct",
        max_stress=60,
        step=1,
    )

    freight_flip, freight_winner = find_decision_flip(
        run_for_threshold,
        {},
        "freight_pct",
        max_stress=40,
        step=1,
    )

    congestion_flip, congestion_winner = find_decision_flip(
        run_for_threshold,
        {},
        "congestion_days",
        max_stress=7,
        step=0.25,
    )


    # ========================================================
    # THRESHOLD TABLE
    # ========================================================

    threshold_rows = [
        [
            "Bunker price",
            (
                f"+{bunker_flip:.1f}%"
                if bunker_flip is not None
                else "> +60%"
            ),
            (
                bunker_winner["contract"]
                if bunker_winner is not None
                else "No flip"
            ),
        ],
        [
            "Freight market",
            (
                f"+{freight_flip:.1f}%"
                if freight_flip is not None
                else "> +40%"
            ),
            (
                freight_winner["contract"]
                if freight_winner is not None
                else "No flip"
            ),
        ],
        [
            "Port congestion",
            (
                f"+{congestion_flip:.2f} days"
                if congestion_flip is not None
                else "> +7 days"
            ),
            (
                congestion_winner["contract"]
                if congestion_winner is not None
                else "No flip"
            ),
        ],
    ]


    st.dataframe(
        pd.DataFrame(
            threshold_rows,
            columns=[
                "Variable",
                "Decision flips after",
                "New decision",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # FIND FIRST BREAKING POINT
    # ========================================================

    flip_candidates = [
        (
            "bunker price",
            bunker_flip,
            bunker_winner,
        ),
        (
            "freight market",
            freight_flip,
            freight_winner,
        ),
        (
            "port congestion",
            congestion_flip,
            congestion_winner,
        ),
    ]


    valid_flips = [
        item
        for item in flip_candidates
        if item[1] is not None
    ]


    if valid_flips:

        name, amount, result = min(
            valid_flips,
            key=lambda item: item[1],
        )

        if name == "port congestion":

            threshold_text = (
                f"+{amount:.2f} days"
            )

        else:

            threshold_text = (
                f"+{amount:.1f}%"
            )

        st.warning(
            f"""
            🎯 **Decision breaking point:**

            If {name} worsens by approximately
            **{threshold_text}** from the baseline,
            VoyageAI changes the optimal decision from
            **{base_winner['contract']}**
            to
            **{result['contract']}**.

            This threshold comes from repeatedly running
            the actual forecast → cost → optimizer pipeline,
            not from a UI-only rule.
            """
        )

    else:

        st.success(
            "🟢 **Stable decision:** no decision flip was "
            "detected within the tested stress ranges."
        )