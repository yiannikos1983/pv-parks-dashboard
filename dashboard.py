import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loader import load_all_parks, PARKS

st.set_page_config(page_title="PV Parks Dashboard", layout="wide")

FREQ_MAP = {
    "15 min":    "15min",
    "Hourly":    "h",
    "Daily":     "D",
    "Weekly":    "W",
    "Monthly":   "ME",
    "Quarterly": "QE",
    "Yearly":    "YE",
}
# Period hours for fixed-duration buckets (used for capacity factor)
SLOT_HOURS = {"15min": 0.25, "h": 1.0, "D": 24.0, "W": 168.0}
PARK_COLORS = px.colors.qualitative.Safe


# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading park data…")
def get_data() -> pd.DataFrame:
    return load_all_parks()


# ── Curtailment helpers ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_baseline(_park_data: pd.DataFrame) -> pd.DataFrame:
    """Median production per (park, calendar-month, hour) on non-curtailed daylight slots."""
    nc = _park_data[(_park_data["curtailed"] == 0) & (_park_data["production_kwh"] > 0)].copy()
    nc["month_str"] = nc["timestamp"].dt.to_period("M").astype(str)
    nc["hour"] = nc["timestamp"].dt.hour
    return (
        nc.groupby(["park_name", "month_str", "hour"])["production_kwh"]
        .median()
        .reset_index()
        .rename(columns={"production_kwh": "baseline_kwh"})
    )


def get_curtailed_df(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    curt = df[df["curtailed"] == 1].copy()
    if curt.empty:
        return curt
    curt["month_str"] = curt["timestamp"].dt.to_period("M").astype(str)
    curt["hour"] = curt["timestamp"].dt.hour
    curt = curt.merge(baseline, on=["park_name", "month_str", "hour"], how="left")
    curt["baseline_kwh"] = curt["baseline_kwh"].fillna(0.0)
    curt["prod_fill"] = curt["production_kwh"].fillna(0.0)
    curt["lost_kwh"] = (curt["baseline_kwh"] - curt["prod_fill"]).clip(lower=0)
    curt["lost_eur"] = curt["lost_kwh"] / 1000 * curt["dam_price"].fillna(0.0)
    return curt


def get_curtailment_events(curt: pd.DataFrame) -> pd.DataFrame:
    if curt.empty:
        return pd.DataFrame()
    c = curt.sort_values(["park_name", "timestamp"]).copy()
    c["time_diff_h"] = c.groupby("park_name")["timestamp"].diff().dt.total_seconds() / 3600
    c["new_event"] = c["time_diff_h"].isna() | (c["time_diff_h"] > 2.0)
    c["event_id"] = c["new_event"].cumsum()
    events = (
        c.groupby(["park_name", "event_id"])
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            actual_mwh=("prod_fill", lambda x: x.sum() / 1000),
            baseline_mwh=("baseline_kwh", lambda x: x.sum() / 1000),
            lost_mwh=("lost_kwh", lambda x: x.sum() / 1000),
            lost_eur=("lost_eur", "sum"),
            avg_dam_price=("dam_price", "mean"),
        )
        .reset_index()
    )
    events["duration_h"] = (
        (events["end"] - events["start"]).dt.total_seconds() / 3600
    ).round(1)
    return events.sort_values(["start", "park_name"]).reset_index(drop=True)


# ── KPI card helper ──────────────────────────────────────────────────────────

def kpi(col, label: str, value: str, delta: str = ""):
    col.metric(label, value, delta if delta else None)


# ── Aggregation helper ───────────────────────────────────────────────────────

def period_label_fmt(freq: str) -> str:
    return {"ME": "%b %Y", "QE": "Q%q %Y", "YE": "%Y"}[freq]


def agg_by(df: pd.DataFrame, freq: str, cols: dict) -> pd.DataFrame:
    """Group by park + time period; cols = {output_col: (source_col, agg_fn)}."""
    grouped = df.groupby(
        ["park_name", pd.Grouper(key="timestamp", freq=freq)],
        observed=True,
    )
    result = grouped.agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in cols.items()})
    result = result.reset_index().rename(columns={"timestamp": "period"})
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

all_data = get_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Filters")
    all_parks = list(PARKS.keys())
    selected_parks = st.multiselect("Parks", all_parks, default=all_parks)

    min_date = all_data["timestamp"].min().date()
    max_date = all_data["timestamp"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    agg_option = st.selectbox("Aggregation", list(FREQ_MAP.keys()), index=4)
    freq = FREQ_MAP[agg_option]

if not selected_parks:
    st.warning("Select at least one park in the sidebar.")
    st.stop()

# Park data (full date history) — used for baseline computation
park_data = all_data[all_data["park_name"].isin(selected_parks)].copy()

# Filtered data for display
df = park_data[
    (park_data["timestamp"].dt.date >= start_date)
    & (park_data["timestamp"].dt.date <= end_date)
].copy()

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Pre-compute curtailment baseline from full park history
baseline = compute_baseline(park_data)
curt_df = get_curtailed_df(df, baseline)

# ── Page title ────────────────────────────────────────────────────────────────

st.title("PV Parks Production Dashboard")
st.caption(
    f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}  |  {agg_option}"
)

park_meta = pd.DataFrame([
    {
        "Park": name,
        "Capacity (kW)": PARKS[name]["capacity_kw"],
        "ΕΔΡΕΘ": PARKS[name]["edreth"],
        "Electrification date": PARKS[name]["electrification_date"],
    }
    for name in selected_parks
])
st.dataframe(park_meta, hide_index=True, width="stretch")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Production", "Revenue", "Curtailment", "DAM Price Exposure"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRODUCTION PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    total_mwh = df["production_kwh"].sum() / 1000
    valid_slots = df["production_kwh"].notna().sum()
    total_slots = len(df)
    data_avail = valid_slots / total_slots * 100 if total_slots else 0

    # Capacity factor: total production / (sum of capacity × period hours)
    # Using calendar days × 24 h approach per park-period
    cf_agg = agg_by(df, freq, {
        "production_kwh": ("production_kwh", "sum"),
        "capacity_kw": ("capacity_kw", "first"),
    })
    cf_agg["production_mwh"] = cf_agg["production_kwh"] / 1000
    if freq in SLOT_HOURS:
        cf_agg["period_hours"] = SLOT_HOURS[freq]
    elif freq == "ME":
        cf_agg["period_hours"] = cf_agg["period"].dt.days_in_month * 24
    elif freq == "QE":
        cf_agg["period_hours"] = cf_agg["period"].apply(
            lambda t: sum(
                pd.Timestamp(t.year, m, 1).days_in_month
                for m in range(t.month - 2, t.month + 1)
                if 1 <= m <= 12
            ) * 24
        )
    else:
        cf_agg["period_hours"] = cf_agg["period"].apply(
            lambda t: 8784 if t.year % 4 == 0 else 8760
        )
    cf_agg["capacity_factor_pct"] = (
        cf_agg["production_mwh"] / (cf_agg["capacity_kw"] / 1000 * cf_agg["period_hours"]) * 100
    ).clip(upper=100)

    # Capacity-weighted CF per period (parks with no data in a period are naturally absent
    # from cf_agg and therefore don't affect that period's weighted average)
    cf_agg["cf_x_cap"] = cf_agg["capacity_factor_pct"] * cf_agg["capacity_kw"]
    period_cf = cf_agg.groupby("period").agg(
        cf_x_cap_sum=("cf_x_cap", "sum"),
        capacity_sum=("capacity_kw", "sum"),
    )
    period_cf["weighted_cf"] = period_cf["cf_x_cap_sum"] / period_cf["capacity_sum"]
    avg_cf = period_cf["weighted_cf"].mean()

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Total production", f"{total_mwh:,.1f} MWh")
    kpi(c2, "Capacity-weighted avg CF", f"{avg_cf:.1f}%")
    kpi(c3, "Data availability", f"{data_avail:.1f}%")

    st.divider()

    # Stacked bar: production per park per period
    fig1 = px.bar(
        cf_agg,
        x="period",
        y="production_mwh",
        color="park_name",
        barmode="stack",
        labels={"production_mwh": "Production (MWh)", "period": "", "park_name": "Park"},
        title="Injected energy by park",
        color_discrete_sequence=PARK_COLORS,
    )
    fig1.update_layout(legend_title_text="Park")
    st.plotly_chart(fig1, width="stretch")

    # Pivot table: park × period
    st.subheader("Production (MWh) — detailed table")
    pivot = cf_agg.pivot_table(
        index="park_name", columns="period", values="production_mwh", aggfunc="sum"
    )
    col_fmt = {
        "ME":    lambda c: c.strftime("%b %Y"),
        "QE":    lambda c: f"Q{((c.month - 1) // 3) + 1} {c.year}",
        "YE":    lambda c: str(c.year),
        "W":     lambda c: c.strftime("%d %b %Y"),
        "D":     lambda c: c.strftime("%d %b %Y"),
        "h":     lambda c: c.strftime("%d %b %H:00"),
        "15min": lambda c: c.strftime("%d %b %H:%M"),
    }
    pivot.columns = [col_fmt[freq](c) for c in pivot.columns]
    pivot.index.name = "Park"
    st.dataframe(
        pivot.style.background_gradient(cmap="YlGn", axis=None).format("{:.2f}"),
        width="stretch",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REVENUE
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    df2 = df.copy()
    df2["expected_eur"] = (
        df2["production_kwh"].fillna(0) / 1000 * df2["dam_price"].fillna(0)
    )
    df2["delta_eur"] = df2["payment_eur"].fillna(0) - df2["expected_eur"]

    total_actual = df2["payment_eur"].sum()
    total_expected = df2["expected_eur"].sum()
    discrepancy = total_actual - total_expected
    disc_pct = discrepancy / total_expected * 100 if total_expected else 0

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Actual payments", f"€ {total_actual:,.2f}")
    kpi(c2, "Theoretical revenue", f"€ {total_expected:,.2f}")
    kpi(
        c3,
        "Discrepancy",
        f"€ {discrepancy:+,.2f}",
        f"{disc_pct:+.2f}%",
    )

    st.divider()

    # Monthly actual vs theoretical
    rev_agg = agg_by(df2, freq, {
        "payment_eur": ("payment_eur", "sum"),
        "expected_eur": ("expected_eur", "sum"),
    })

    fig2 = go.Figure()
    fig2.add_bar(
        x=rev_agg["period"],
        y=rev_agg["expected_eur"],
        name="Theoretical",
        marker_color="#a8d8ea",
    )
    fig2.add_bar(
        x=rev_agg["period"],
        y=rev_agg["payment_eur"],
        name="Actual",
        marker_color="#2a9d8f",
    )
    fig2.update_layout(
        barmode="group",
        title="Actual vs theoretical revenue",
        xaxis_title="",
        yaxis_title="Revenue (€)",
        legend_title_text="",
    )
    st.plotly_chart(fig2, width="stretch")

    # Anomaly table
    st.subheader("Settlement anomalies  (|actual − expected| > €0.10)")
    anomalies = df2[df2["delta_eur"].abs() > 0.10][
        ["timestamp", "park_name", "production_kwh", "dam_price", "expected_eur", "payment_eur", "delta_eur"]
    ].copy()
    anomalies["production_kwh"] = anomalies["production_kwh"].round(3)
    anomalies.rename(columns={
        "timestamp": "Timestamp",
        "park_name": "Park",
        "production_kwh": "Production (kWh)",
        "dam_price": "DAM Price (€/MWh)",
        "expected_eur": "Expected (€)",
        "payment_eur": "Actual (€)",
        "delta_eur": "Delta (€)",
    }, inplace=True)
    if anomalies.empty:
        st.success("No settlement anomalies found in the selected period.")
    else:
        st.dataframe(anomalies.reset_index(drop=True), width="stretch")

    st.divider()

    # ── Implied captured price vs DAM price ───────────────────────────────────
    st.subheader("Implied captured price vs DAM price")

    ip = df2[
        (df2["production_kwh"] > 0.5)
        & df2["dam_price"].notna()
        & df2["payment_eur"].notna()
    ].copy()
    ip["implied_price"] = ip["payment_eur"] / (ip["production_kwh"] / 1000)
    ip["spread"] = ip["implied_price"] - ip["dam_price"]

    if ip.empty:
        st.info("Insufficient data for implied price analysis.")
    else:
        total_prod = ip["production_kwh"].sum()
        w_implied = (ip["implied_price"] * ip["production_kwh"]).sum() / total_prod
        w_dam    = (ip["dam_price"]     * ip["production_kwh"]).sum() / total_prod
        w_spread = w_implied - w_dam

        c1, c2, c3 = st.columns(3)
        kpi(c1, "Avg implied price (prod-weighted)", f"€ {w_implied:.2f}/MWh")
        kpi(c2, "Avg DAM price (prod-weighted)",     f"€ {w_dam:.2f}/MWh")
        kpi(c3, "Avg spread (implied − DAM)",        f"€ {w_spread:+.2f}/MWh")

        # Monthly production-weighted implied vs DAM price
        ip["prod_x_implied"] = ip["implied_price"] * ip["production_kwh"]
        ip["prod_x_dam"]     = ip["dam_price"]     * ip["production_kwh"]
        monthly_ip = (
            ip.groupby(pd.Grouper(key="timestamp", freq=freq))
            .agg(
                prod_x_implied=("prod_x_implied", "sum"),
                prod_x_dam=("prod_x_dam",     "sum"),
                production_kwh=("production_kwh",  "sum"),
            )
            .reset_index()
            .rename(columns={"timestamp": "period"})
        )
        monthly_ip["implied_price"] = monthly_ip["prod_x_implied"] / monthly_ip["production_kwh"]
        monthly_ip["dam_price_avg"] = monthly_ip["prod_x_dam"]     / monthly_ip["production_kwh"]

        fig_ip = go.Figure()
        fig_ip.add_scatter(
            x=monthly_ip["period"], y=monthly_ip["dam_price_avg"],
            mode="lines+markers", name="DAM price",
            line=dict(color="#999999", dash="dot"),
        )
        fig_ip.add_scatter(
            x=monthly_ip["period"], y=monthly_ip["implied_price"],
            mode="lines+markers", name="Implied captured price",
            line=dict(color="#e76f51"),
        )
        fig_ip.update_layout(
            title="Implied captured price vs DAM price (production-weighted)",
            xaxis_title="", yaxis_title="€/MWh", legend_title_text="",
        )
        st.plotly_chart(fig_ip, width="stretch")

        # Spread distribution per park — box plot
        fig_box = px.box(
            ip, x="park_name", y="spread", color="park_name",
            title="Spread distribution per park  (implied − DAM, €/MWh)",
            labels={"spread": "Spread (€/MWh)", "park_name": "Park"},
            color_discrete_sequence=PARK_COLORS,
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
        st.plotly_chart(fig_box, width="stretch")

        if "Aiginio" in selected_parks:
            st.caption(
                "Aiginio: payment data is all-zero in the source file — its implied price "
                "of €0/MWh reflects a data quality issue, not actual settlement. "
                "Follow up with FORENA."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CURTAILMENT IMPACT
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    if curt_df.empty:
        st.info("No curtailment events in the selected period.")
    else:
        events = get_curtailment_events(curt_df)
        n_events = len(events)
        impactful = curt_df[curt_df["baseline_kwh"] > 0]
        total_lost_mwh = curt_df["lost_kwh"].sum() / 1000
        total_lost_eur = curt_df["lost_eur"].sum()

        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Curtailment events", f"{n_events}")
        kpi(c2, "Impactful slots", f"{len(impactful):,}")
        kpi(c3, "Estimated lost production", f"{total_lost_mwh:,.2f} MWh")
        kpi(c4, "Estimated lost revenue", f"€ {total_lost_eur:,.2f}")

        st.divider()

        # Monthly curtailment impact chart
        curt_agg = (
            curt_df.groupby(
                ["park_name", pd.Grouper(key="timestamp", freq=freq)],
                observed=True,
            )
            .agg(lost_mwh=("lost_kwh", lambda x: x.sum() / 1000),
                 lost_eur=("lost_eur", "sum"))
            .reset_index()
            .rename(columns={"timestamp": "period"})
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig3a = px.bar(
                curt_agg,
                x="period",
                y="lost_mwh",
                color="park_name",
                barmode="stack",
                title="Estimated lost production (MWh)",
                labels={"lost_mwh": "Lost (MWh)", "period": "", "park_name": "Park"},
                color_discrete_sequence=PARK_COLORS,
            )
            st.plotly_chart(fig3a, width="stretch")
        with col_b:
            fig3b = px.bar(
                curt_agg,
                x="period",
                y="lost_eur",
                color="park_name",
                barmode="stack",
                title="Estimated lost revenue (€)",
                labels={"lost_eur": "Lost (€)", "period": "", "park_name": "Park"},
                color_discrete_sequence=PARK_COLORS,
            )
            st.plotly_chart(fig3b, width="stretch")

        # Event detail table
        st.subheader("Curtailment event detail")
        ev_display = events[
            ["park_name", "start", "end", "duration_h",
             "actual_mwh", "baseline_mwh", "lost_mwh", "lost_eur", "avg_dam_price"]
        ].copy()
        ev_display.columns = [
            "Park", "Start", "End", "Duration (h)",
            "Actual (MWh)", "Baseline (MWh)", "Lost (MWh)", "Lost (€)", "Avg DAM (€/MWh)",
        ]
        for col in ["Actual (MWh)", "Baseline (MWh)", "Lost (MWh)"]:
            ev_display[col] = ev_display[col].round(3)
        ev_display["Lost (€)"] = ev_display["Lost (€)"].round(2)
        ev_display["Avg DAM (€/MWh)"] = ev_display["Avg DAM (€/MWh)"].round(2)
        st.dataframe(ev_display, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DAM PRICE EXPOSURE
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    # Only slots with production and a known DAM price
    prod_slots = df[df["production_kwh"] > 0.1].dropna(subset=["dam_price"]).copy()

    if prod_slots.empty:
        st.info("No production slots with DAM price data in the selected period.")
    else:
        neg_slots = prod_slots[prod_slots["dam_price"] < 0]
        low_slots = prod_slots[prod_slots["dam_price"] < 20]
        neg_pct = len(neg_slots) / len(prod_slots) * 100
        low_pct = len(low_slots) / len(prod_slots) * 100

        total_prod = prod_slots["production_kwh"].sum()
        w_avg_price = (
            (prod_slots["production_kwh"] * prod_slots["dam_price"]).sum() / total_prod
            if total_prod > 0 else float("nan")
        )

        c1, c2, c3 = st.columns(3)
        kpi(c1, "Slots with negative price", f"{neg_pct:.1f}% ({len(neg_slots):,} slots)")
        kpi(c2, "Slots with price < €20/MWh", f"{low_pct:.1f}% ({len(low_slots):,} slots)")
        kpi(c3, "Production-weighted avg price", f"€ {w_avg_price:.2f}/MWh")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            fig4a = px.histogram(
                prod_slots,
                x="dam_price",
                color="park_name",
                nbins=60,
                barmode="overlay",
                opacity=0.7,
                title="DAM price distribution during production hours",
                labels={"dam_price": "DAM Price (€/MWh)", "park_name": "Park"},
                color_discrete_sequence=PARK_COLORS,
            )
            fig4a.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="€0")
            fig4a.add_vline(x=20, line_dash="dot", line_color="orange", annotation_text="€20")
            st.plotly_chart(fig4a, width="stretch")

        with col_b:
            # Monthly weighted avg price vs simple avg price
            price_agg = (
                prod_slots.groupby(pd.Grouper(key="timestamp", freq=freq))
                .apply(
                    lambda g: pd.Series({
                        "weighted_avg": (g["production_kwh"] * g["dam_price"]).sum() / g["production_kwh"].sum()
                        if g["production_kwh"].sum() > 0 else float("nan"),
                        "simple_avg": g["dam_price"].mean(),
                    })
                )
                .reset_index()
                .rename(columns={"timestamp": "period"})
            )

            fig4b = go.Figure()
            fig4b.add_scatter(
                x=price_agg["period"],
                y=price_agg["simple_avg"],
                mode="lines+markers",
                name="Simple avg DAM price",
                line=dict(color="#999999", dash="dot"),
            )
            fig4b.add_scatter(
                x=price_agg["period"],
                y=price_agg["weighted_avg"],
                mode="lines+markers",
                name="Production-weighted avg price",
                line=dict(color="#2a9d8f"),
            )
            fig4b.update_layout(
                title="Effective price received vs market average",
                xaxis_title="",
                yaxis_title="€/MWh",
                legend_title_text="",
            )
            st.plotly_chart(fig4b, width="stretch")

        # Negative-price production table
        st.subheader("Slots with negative DAM price and active production")
        if neg_slots.empty:
            st.success("No production during negative-price slots in the selected period.")
        else:
            neg_display = neg_slots[
                ["timestamp", "park_name", "production_kwh", "dam_price", "payment_eur", "curtailed"]
            ].copy()
            neg_display["production_kwh"] = neg_display["production_kwh"].round(3)
            neg_display["dam_price"] = neg_display["dam_price"].round(2)
            neg_display["payment_eur"] = neg_display["payment_eur"].round(2)
            neg_display.rename(columns={
                "timestamp": "Timestamp",
                "park_name": "Park",
                "production_kwh": "Production (kWh)",
                "dam_price": "DAM Price (€/MWh)",
                "payment_eur": "Payment (€)",
                "curtailed": "Curtailed",
            }, inplace=True)
            st.dataframe(neg_display.reset_index(drop=True), width="stretch")
