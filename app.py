# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FDI Analytics • Scoring", layout="wide")
st.title("FDI Analytics Dashboard")
st.caption("Scoring • (World Bank–based)")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .metric-value { font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# DATA (GitHub raw)
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
WB_FILE  = "world_bank_data_with_scores_and_continent.csv"

def gh_raw_url(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    for c in cands:
        for col in cols:
            if c.lower() in col.lower(): return col
    return None

@st.cache_data(show_spinner=True)
def load_scoring() -> pd.DataFrame:
    url = gh_raw_url(WB_FILE)
    try:
        df = pd.read_csv(url)
    except (URLError, HTTPError, FileNotFoundError) as e:
        raise RuntimeError(f"Could not fetch {WB_FILE} from GitHub: {e}")

    country = find_col(df.columns, "country", "country_name", "Country Name")
    year    = find_col(df.columns, "year")
    cont    = find_col(df.columns, "continent", "region")
    score   = find_col(df.columns, "score", "viability_score", "composite_score")
    grade   = find_col(df.columns, "grade", "letter_grade")

    req = {"country": country, "year": year, "continent": cont}
    missing = [k for k,v in req.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns in {WB_FILE}: {missing} (found {list(df.columns)})")

    df = df.rename(columns={
        country: "country",
        year: "year",
        cont: "continent",
        **({score: "score"} if score else {}),
        **({grade: "grade"} if grade else {}),
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "score" not in df.columns: df["score"] = np.nan
    if "grade" not in df.columns: df["grade"] = np.nan

    df["country"]   = df["country"].astype(str).str.strip()
    df["continent"] = df["continent"].astype(str).str.strip()

    order = ["A+", "A", "B", "C", "D"]
    df["grade"] = df["grade"].astype(str).str.strip()
    df.loc[~df["grade"].isin(order), "grade"] = np.nan
    df["grade"] = pd.Categorical(df["grade"], categories=order, ordered=True)

    return df

wb = load_scoring()

# ──────────────────────────────────────────────────────────────────────────────
# FILTERS (safe defaults; no post-widget mutation)
# ──────────────────────────────────────────────────────────────────────────────
years = sorted(wb["year"].dropna().astype(int).unique().tolist())
c1, c2, c3 = st.columns([1, 1, 2], gap="small")

with c1:
    sel_year = st.selectbox("Year", years, index=0, key="year")

prev_country = st.session_state.get("country", "All")
desired_cont = None
if prev_country != "All":
    lookup = wb[(wb["year"] == sel_year) & (wb["country"] == prev_country)]["continent"].dropna()
    if not lookup.empty:
        desired_cont = lookup.iloc[0]

cont_options = ["All"] + sorted(wb.loc[wb["year"] == sel_year, "continent"].dropna().unique().tolist())
default_cont = desired_cont if (desired_cont in cont_options) else st.session_state.get("continent", "All")
if default_cont not in cont_options:
    default_cont = "All"
with c2:
    sel_cont = st.selectbox("Continent", cont_options, index=cont_options.index(default_cont), key="continent")

wb_scope = wb[wb["year"] == sel_year].copy()
if sel_cont != "All":
    wb_scope = wb_scope[wb_scope["continent"] == sel_cont]
country_options = ["All"] + sorted(wb_scope["country"].unique().tolist())
default_country = prev_country if prev_country in country_options else "All"
with c3:
    sel_country = st.selectbox("Country", country_options, index=country_options.index(default_country), key="country")

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["year"] == sel_year].copy()
    if st.session_state.continent != "All":
        out = out[out["continent"] == st.session_state.continent]
    if st.session_state.country != "All":
        out = out[out["country"] == st.session_state.country]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────────────────────────────────────
tab_scoring, = st.tabs(["Scoring"])

with tab_scoring:
    where_title = (
        st.session_state.country
        if st.session_state.country != "All"
        else (st.session_state.continent if st.session_state.continent != "All" else "Worldwide")
    )
    st.subheader(where_title)

    # KPIs (shown ABOVE plots when a country is selected)
    if st.session_state.country != "All":
        ctry_rows = wb[(wb["year"] == st.session_state.year) & (wb["country"] == st.session_state.country)]
        country_score = float(ctry_rows["score"].mean()) if not ctry_rows.empty else np.nan
        country_grade = ctry_rows["grade"].astype(str).dropna().iloc[0] if not ctry_rows.empty and ctry_rows["grade"].notna().any() else "-"

        ctry_cont = ctry_rows["continent"].dropna().iloc[0] if not ctry_rows.empty and ctry_rows["continent"].notna().any() else None
        cont_avg = float(wb[(wb["year"] == st.session_state.year) & (wb["continent"] == ctry_cont)]["score"].mean()) if ctry_cont else np.nan

        k1, k2, k3 = st.columns(3, gap="large")
        with k1:
            st.metric("Country Score", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
        with k2:
            st.metric("Grade", country_grade)
        with k3:
            label = f"{ctry_cont} Avg Score" if ctry_cont else "Continent Avg Score"
            st.metric(label, "-" if np.isnan(cont_avg) else f"{cont_avg:,.3f}")

    # TOP ROW: YoY line • Map
    t1, t2 = st.columns([1, 2], gap="large")

    # YoY line (no gridlines; x = categorical '2021','2022','2023')
    with t1:
        if st.session_state.country != "All":
            base = wb[wb["country"] == st.session_state.country]
            title = f"Year-over-Year Viability Score — {st.session_state.country}"
        elif st.session_state.continent != "All":
            base = wb[wb["continent"] == st.session_state.continent]
            title = f"Year-over-Year Viability Score — {st.session_state.continent}"
        else:
            base = wb.copy()
            title = "Year-over-Year Viability Score — Global"

        yoy_df = base.groupby("year", as_index=False)["score"].mean().sort_values("year")
        yoy_df["year_str"] = yoy_df["year"].astype(int).astype(str)

        fig_line = px.line(
            yoy_df, x="year_str", y="score", markers=True,
            labels={"year_str": "", "score": "Mean score"},
            title=title
        )
        # enforce category axis and remove gridlines
        fig_line.update_xaxes(type="category", categoryorder="array", categoryarray=yoy_df["year_str"].tolist(), showgrid=False)
        fig_line.update_yaxes(showgrid=False)
        fig_line.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=340)
        st.plotly_chart(fig_line, use_container_width=True)

    # Choropleth Map (blue gradient + zoom)
    with t2:
        map_df = apply_filters(wb)[["country", "score"]].copy()
        if map_df.empty:
            st.info("No data for this selection.")
        else:
            fig_map = px.choropleth(
                map_df,
                locations="country",
                locationmode="country names",
                color="score",
                color_continuous_scale="Blues",
                title="Global Performance Map",
            )
            fig_map.update_coloraxes(showscale=True)
            scope_map = {
                "Africa": "africa", "Asia": "asia", "Europe": "europe",
                "North America": "north america", "South America": "south america",
                "Oceania": "world", "All": "world"
            }
            current_scope = scope_map.get(st.session_state.continent, "world")
            fig_map.update_geos(scope=current_scope, projection_type="natural earth", showcountries=True, showcoastlines=True)
            if st.session_state.continent != "All" or st.session_state.country != "All":
                fig_map.update_geos(fitbounds="locations")
            fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410)
            st.plotly_chart(fig_map, use_container_width=True)

    # BOTTOM: charts if no specific country
    if st.session_state.country == "All":
        b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")

        # Top 10 (remove numbers on bars)
        with b1:
            top_base = apply_filters(wb)[["country", "score"]].dropna()
            top10 = top_base.sort_values("score", ascending=False).head(10)
            if top10.empty:
                st.info("No countries available for Top 10 with this filter.")
            else:
                fig_top = px.bar(
                    top10.sort_values("score"),
                    x="score", y="country", orientation="h",
                    color="score", color_continuous_scale="Blues",
                    labels={"score": "", "country": ""},
                    title="Top 10 Performing Countries"
                )
                fig_top.update_coloraxes(showscale=False)
                # hide value labels and x-axis ticks if you want a cleaner card look
                fig_top.update_traces(text=None)
                # keep axis ticks? If you want none, uncomment next line:
                # fig_top.update_xaxes(showticklabels=False)
                fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_top, use_container_width=True)

        # Grade Distribution (blues)
        with b2:
            donut_base = apply_filters(wb)
            if donut_base.empty or donut_base["grade"].isna().all():
                st.info("No grade data for this selection.")
            else:
                grades = ["A+", "A", "B", "C", "D"]
                donut = (
                    donut_base.assign(grade=donut_base["grade"].astype(str))
                              .loc[lambda d: d["grade"].isin(grades)]
                              .groupby("grade", as_index=False)["country"].nunique()
                              .rename(columns={"country": "count"})
                ).set_index("grade").reindex(grades, fill_value=0).reset_index()

                blues = px.colors.sequential.Blues
                shades = [blues[-1], blues[-2], blues[-3], blues[-4], blues[-5]]
                color_map = {g: c for g, c in zip(grades, shades)}

                fig_donut = px.pie(
                    donut, names="grade", values="count", hole=0.55,
                    title="Grade Distribution", color="grade",
                    color_discrete_map=color_map
                )
                fig_donut.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, showlegend=True)
                st.plotly_chart(fig_donut, use_container_width=True)

        # Continent bars (remove numbers on bars)
        with b3:
            cont_base = wb[wb["year"] == st.session_state.year].copy()
            if st.session_state.continent != "All":
                cont_base = cont_base[cont_base["continent"] == st.session_state.continent]
            cont_bar = (cont_base.groupby("continent", as_index=False)["score"].mean()
                                  .sort_values("score", ascending=True))
            if cont_bar.empty:
                st.info("No continent data for this selection.")
            else:
                fig_cont = px.bar(
                    cont_bar, x="score", y="continent", orientation="h",
                    color="score", color_continuous_scale="Blues",
                    labels={"score": "", "continent": ""},
                    title="Continent Viability Score"
                )
                fig_cont.update_coloraxes(showscale=False)
                fig_cont.update_traces(text=None)
                # Optional: hide x-axis tick numbers too
                # fig_cont.update_xaxes(showticklabels=False)
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_cont, use_container_width=True)

    # Indicator Weights: sort DESC and hide row index
    st.markdown("### Indicator Weights (%)")
    weights = pd.DataFrame({
        "Indicator": [
            "GDP growth (annual %)",
            "GDP per capita, PPP (current international $)",
            "Current account balance (% of GDP)",
            "Foreign direct investment, net outflows (% of GDP)",
            "Inflation, consumer prices (annual %)",
            "Exports of goods and services (% of GDP)",
            "Imports of goods and services (% of GDP)",
            "Political Stability and Absence of Violence/Terrorism: Estimate",
            "Government Effectiveness: Estimate",
            "Control of Corruption: Estimate",
            "Access to electricity (% of population)",
            "Individuals using the Internet (% of population)",
            "Total reserves in months of imports",
        ],
        "Weight (%)": [10, 8, 6, 6, 5, 5, 5, 12, 10, 8, 9, 8, 8],
    }).sort_values("Weight (%)", ascending=False, kind="mergesort")
    # Use dataframe widget to hide index
    st.dataframe(weights, hide_index=True, use_container_width=True)
