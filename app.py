# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError

# ──────────────────────────────────────────────────────────────────────────────
# PAGE
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FDI Analytics", layout="wide")
st.title("FDI Analytics Dashboard")

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

FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    # EDA data – we try the simple name first, then the older "(9)" one, then Excel
    "cap_csv": "capex_EDA_cleaned_filled.csv",
    "cap_csv_alt": "capex_EDA_cleaned_filled.csv",
    "cap_xlsx": "capex_EDA.xlsx",
}

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

# ──────────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_world_bank() -> pd.DataFrame:
    url = gh_raw_url(FILES["wb"])
    try:
        df = pd.read_csv(url)
    except (URLError, HTTPError, FileNotFoundError) as e:
        raise RuntimeError(f"Could not fetch {FILES['wb']}: {e}")

    country = find_col(df.columns, "country", "country_name", "Country Name")
    year    = find_col(df.columns, "year")
    cont    = find_col(df.columns, "continent", "region")
    score   = find_col(df.columns, "score", "viability_score", "composite_score")
    grade   = find_col(df.columns, "grade", "letter_grade")

    for need, col in [("country", country), ("year", year), ("continent", cont)]:
        if col is None:
            raise ValueError(f"World Bank CSV missing column: {need}. Found: {list(df.columns)}")

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

    # clean grade to ordered categories
    order = ["A+", "A", "B", "C", "D"]
    df["grade"] = df["grade"].astype(str).str.strip()
    df.loc[~df["grade"].isin(order), "grade"] = np.nan
    df["grade"] = pd.Categorical(df["grade"], categories=order, ordered=True)
    return df

def _melt_capex_wide(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    src = find_col(cols, "Source Country", "Source Co", "Country")
    if not src:
        raise ValueError("CAPEX: 'Source Country' column not found.")
    # year columns are 4-digit numbers
    year_cols = [c for c in cols if str(c).isdigit() and len(str(c)) == 4]
    if not year_cols:
        raise ValueError("CAPEX: no 4-digit year columns detected.")
    # keep Grade if present as id var
    id_vars = [src]
    grade_col = find_col(cols, "Grade")
    if grade_col: id_vars.append(grade_col)

    melted = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name="capex"
    ).rename(columns={src: "country", grade_col if grade_col else "": "grade"})
    melted["year"] = pd.to_numeric(melted["year"], errors="coerce").astype("Int64")
    # numeric clean
    def numify(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
        s = str(x).replace(",", "").strip()
        try: return float(s)
        except Exception: return np.nan
    melted["capex"]  = melted["capex"].map(numify)
    melted["country"] = melted["country"].astype(str).str.strip()
    if "grade" in melted.columns:
        order = ["A+", "A", "B", "C", "D"]
        melted["grade"] = melted["grade"].astype(str).str.strip()
        melted.loc[~melted["grade"].isin(order), "grade"] = np.nan
        melted["grade"] = pd.Categorical(melted["grade"], categories=order, ordered=True)
    return melted

@st.cache_data(show_spinner=True)
def load_capex_long() -> pd.DataFrame:
    # try preferred CSV
    for key in ("cap_csv", "cap_csv_alt"):
        try:
            url = gh_raw_url(FILES[key])
            df = pd.read_csv(url)
            return _melt_capex_wide(df)
        except Exception:
            continue
    # fallback: Excel
    try:
        url = gh_raw_url(FILES["cap_xlsx"])
        df_x = pd.read_excel(url, sheet_name=0)
        return _melt_capex_wide(df_x)
    except Exception as e:
        raise RuntimeError(f"Could not load CAPEX from CSV or Excel: {e}")

wb = load_world_bank()
capx = load_capex_long()

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL FILTERS (for SCORING tab) — safe defaults (no post-widget mutation)
# ──────────────────────────────────────────────────────────────────────────────
years_scoring = sorted(wb["year"].dropna().astype(int).unique().tolist())
c1, c2, c3 = st.columns([1, 1, 2], gap="small")

with c1:
    sel_year = st.selectbox("Year", years_scoring, index=0, key="year")

# pre-compute desired continent if a previously selected country exists
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

def filt_wb(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["year"] == sel_year].copy()
    if st.session_state.continent != "All":
        out = out[out["continent"] == st.session_state.continent]
    if st.session_state.country != "All":
        out = out[out["country"] == st.session_state.country]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_scoring, tab_eda = st.tabs(["Scoring", "EDA"])

# =============================================================================
# SCORING TAB (World Bank)
# =============================================================================
with tab_scoring:
    st.caption("Scoring • (World Bank–based)")
    where_title = (
        st.session_state.country
        if st.session_state.country != "All"
        else (st.session_state.continent if st.session_state.continent != "All" else "Worldwide")
    )
    st.subheader(where_title)

    # KPIs (top) when a country is selected
    if st.session_state.country != "All":
        rows = wb[(wb["year"] == st.session_state.year) & (wb["country"] == st.session_state.country)]
        country_score = float(rows["score"].mean()) if not rows.empty else np.nan
        country_grade = rows["grade"].astype(str).dropna().iloc[0] if not rows.empty and rows["grade"].notna().any() else "-"

        ctry_cont = rows["continent"].dropna().iloc[0] if not rows.empty and rows["continent"].notna().any() else None
        cont_avg = float(wb[(wb["year"] == st.session_state.year) & (wb["continent"] == ctry_cont)]["score"].mean()) if ctry_cont else np.nan

        k1, k2, k3 = st.columns(3, gap="large")
        with k1:
            st.metric("Country Score", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
        with k2:
            st.metric("Grade", country_grade)
        with k3:
            label = f"{ctry_cont} Avg Score" if ctry_cont else "Continent Avg Score"
            st.metric(label, "-" if np.isnan(cont_avg) else f"{cont_avg:,.3f}")

    # YoY line (integer ticks, no gridlines)
    t1, t2 = st.columns([1, 2], gap="large")
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
        fig_line = px.line(yoy_df, x="year_str", y="score", markers=True,
                           labels={"year_str": "", "score": "Mean score"}, title=title)
        fig_line.update_xaxes(type="category", categoryorder="array",
                              categoryarray=yoy_df["year_str"].tolist(), showgrid=False)
        fig_line.update_yaxes(showgrid=False)
        fig_line.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=340)
        st.plotly_chart(fig_line, use_container_width=True)

    # Choropleth (blue gradient + zoom)
    with t2:
        map_df = filt_wb(wb)[["country", "score"]].copy()
        if map_df.empty:
            st.info("No data for this selection.")
        else:
            fig_map = px.choropleth(map_df, locations="country", locationmode="country names",
                                    color="score", color_continuous_scale="Blues",
                                    title="Global Performance Map")
            fig_map.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(st.session_state.continent, "world")
            fig_map.update_geos(scope=current_scope, projection_type="natural earth",
                                showcountries=True, showcoastlines=True)
            if st.session_state.continent != "All" or st.session_state.country != "All":
                fig_map.update_geos(fitbounds="locations")
            fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410)
            st.plotly_chart(fig_map, use_container_width=True)

    # bottom charts only when Country == All
    if st.session_state.country == "All":
        b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")

        with b1:
            top_base = filt_wb(wb)[["country", "score"]].dropna()
            top10 = top_base.sort_values("score", ascending=False).head(10)
            if top10.empty:
                st.info("No countries available for Top 10 with this filter.")
            else:
                fig_top = px.bar(top10.sort_values("score"),
                                 x="score", y="country", orientation="h",
                                 color="score", color_continuous_scale="Blues",
                                 labels={"score": "", "country": ""},
                                 title="Top 10 Performing Countries")
                fig_top.update_coloraxes(showscale=False)
                fig_top.update_traces(text=None)
                fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_top, use_container_width=True)

        with b2:
            donut_base = filt_wb(wb)
            if donut_base.empty or donut_base["grade"].isna().all():
                st.info("No grade data for this selection.")
            else:
                grades = ["A+", "A", "B", "C", "D"]
                donut = (donut_base.assign(grade=donut_base["grade"].astype(str))
                                     .loc[lambda d: d["grade"].isin(grades)]
                                     .groupby("grade", as_index=False)["country"].nunique()
                                     .rename(columns={"country": "count"})
                        ).set_index("grade").reindex(grades, fill_value=0).reset_index()
                blues = px.colors.sequential.Blues
                shades = [blues[-1], blues[-2], blues[-3], blues[-4], blues[-5]]
                color_map = {g:c for g,c in zip(grades, shades)}
                fig_donut = px.pie(donut, names="grade", values="count", hole=0.55,
                                   title="Grade Distribution", color="grade",
                                   color_discrete_map=color_map)
                fig_donut.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, showlegend=True)
                st.plotly_chart(fig_donut, use_container_width=True)

        with b3:
            cont_base = wb[wb["year"] == st.session_state.year].copy()
            if st.session_state.continent != "All":
                cont_base = cont_base[cont_base["continent"] == st.session_state.continent]
            cont_bar = (cont_base.groupby("continent", as_index=False)["score"].mean()
                                   .sort_values("score", ascending=True))
            if cont_bar.empty:
                st.info("No continent data for this selection.")
            else:
                fig_cont = px.bar(cont_bar, x="score", y="continent", orientation="h",
                                  color="score", color_continuous_scale="Blues",
                                  labels={"score": "", "continent": ""},
                                  title="Continent Viability Score")
                fig_cont.update_coloraxes(showscale=False)
                fig_cont.update_traces(text=None)
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_cont, use_container_width=True)

    # Indicator Weights (sorted desc, no index)
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
        "Weight (%)": [12, 10, 10, 8, 6, 5, 5, 9, 8, 8, 6, 5, 5],
    }).sort_values("Weight (%)", ascending=False, kind="mergesort")
    st.dataframe(weights, hide_index=True, use_container_width=True)

# =============================================================================
# EDA TAB (CAPEX dataset) — own filters (Year supports 2024 + “All”), Grade filter
# =============================================================================
with tab_eda:
    st.caption("Exploratory Data Analysis • (CAPEX)")

    # Build a country→continent map per year from WB, then join onto CAPEX
    wb_year_cmap = wb[["year", "country", "continent"]].dropna()
    capx_enriched = capx.merge(wb_year_cmap, on=["year", "country"], how="left")

    # ── EDA filters (independent from global ones)
    years_eda = sorted(capx_enriched["year"].dropna().astype(int).unique().tolist())
    # ensure 2024 present if in data; add "All" option
    if 2024 not in years_eda:
        years_eda.append(2024)  # harmless if missing in data; we'll handle empties
    years_eda = sorted(set(years_eda))
    years_eda_with_all = ["All"] + years_eda

    grades_all = ["All", "A+", "A", "B", "C", "D"]

    e1, e2, e3, e4 = st.columns([1, 1, 1, 2], gap="small")
    with e1:
        sel_year_eda = st.selectbox("EDA Year", years_eda_with_all, index=0, key="eda_year")
    with e2:
        sel_cont_eda = st.selectbox("EDA Continent", ["All"] + sorted(capx_enriched["continent"].dropna().unique().tolist()),
                                    index=0, key="eda_cont")
    # Country options depend on continent (EDA scope)
    eda_country_scope = capx_enriched.copy()
    if sel_cont_eda != "All":
        eda_country_scope = eda_country_scope[eda_country_scope["continent"] == sel_cont_eda]
    with e3:
        sel_country_eda = st.selectbox("EDA Country", ["All"] + sorted(eda_country_scope["country"].dropna().unique().tolist()),
                                       index=0, key="eda_country")
    with e4:
        sel_grade_eda = st.selectbox("Grade", grades_all, index=0, key="eda_grade")

    # Helper: filter CAPEX according to EDA controls
    def filt_capx_eda(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if sel_year_eda != "All":
            out = out[out["year"] == int(sel_year_eda)]
        if sel_cont_eda != "All":
            out = out[out["continent"] == sel_cont_eda]
        if sel_country_eda != "All":
            out = out[out["country"] == sel_country_eda]
        if sel_grade_eda != "All" and "grade" in out.columns:
            out = out[out["grade"] == sel_grade_eda]
        return out

    capx_f = filt_capx_eda(capx_enriched)

    # ── TOP: Global CAPEX Trend (always spans all years) • CAPEX Map (for selected year or latest when All)
    tA, tB = st.columns([1.6, 2], gap="large")

    with tA:
        # Trend by year for the current continent/country/grade filters (but spanning all years)
        trend_all = capx_enriched.copy()
        if sel_cont_eda != "All":
            trend_all = trend_all[trend_all["continent"] == sel_cont_eda]
        if sel_country_eda != "All":
            trend_all = trend_all[trend_all["country"] == sel_country_eda]
        if sel_grade_eda != "All" and "grade" in trend_all.columns:
            trend_all = trend_all[trend_all["grade"] == sel_grade_eda]

        trend = trend_all.groupby("year", as_index=False)["capex"].sum().sort_values("year")
        if trend.empty:
            st.info("No CAPEX data for the selected filters.")
        else:
            trend["year_str"] = trend["year"].astype(int).astype(str)
            fig = px.line(trend, x="year_str", y="capex", markers=True,
                          labels={"year_str": "", "capex": "Global CAPEX ($B)"},
                          title="Global CAPEX Trend")
            fig.update_xaxes(type="category", categoryorder="array",
                             categoryarray=trend["year_str"].tolist(), showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tB:
        # If EDA year = All, show latest available year for the current filter context
        map_frame = capx_enriched.copy()
        if sel_cont_eda != "All":
            map_frame = map_frame[map_frame["continent"] == sel_cont_eda]
        if sel_country_eda != "All":
            map_frame = map_frame[map_frame["country"] == sel_country_eda]
        if sel_grade_eda != "All" and "grade" in map_frame.columns:
            map_frame = map_frame[map_frame["grade"] == sel_grade_eda]

        if sel_year_eda == "All":
            if map_frame.empty:
                map_year = None
            else:
                map_year = int(map_frame["year"].dropna().max())
        else:
            map_year = int(sel_year_eda)

        if map_year is None:
            st.info("No CAPEX data for the selected filters.")
        else:
            map_df = map_frame[map_frame["year"] == map_year]
            if map_df.empty:
                st.info("No CAPEX data for the selected filters.")
            else:
                fig = px.choropleth(map_df, locations="country", locationmode="country names",
                                    color="capex", color_continuous_scale="Blues",
                                    title=f"CAPEX Map — {map_year}")
                fig.update_coloraxes(showscale=True)
                scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                             "North America":"north america","South America":"south america",
                             "Oceania":"world","All":"world"}
                current_scope = scope_map.get(sel_cont_eda, "world")
                fig.update_geos(scope=current_scope, projection_type="natural earth",
                                showcountries=True, showcoastlines=True)
                if sel_cont_eda != "All" or sel_country_eda != "All":
                    fig.update_geos(fitbounds="locations")
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)

    # ── BOTTOM: Top 10 by CAPEX (for selected year; if All -> latest), CAPEX Trend by Grade, Top-10 by CAPEX Growth
    bA, bB, bC = st.columns([1.2, 1.2, 1.6], gap="large")

    # Top 10 Countries by CAPEX (level, a single year)
    with bA:
        year_for_level = map_year  # use same logic as the map
        if year_for_level is None:
            st.info("No CAPEX data for Top 10 with this filter.")
        else:
            level_df = capx_f.copy()
            level_df = level_df[level_df["year"] == year_for_level]
            top10 = level_df.dropna(subset=["capex"]).sort_values("capex", ascending=False).head(10)
            if top10.empty:
                st.info("No CAPEX data for Top 10 with this filter.")
            else:
                fig = px.bar(top10.sort_values("capex"),
                             x="capex", y="country", orientation="h",
                             color="capex", color_continuous_scale="Blues",
                             labels={"capex": "", "country": ""},
                             title=f"Top 10 Countries by CAPEX — {year_for_level}")
                fig.update_coloraxes(showscale=False)
                fig.update_traces(text=None)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)

    # CAPEX Trend by Grade (multi-line)
    with bB:
        trend_grade = capx_f.copy()
        if sel_year_eda != "All":
            # to display a trend, we need all years; override to all years but keep continent/country and grade filters
            trend_grade = capx_enriched.copy()
            if sel_cont_eda != "All":
                trend_grade = trend_grade[trend_grade["continent"] == sel_cont_eda]
            if sel_country_eda != "All":
                trend_grade = trend_grade[trend_grade["country"] == sel_country_eda]
            if sel_grade_eda != "All" and "grade" in trend_grade.columns:
                trend_grade = trend_grade[trend_grade["grade"] == sel_grade_eda]
        # group by year+grade
        if "grade" in trend_grade.columns and not trend_grade.empty:
            tg = (trend_grade.groupby(["year", "grade"], as_index=False)["capex"]
                            .sum().sort_values("year"))
            if tg.empty:
                st.info("No CAPEX data for grade trend.")
            else:
                tg["year_str"] = tg["year"].astype(int).astype(str)
                blues = px.colors.sequential.Blues
                shades = [blues[-1], blues[-2], blues[-3], blues[-4], blues[-5]]
                grades = ["A+", "A", "B", "C", "D"]
                cmap = {g:c for g,c in zip(grades, shades)}
                fig = px.line(tg, x="year_str", y="capex", color="grade",
                              color_discrete_map=cmap,
                              labels={"year_str": "", "capex": "CAPEX ($B)", "grade": "Grade"},
                              title="CAPEX Trend by Grade")
                fig.update_xaxes(type="category", categoryorder="array",
                                 categoryarray=sorted(tg["year_str"].unique().tolist()), showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CAPEX data for grade trend.")

    # Top 10 Countries by CAPEX Growth (over time)
    with bC:
        # Base = all years for chosen continent/country and grade
        growth_base = capx_enriched.copy()
        if sel_cont_eda != "All":
            growth_base = growth_base[growth_base["continent"] == sel_cont_eda]
        if sel_country_eda != "All":
            growth_base = growth_base[growth_base["country"] == sel_country_eda]
        if sel_grade_eda != "All" and "grade" in growth_base.columns:
            growth_base = growth_base[growth_base["grade"] == sel_grade_eda]

        if growth_base.empty:
            st.info("No CAPEX data for growth ranking.")
        else:
            # compute growth between first and last available year per country
            agg = (growth_base.groupby(["country", "year"], as_index=False)["capex"].sum())
            first_year = int(agg["year"].min()) if not agg.empty else None
            last_year  = int(agg["year"].max()) if not agg.empty else None
            if first_year is None or last_year is None or first_year == last_year:
                st.info("Not enough years to compute growth.")
            else:
                start = agg[agg["year"] == first_year][["country", "capex"]].rename(columns={"capex": "capex_start"})
                end   = agg[agg["year"] == last_year][["country", "capex"]].rename(columns={"capex": "capex_end"})
                joined = start.merge(end, on="country", how="inner")
                joined["growth_abs"] = joined["capex_end"] - joined["capex_start"]
                # growth % (avoid div by zero)
                joined["growth_pct"] = np.where(joined["capex_start"] == 0,
                                                np.nan,
                                                (joined["capex_end"] - joined["capex_start"]) / np.abs(joined["capex_start"]) * 100.0)
                # rank by absolute growth; change to "growth_pct" if you prefer %
                topg = joined.sort_values("growth_abs", ascending=False).head(10)
                if topg.empty:
                    st.info("No CAPEX growth could be computed.")
                else:
                    fig = px.bar(topg.sort_values("growth_abs"),
                                 x="growth_abs", y="country", orientation="h",
                                 color="growth_abs", color_continuous_scale="Blues",
                                 labels={"growth_abs": "", "country": ""},
                                 title=(f"Top 10 Countries by CAPEX Growth "
                                        f"{'(Grade '+sel_grade_eda+')' if sel_grade_eda!='All' else '(All Grades)'} "
                                        f"[{first_year} → {last_year}]"))
                    fig.update_coloraxes(showscale=False)
                    fig.update_traces(text=None)
                    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                    st.plotly_chart(fig, use_container_width=True)
