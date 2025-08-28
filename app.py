# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import re
from urllib.parse import quote
from urllib.error import URLError, HTTPError

# ──────────────────────────────────────────────────────────────────────────────
# App chrome / theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FDI Analytics", layout="wide")
st.title("FDI Analytics Dashboard")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .metric-value { font-weight: 700 !important; }
      .kpi-box {text-align:center; padding:18px 0;}
      .kpi-title {font-weight:700;}
      .kpi-number {font-size:64px; line-height:1; font-weight:800; margin:10px 0 2px 0;}
      .kpi-sub {opacity:.75; margin-top:0}
      .section-top-gap { margin-top: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Data sources (GitHub raw)
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    "cap_csv": "capex_EDA_cleaned_filled.csv",
    "cap_csv_alt": "capex_EDA_cleaned_filled.csv",
    "cap_xlsx": "capex_EDA.xlsx",
    "sectors": "merged_sectors_data.csv",          # sectors data from your notebook
    "destinations": "merged_destinations_data.csv" # destinations data
}

def gh_raw_url(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def find_col(cols, *cands):
    """Find a column by exact (case-insensitive) name or substring."""
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for cand in cands:
        for col in cols:
            if cand.lower() in col.lower():
                return col
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Load World Bank (Scoring)  + CAPEX (EDA)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_world_bank() -> pd.DataFrame]:
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
    year_cols = [c for c in cols if str(c).isdigit() and len(str(c)) == 4]
    if not year_cols:
        raise ValueError("CAPEX: no 4-digit year columns detected.")
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

    def numify(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
        s = str(x).replace(",", "").strip()
        try: return float(s)
        except Exception: return np.nan

    melted["capex"]   = melted["capex"].map(numify)
    melted["country"] = melted["country"].astype(str).str.strip()
    if "grade" in melted.columns:
        order = ["A+", "A", "B", "C", "D"]
        melted["grade"] = melted["grade"].astype(str).str.strip()
        melted.loc[~melted["grade"].isin(order), "grade"] = np.nan
        melted["grade"] = pd.Categorical(melted["grade"], categories=order, ordered=True)
    return melted

@st.cache_data(show_spinner=True)
def load_capex_long() -> pd.DataFrame:
    for key in ("cap_csv", "cap_csv_alt"):
        try:
            url = gh_raw_url(FILES[key])
            df = pd.read_csv(url)
            return _melt_capex_wide(df)
        except Exception:
            continue
    try:
        url = gh_raw_url(FILES["cap_xlsx"])
        df_x = pd.read_excel(url, sheet_name=0)
        return _melt_capex_wide(df_x)
    except Exception as e:
        raise RuntimeError(f"Could not load CAPEX from CSV or Excel: {e}")

wb   = load_world_bank()
capx = load_capex_long()

# enrich CAPEX with continent
wb_year_cc = wb[["year", "country", "continent"]].dropna()
capx_enriched = capx.merge(wb_year_cc, on=["year", "country"], how="left")

# small helper used inside tabs
def filt_wb_single_year(df: pd.DataFrame, year_any, sel_cont, sel_country, years_wb) -> tuple[pd.DataFrame, int]:
    yy = int(year_any) if (isinstance(year_any, int) and year_any in years_wb) else int(max(years_wb))
    out = df[df["year"] == yy].copy()
    if sel_cont != "All":
        out = out[out["continent"] == sel_cont]
    if sel_country != "All":
        out = out[out["country"] == sel_country]
    return out, yy

# ──────────────────────────────────────────────────────────────────────────────
# Tabs (filters are now INSIDE Scoring and CAPEX only)
# ──────────────────────────────────────────────────────────────────────────────
tab_scoring, tab_eda, tab_sectors, tab_dest = st.tabs(["Scoring", "CAPEX", "Sectors", "Destinations"])

# =============================================================================
# SCORING TAB (now owns Year/Continent/Country filters)
# =============================================================================
with tab_scoring:
    # --- local filters for Scoring
    years_wb  = sorted(wb["year"].dropna().astype(int).unique().tolist())
    years_all = ["All"] + years_wb  # union not needed here; WB drives Scoring

    c1, c2, c3 = st.columns([1, 1, 2], gap="small")
    with c1:
        sel_year_any_sc = st.selectbox("Year", years_all, index=0, key="sc_year_any")

    # auto continent suggestion based on previously chosen country in this tab
    prev_country = st.session_state.get("sc_country", "All")
    suggested_cont = None
    if prev_country != "All":
        rows = wb[(wb["country"] == prev_country)] if sel_year_any_sc == "All" else wb[(wb["year"] == sel_year_any_sc) & (wb["country"] == prev_country)]
        if not rows.empty and rows["continent"].notna().any():
            suggested_cont = rows["continent"].dropna().iloc[0]

    valid_year_for_wb = sel_year_any_sc if (isinstance(sel_year_any_sc, int) and sel_year_any_sc in years_wb) else max(years_wb)
    cont_options = ["All"] + sorted(wb.loc[wb["year"] == valid_year_for_wb, "continent"].dropna().unique().tolist())

    saved_cont = st.session_state.get("sc_continent", "All")
    default_cont = suggested_cont if (suggested_cont in cont_options) else (saved_cont if saved_cont in cont_options else "All")

    with c2:
        sel_cont_sc = st.selectbox("Continent", cont_options, index=cont_options.index(default_cont), key="sc_continent")

    wb_scope = wb[wb["year"] == valid_year_for_wb].copy()
    if sel_cont_sc != "All":
        wb_scope = wb_scope[wb_scope["continent"] == sel_cont_sc]
    country_options = ["All"] + sorted(wb_scope["country"].unique().tolist())

    saved_country = st.session_state.get("sc_country", "All")
    default_country = saved_country if saved_country in country_options else "All"
    with c3:
        sel_country_sc = st.selectbox("Country", country_options, index=country_options.index(default_country), key="sc_country")

    # --- scoring visuals
    st.caption("Scoring • (World Bank–based)")
    where_title = sel_country_sc if sel_country_sc != "All" else (sel_cont_sc if sel_cont_sc != "All" else "Worldwide")
    st.subheader(where_title)

    wb_year_df, scoring_year = filt_wb_single_year(wb, sel_year_any_sc, sel_cont_sc, sel_country_sc, years_wb)

    if sel_country_sc != "All":
        rows = wb[(wb["year"] == scoring_year) & (wb["country"] == sel_country_sc)]
        country_score = float(rows["score"].mean()) if not rows.empty else np.nan
        country_grade = rows["grade"].astype(str).dropna().iloc[0] if not rows.empty and rows["grade"].notna().any() else "-"
        ctry_cont = rows["continent"].dropna().iloc[0] if not rows.empty and rows["continent"].notna().any() else None
        cont_avg = float(wb[(wb["year"] == scoring_year) & (wb["continent"] == ctry_cont)]["score"].mean()) if ctry_cont else np.nan

        k1, k2, k3 = st.columns(3, gap="large")
        with k1:
            st.metric("Country Score", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
        with k2:
            st.metric("Grade", country_grade)
        with k3:
            label = f"{ctry_cont} Avg Score" if ctry_cont else "Continent Avg Score"
            st.metric(label, "-" if np.isnan(cont_avg) else f"{cont_avg:,.3f}")

    t1, t2 = st.columns([1, 2], gap="large")
    with t1:
        if sel_country_sc != "All":
            base = wb[wb["country"] == sel_country_sc]; title = f"Year-over-Year Viability Score — {sel_country_sc}"
        elif sel_cont_sc != "All":
            base = wb[wb["continent"] == sel_cont_sc]; title = f"Year-over-Year Viability Score — {sel_cont_sc}"
        else:
            base = wb.copy(); title = "Year-over-Year Viability Score — Global"
        yoy_df = base.groupby("year", as_index=False)["score"].mean().sort_values("year")
        yoy_df["year_str"] = yoy_df["year"].astype(int).astype(str)
        fig_line = px.line(yoy_df, x="year_str", y="score", markers=True,
                           labels={"year_str": "", "score": "Mean score"}, title=title)
        fig_line.update_xaxes(type="category", categoryorder="array",
                              categoryarray=yoy_df["year_str"].tolist(), showgrid=False)
        fig_line.update_yaxes(showgrid=False)
        fig_line.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=340)
        st.plotly_chart(fig_line, use_container_width=True)

    with t2:
        map_df = wb_year_df[["country", "score"]].copy()
        if map_df.empty:
            st.info("No data for this selection.")
        else:
            fig_map = px.choropleth(map_df, locations="country", locationmode="country names",
                                    color="score", color_continuous_scale="Blues",
                                    title=f"Global Performance Map — {scoring_year}")
            fig_map.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(sel_cont_sc, "world")
            fig_map.update_geos(scope=current_scope, projection_type="natural earth",
                                showcountries=True, showcoastlines=True)
            if sel_cont_sc != "All" or sel_country_sc != "All":
                fig_map.update_geos(fitbounds="locations")
            fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410)
            st.plotly_chart(fig_map, use_container_width=True)

    if sel_country_sc == "All":
        b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")
        with b1:
            top10 = wb_year_df[["country", "score"]].dropna().sort_values("score", ascending=False).head(10)
            if top10.empty: st.info("No countries available for Top 10 with this filter.")
            else:
                fig_top = px.bar(top10.sort_values("score"), x="score", y="country", orientation="h",
                                 color="score", color_continuous_scale="Blues",
                                 labels={"score": "", "country": ""},
                                 title=f"Top 10 Performing Countries — {scoring_year}")
                fig_top.update_coloraxes(showscale=False)
                fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_top, use_container_width=True)
        with b2:
            donut_base = wb_year_df.copy()
            if donut_base.empty or donut_base["grade"].isna().all(): st.info("No grade data for this selection.")
            else:
                grades = ["A+", "A", "B", "C", "D"]
                donut = (donut_base.assign(grade=donut_base["grade"].astype(str))
                                     .loc[lambda d: d["grade"].isin(grades)]
                                     .groupby("grade", as_index=False)["country"].nunique()
                                     .rename(columns={"country": "count"})
                        ).set_index("grade").reindex(grades, fill_value=0).reset_index()
                shades = [px.colors.sequential.Blues[-1-i] for i in range(5)]
                cmap = {g:c for g, c in zip(grades, shades)}
                fig_donut = px.pie(donut, names="grade", values="count", hole=0.55,
                                   title=f"Grade Distribution — {scoring_year}",
                                   color="grade", color_discrete_map=cmap)
                fig_donut.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, showlegend=True)
                st.plotly_chart(fig_donut, use_container_width=True)
        with b3:
            cont_base = wb[wb["year"] == scoring_year].copy()
            if sel_cont_sc != "All": cont_base = cont_base[cont_base["continent"] == sel_cont_sc]
            cont_bar = cont_base.groupby("continent", as_index=False)["score"].mean().sort_values("score", ascending=True)
            if cont_bar.empty: st.info("No continent data for this selection.")
            else:
                fig_cont = px.bar(cont_bar, x="score", y="continent", orientation="h",
                                  color="score", color_continuous_scale="Blues",
                                  labels={"score": "", "continent": ""},
                                  title=f"Continent Viability Score — {scoring_year}")
                fig_cont.update_coloraxes(showscale=False)
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_cont, use_container_width=True)

    # Indicator weights table
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
# CAPEX TAB (now owns Year/Continent/Country + Grade filters)
# =============================================================================
with tab_eda:
    st.caption("CAPEX Analysis")

    # local filters
    years_wb  = sorted(wb["year"].dropna().astype(int).unique().tolist())
    years_cap = sorted(capx_enriched["year"].dropna().astype(int).unique().tolist())
    years_all = ["All"] + sorted(set(years_wb).union(years_cap))

    c1, c2, c3 = st.columns([1, 1, 2], gap="small")
    with c1:
        sel_year_any_eda = st.selectbox("Year", years_all, index=0, key="eda_year_any")

    prev_country = st.session_state.get("eda_country", "All")
    suggested_cont = None
    if prev_country != "All" and isinstance(sel_year_any_eda, int):
        g = wb[(wb["year"] == sel_year_any_eda) & (wb["country"] == prev_country)]
        if not g.empty and g["continent"].notna().any():
            suggested_cont = g["continent"].dropna().iloc[0]

    valid_year_for_wb = int(max(years_wb))
    cont_options = ["All"] + sorted(wb.loc[wb["year"] == valid_year_for_wb, "continent"].dropna().unique().tolist())
    saved_cont = st.session_state.get("eda_continent", "All")
    default_cont = suggested_cont if (suggested_cont in cont_options) else (saved_cont if saved_cont in cont_options else "All")

    with c2:
        sel_cont_eda = st.selectbox("Continent", cont_options, index=cont_options.index(default_cont), key="eda_continent")

    wb_scope = wb[wb["year"] == valid_year_for_wb].copy()
    if sel_cont_eda != "All":
        wb_scope = wb_scope[wb_scope["continent"] == sel_cont_eda]
    country_options = ["All"] + sorted(wb_scope["country"].unique().tolist())

    saved_country = st.session_state.get("eda_country", "All")
    default_country = saved_country if saved_country in country_options else "All"
    with c3:
        sel_country_eda = st.selectbox("Country", country_options, index=country_options.index(default_country), key="eda_country")

    # grade (EDA)
    grade_options = ["All", "A+", "A", "B", "C", "D"]
    auto_grade = st.session_state.get("grade_eda", "All")
    if sel_country_eda != "All" and isinstance(sel_year_any_eda, int):
        g_rows = wb[(wb["year"] == sel_year_any_eda) & (wb["country"] == sel_country_eda)]
        if not g_rows.empty and g_rows["grade"].notna().any():
            gval = str(g_rows["grade"].dropna().iloc[0])
            if gval in grade_options:
                auto_grade = gval
    sel_grade_eda = st.selectbox("Grade (EDA)", grade_options,
                                 index=grade_options.index(auto_grade if auto_grade in grade_options else "All"),
                                 key="grade_eda")

    # filter CAPEX with continent/country/grade; keep all years if Year == All
    capx_eda = capx_enriched.copy()
    if sel_cont_eda != "All":    capx_eda = capx_eda[capx_eda["continent"] == sel_cont_eda]
    if sel_country_eda != "All": capx_eda = capx_eda[capx_eda["country"] == sel_country_eda]
    if sel_grade_eda != "All" and "grade" in capx_eda.columns:
        capx_eda = capx_eda[capx_eda["grade"] == sel_grade_eda]
    if isinstance(sel_year_any_eda, int):
        capx_eda = capx_eda[capx_eda["year"] == sel_year_any_eda]

    e1, e2 = st.columns([1.6, 2], gap="large")
    with e1:
        # KPI if specific country + year; else trend
        if sel_country_eda != "All" and isinstance(sel_year_any_eda, int):
            cval = capx_eda["capex"].sum()
            st.markdown(
                f"""
                <div class="kpi-box">
                  <div class="kpi-title">{sel_country_eda} CAPEX — {sel_year_any_eda}</div>
                  <div class="kpi-number">{cval:,.1f}</div>
                  <div class="kpi-sub">$B</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            trend = capx_eda.groupby("year", as_index=False)["capex"].sum().sort_values("year")
            if trend.empty: st.info("No CAPEX data for the selected filters.")
            else:
                trend["year_str"] = trend["year"].astype(int).astype(str)
                title = (f"{sel_country_eda} CAPEX Trend" if sel_country_eda != "All"
                         else "Global CAPEX Trend")
                fig = px.line(trend, x="year_str", y="capex", markers=True,
                              labels={"year_str": "", "capex": "Global CAPEX ($B)"},
                              title=title)
                fig.update_xaxes(type="category", categoryorder="array",
                                 categoryarray=trend["year_str"].tolist(), showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True)

    with e2:
        if isinstance(sel_year_any_eda, int):
            map_df = capx_eda.copy(); map_title = f"CAPEX Map — {sel_year_any_eda}"
        else:
            map_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
            map_title = "CAPEX Map — All Years (aggregated)"
        if map_df.empty: st.info("No CAPEX data for this selection.")
        else:
            fig = px.choropleth(map_df, locations="country", locationmode="country names",
                                color="capex", color_continuous_scale="Blues", title=map_title)
            fig.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(sel_cont_eda, "world")
            fig.update_geos(scope=current_scope, projection_type="natural earth",
                            showcountries=True, showcoastlines=True)
            if sel_cont_eda != "All" or sel_country_eda != "All": fig.update_geos(fitbounds="locations")
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)

    show_grade_trend = (sel_grade_eda == "All")
    if show_grade_trend:
        b1, b2, b3 = st.columns([1.2, 1.2, 1.6], gap="large")
    else:
        b1, b3 = st.columns([1.2, 1.6], gap="large")

    with b1:
        if isinstance(sel_year_any_eda, int):
            level_df = capx_eda.copy(); title_top10 = f"Top 10 Countries by CAPEX — {sel_year_any_eda}"
        else:
            level_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
            title_top10 = "Top 10 Countries by CAPEX — All Years (aggregated)"
        top10 = level_df.dropna(subset=["capex"]).sort_values("capex", ascending=False).head(10)
        if top10.empty: st.info("No CAPEX data for Top 10 with this filter.")
        else:
            fig = px.bar(top10.sort_values("capex"), x="capex", y="country", orientation="h",
                         color="capex", color_continuous_scale="Blues",
                         labels={"capex": "", "country": ""}, title=title_top10)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)

    if show_grade_trend:
        with b2:
            if "grade" in capx_eda.columns and not capx_eda.empty:
                tg = (capx_eda.assign(grade=capx_eda["grade"].astype(str))
                               .groupby(["year", "grade"], as_index=False, observed=True)["capex"]
                               .sum()
                               .sort_values("year"))
                if tg.empty: st.info("No CAPEX data for grade trend.")
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

    with b3:
        growth_base = capx_eda.copy()
        if growth_base.empty:
            st.info("No CAPEX data for growth ranking.")
        else:
            agg = growth_base.groupby(["country", "year"], as_index=False)["capex"].sum()
            first_year = int(agg["year"].min()) if not agg.empty else None
            last_year  = int(agg["year"].max()) if not agg.empty else None
            if first_year is None or last_year is None or first_year == last_year:
                st.info("Not enough years to compute growth.")
            else:
                start = agg[agg["year"] == first_year][["country", "capex"]].rename(columns={"capex": "capex_start"})
                end   = agg[agg["year"] == last_year][["country", "capex"]].rename(columns={"capex": "capex_end"})
                joined = start.merge(end, on="country", how="inner")
                joined["growth_abs"] = joined["capex_end"] - joined["capex_start"]
                label_grade = f"(Grade {sel_grade_eda})" if sel_grade_eda != "All" else "(All Grades)"
                fig = px.bar(joined.sort_values("growth_abs").tail(10),
                             x="growth_abs", y="country", orientation="h",
                             color="growth_abs", color_continuous_scale="Blues",
                             labels={"growth_abs": "", "country": ""},
                             title=f"Top 10 Countries by CAPEX Growth {label_grade} [{first_year} → {last_year}]")
                fig.update_coloraxes(showscale=False)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SECTORS TAB (unchanged)
# =============================================================================
SECTORS_CANON = [
    "Software & IT services","Business services","Communications","Financial services",
    "Transportation & Warehousing","Real estate","Consumer products","Food and Beverages",
    "Automotive OEM","Automotive components","Chemicals","Pharmaceuticals",
    "Metals","Coal, oil & gas","Space & defence","Leisure & entertainment"
]

SECTOR_COUNTRIES_10 = [
    "United States","United Kingdom","Germany","France","China",
    "Japan","South Korea","Canada","Netherlands","United Arab Emirates"
]

def _numify_generic(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = re.sub(r"[^\d\.\-]", "", str(x))
    try: return float(s)
    except Exception: return np.nan

def _canon_country(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.strip()
    swap = {
        "usa":"United States","us":"United States","u.s.":"United States",
        "uk":"United Kingdom","u.k.":"United Kingdom",
        "south korea":"South Korea","republic of korea":"South Korea",
        "uae":"United Arab Emirates",
    }
    low = s.lower()
    if low in swap: return swap[low]
    t = " ".join(w.capitalize() for w in low.split())
    t = t.replace("Of", "of")
    return t

def _canon_sector(sector: str) -> str:
    if not isinstance(sector, str): return ""
    s = sector.strip().lower()
    s = s.replace("&amp;", "&").replace(" and ", " & ")
    s = s.replace("defense", "defence")
    s = re.sub(r"\s+", " ", s)

    mapping = [
        (["software", "it"], "Software & IT services"),
        (["business services"], "Business services"),
        (["communication"], "Communications"),
        (["financial"], "Financial services"),
        (["transport", "warehouse"], "Transportation & Warehousing"),
        (["real estate"], "Real estate"),
        (["consumer product"], "Consumer products"),
        (["food", "beverage"], "Food and Beverages"),
        (["automotive oem"], "Automotive OEM"),
        (["automotive component"], "Automotive components"),
        (["chemical"], "Chemicals"),
        (["pharma"], "Pharmaceuticals"),
        (["metal"], "Metals"),
        (["coal", "oil", "gas"], "Coal, oil & gas"),
        (["space", "defence"], "Space & defence"),
        (["leisure", "entertain"], "Leisure & entertainment"),
    ]
    for pats, label in mapping:
        if all(p in s for p in pats):
            return label
    return " ".join(w.capitalize() if w != "&" else "&" for w in s.split())

@st.cache_data(show_spinner=True)
def load_sectors_raw() -> pd.DataFrame:
    url = gh_raw_url(FILES["sectors"])
    df = pd.read_csv(url)

    col_country = find_col(df.columns, "country")
    col_sector  = find_col(df.columns, "sector")
    col_comp    = find_col(df.columns, "companies", "# companies", "number of companies")
    col_jobs    = find_col(df.columns, "jobs created", "jobs", "job")
    col_capex   = find_col(df.columns, "capex", "capital expenditure", "capex (in million usd)")
    col_proj    = find_col(df.columns, "projects")

    for need, col in [
        ("country", col_country), ("sector", col_sector),
        ("companies", col_comp), ("jobs", col_jobs),
        ("capex", col_capex), ("projects", col_proj)
    ]:
        if col is None:
            raise ValueError(f"Sectors CSV missing column for {need}. Found: {list(df.columns)}")

    df = df.rename(columns={
        col_country: "country_raw",
        col_sector : "sector_raw",
        col_comp   : "companies",
        col_jobs   : "jobs_created",
        col_capex  : "capex",
        col_proj   : "projects",
    })

    for c in ["companies", "jobs_created", "capex", "projects"]:
        df[c] = df[c].map(_numify_generic)

    df["country"] = df["country_raw"].astype(str).map(_canon_country)
    df["sector"]  = df["sector_raw"].astype(str).map(_canon_sector)

    df = df[df["sector"].isin(SECTORS_CANON)]
    df = df[df["country"].isin(SECTOR_COUNTRIES_10)]

    df = (df.groupby(["country", "sector"], as_index=False)[["companies","jobs_created","capex","projects"]]
            .sum(min_count=1))
    return df

sectors_df = load_sectors_raw()

with tab_sectors:
    st.caption("Sectors Analysis")

    sc1, sc2 = st.columns([1, 2], gap="small")
    with sc1:
        sector_opt = ["All"] + SECTORS_CANON
        sel_sector = st.selectbox("Sector", sector_opt, index=0, key="sector_sel")

    with sc2:
        countries = SECTOR_COUNTRIES_10
        default_c = st.session_state.get("sector_country", countries[0])
        if default_c not in countries: default_c = countries[0]
        sel_sector_country = st.selectbox("Source Country", countries,
                                          index=countries.index(default_c), key="sector_country")

    metric = st.radio("Metric", ["Companies", "Jobs Created", "Capex", "Projects"],
                      horizontal=True, index=0, key="metric_sel")

    cdf = sectors_df[sectors_df["country"] == sel_sector_country].copy()

    if not cdf.empty:
        out_cols = ["country","sector","companies","jobs_created","capex","projects"]
        csv_bytes = cdf[out_cols].rename(columns={
            "country":"Country","sector":"Sector",
            "companies":"Companies","jobs_created":"Jobs Created",
            "capex":"Capex","projects":"Projects"
        }).to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {sel_sector_country} sectors CSV",
            data=csv_bytes,
            file_name=f"{sel_sector_country.lower().replace(' ','_')}_sectors_data.csv",
            mime="text/csv",
            key="dl_country_sectors_csv",
        )

    value_col = {
        "Companies":"companies",
        "Jobs Created":"jobs_created",
        "Capex":"capex",
        "Projects":"projects",
    }[metric]

    if sel_sector == "All":
        bars = cdf[["sector", value_col]].copy()
        bars = bars.set_index("sector").reindex(SECTORS_CANON, fill_value=0).reset_index()
        title = f"{metric} by Sector — {sel_sector_country}"
        if bars[value_col].sum() == 0:
            st.info("No data for this selection.")
        else:
            fig = px.bar(
                bars, x=value_col, y="sector", orientation="h",
                title=title, labels={value_col:"", "sector":""},
                color=value_col, color_continuous_scale="Blues"
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=520)
            st.plotly_chart(fig, use_container_width=True)
    else:
        val = float(cdf.loc[cdf["sector"] == sel_sector, value_col].sum()) if not cdf.empty else 0.0
        unit = {"Companies":"", "Jobs Created":"", "Capex":" (USD m)", "Projects":""}[metric]
        st.markdown(
            f"""
            <div class="kpi-box">
              <div class="kpi-title">{sel_sector_country} — {sel_sector} • {metric}</div>
              <div class="kpi-number">{val:,.0f}</div>
              <div class="kpi-sub">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# DESTINATIONS TAB (unchanged behavior)
# =============================================================================
@st.cache_data(show_spinner=True)
def load_destinations_raw() -> pd.DataFrame:
    url = gh_raw_url(FILES["destinations"])
    df = pd.read_csv(url)

    col_source = find_col(df.columns, "source country", "source_country", "source")
    col_dest   = find_col(df.columns, "destination country", "destination_country", "destination", "dest")
    col_comp   = find_col(df.columns, "companies", "# companies", "number of companies")
    col_jobs   = find_col(df.columns, "jobs created", "jobs", "job")
    col_capex  = find_col(df.columns, "capex", "capital expenditure", "capex (in million usd)")
    col_proj   = find_col(df.columns, "projects")

    for need, col in [
        ("source country", col_source), ("destination country", col_dest),
        ("companies", col_comp), ("jobs", col_jobs),
        ("capex", col_capex), ("projects", col_proj)
    ]:
        if col is None:
            raise ValueError(f"Destinations CSV missing column for {need}. Found: {list(df.columns)}")

    df = df.rename(columns={
        col_source: "source_raw",
        col_dest  : "dest_raw",
        col_comp  : "companies",
        col_jobs  : "jobs_created",
        col_capex : "capex",
        col_proj  : "projects",
    })

    for c in ["companies", "jobs_created", "capex", "projects"]:
        df[c] = df[c].map(_numify_generic)

    df["source_country"]      = df["source_raw"].astype(str).map(_canon_country)
    df["destination_country"] = df["dest_raw"].astype(str).map(_canon_country)

    bad_labels = {"total", "all destinations", "all", "overall"}
    df = df[~df["destination_country"].astype(str).str.strip().str.lower().isin(bad_labels)]

    df = (df.groupby(["source_country","destination_country"], as_index=False)[
            ["companies","jobs_created","capex","projects"]
          ].sum(min_count=1))
    return df

with tab_dest:
    st.caption("Destinations Analysis")

    dest_df = load_destinations_raw()

    src_countries = sorted(dest_df["source_country"].dropna().unique().tolist())
    default_src = st.session_state.get("dest_src", src_countries[0] if src_countries else "")
    if default_src not in src_countries and src_countries:
        default_src = src_countries[0]

    c1, c2 = st.columns([1, 3], gap="small")

    # Source first
    with c2:
        sel_src_country = st.selectbox("Source Country", src_countries,
                                       index=(src_countries.index(default_src) if default_src in src_countries else 0),
                                       key="dest_src")

    # Destinations options for that source
    dest_opts_all = sorted(
        dest_df.loc[dest_df["source_country"] == sel_src_country,
                    "destination_country"].dropna().unique().tolist()
    )
    dest_opts_all = [d for d in dest_opts_all if str(d).strip().lower() != "total"]
    dest_options = ["All"] + dest_opts_all

    with c1:
        sel_dest_country = st.selectbox("Destination Country", dest_options, index=0, key="dest_country")

    metric_dest = st.radio("Metric", ["Companies","Jobs Created","Capex","Projects"],
                           horizontal=True, index=0, key="metric_dest")
    value_col_dest = {
        "Companies":"companies",
        "Jobs Created":"jobs_created",
        "Capex":"capex",
        "Projects":"projects",
    }[metric_dest]

    export = (dest_df[dest_df["source_country"] == sel_src_country]
                 .copy())
    export = export[export["destination_country"].astype(str).str.strip().str.lower() != "total"]
    if not export.empty:
        out_cols = ["source_country","destination_country","companies","jobs_created","capex","projects"]
        csv_bytes = export[out_cols].rename(columns={
            "source_country":"Source Country",
            "destination_country":"Destination Country",
            "companies":"Companies","jobs_created":"Jobs Created",
            "capex":"Capex","projects":"Projects"
        }).to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {sel_src_country} destinations CSV",
            data=csv_bytes,
            file_name=f"{sel_src_country.lower().replace(' ','_')}_destinations_data.csv",
            mime="text/csv",
            key="dl_country_destinations_csv",
        )

    ddf = dest_df[dest_df["source_country"] == sel_src_country].copy()
    ddf = ddf[ddf["destination_country"].astype(str).str.strip().str.lower() != "total"]

    if sel_dest_country == "All":
        bars = (ddf[["destination_country", value_col_dest]]
                    .groupby("destination_country", as_index=False)
                    [value_col_dest].sum()
                    .sort_values(value_col_dest, ascending=False)
                    .head(15))
        title = f"{metric_dest} by Destination Country — {sel_src_country} (Top 15)"
        if bars.empty or (bars[value_col_dest].sum() == 0):
            st.info("No data for this selection.")
        else:
            fig = px.bar(
                bars.sort_values(value_col_dest), x=value_col_dest, y="destination_country",
                orientation="h", title=title, labels={value_col_dest:"", "destination_country":""},
                color=value_col_dest, color_continuous_scale="Blues"
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=520)
            st.plotly_chart(fig, use_container_width=True)
    else:
        val = float(ddf.loc[ddf["destination_country"] == sel_dest_country, value_col_dest].sum()) if not ddf.empty else 0.0
        unit = {"Companies":"", "Jobs Created":"", "Capex":" (USD m)", "Projects":""}[metric_dest]
        st.markdown(
            f"""
            <div class="kpi-box">
              <div class="kpi-title">{sel_src_country} → {sel_dest_country} • {metric_dest}</div>
              <div class="kpi-number">{val:,.0f}</div>
              <div class="kpi-sub">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
