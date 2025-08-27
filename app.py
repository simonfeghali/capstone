# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError

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
# Data sources (GitHub raw)
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":           "world_bank_data_with_scores_and_continent.csv",
    "cap_csv":      "capex_EDA_cleaned_filled.csv",
    "cap_csv_alt":  "capex_EDA_cleaned_filled.csv",
    "cap_xlsx":     "capex_EDA.xlsx",
    # Sectors dataset
    "sectors_csv":  "merged_sectors_data.csv",
}

def gh_raw_url(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for c in cands:
        for col in cols:
            if c.lower() in col.lower():
                return col
    return None

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
    if "score" not in df.columns:
        df["score"] = np.nan
    if "grade" not in df.columns:
        df["grade"] = np.nan
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
    if grade_col:
        id_vars.append(grade_col)

    melted = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name="capex"
    ).rename(columns={src: "country", grade_col if grade_col else "": "grade"})
    melted["year"] = pd.to_numeric(melted["year"], errors="coerce").astype("Int64")

    def numify(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            return np.nan

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

@st.cache_data(show_spinner=True)
def load_sectors() -> pd.DataFrame:
    url = gh_raw_url(FILES["sectors_csv"])
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(f"Could not fetch {FILES['sectors_csv']}: {e}")

    cols = list(df.columns)
    c_country  = find_col(cols, "country", "Country")
    c_sector   = find_col(cols, "sector", "industry", "Sector")
    c_comp     = find_col(cols, "companies", "company_count", "num_companies")
    c_jobs     = find_col(cols, "jobs_created", "jobs created", "jobs", "jobs_created_total")
    c_capex    = find_col(cols, "capex", "capex_usd_b", "capex (b)")
    c_projects = find_col(cols, "projects", "num_projects", "project_count")

    rename_map = {}
    if c_country:  rename_map[c_country]  = "country"
    if c_sector:   rename_map[c_sector]   = "sector"
    if c_comp:     rename_map[c_comp]     = "companies"
    if c_jobs:     rename_map[c_jobs]     = "jobs"
    if c_capex:    rename_map[c_capex]    = "capex"
    if c_projects: rename_map[c_projects] = "projects"

    df = df.rename(columns=rename_map)
    for need in ["country", "sector", "companies", "jobs", "capex", "projects"]:
        if need not in df.columns:
            raise ValueError(f"Sectors CSV missing column: {need}. Found: {list(df.columns)}")

    for col in ["companies", "jobs", "capex", "projects"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["country"] = df["country"].astype(str).str.strip()
    df["sector"]  = df["sector"].astype(str).str.strip()
    return df

wb   = load_world_bank()
capx = load_capex_long()
sectors_df = load_sectors()

# Enrich CAPEX with continent
wb_year_cc = wb[["year", "country", "continent"]].dropna()
capx_enriched = capx.merge(wb_year_cc, on=["year", "country"], how="left")

# ──────────────────────────────────────────────────────────────────────────────
# Global filter row (Year, Continent, Country)
# ──────────────────────────────────────────────────────────────────────────────
years_wb  = sorted(wb["year"].dropna().astype(int).unique().tolist())
years_cap = sorted(capx_enriched["year"].dropna().astype(int).unique().tolist())
years_all = ["All"] + sorted(set(years_wb).union(years_cap))

c1, c2, c3 = st.columns([1, 1, 2], gap="small")

with c1:
    sel_year_any = st.selectbox("Year", years_all, index=0, key="year_any")

prev_country = st.session_state.get("country", "All")
suggested_cont = None
if prev_country != "All":
    if isinstance(sel_year_any, int):
        rows = wb[(wb["year"] == sel_year_any) & (wb["country"] == prev_country)]
    else:
        rows = wb[wb["country"] == prev_country]
    if not rows.empty and rows["continent"].notna().any():
        suggested_cont = rows["continent"].dropna().iloc[0]

valid_year_for_wb = sel_year_any if (isinstance(sel_year_any, int) and sel_year_any in years_wb) else max(years_wb)
cont_options = ["All"] + sorted(wb.loc[wb["year"] == valid_year_for_wb, "continent"].dropna().unique().tolist())

saved_cont = st.session_state.get("continent", "All")
default_cont = suggested_cont if (suggested_cont in cont_options) else (saved_cont if saved_cont in cont_options else "All")

with c2:
    sel_cont = st.selectbox("Continent", cont_options, index=cont_options.index(default_cont), key="continent")

wb_scope = wb[wb["year"] == valid_year_for_wb].copy()
if sel_cont != "All":
    wb_scope = wb_scope[wb_scope["continent"] == sel_cont]
country_options = ["All"] + sorted(wb_scope["country"].unique().tolist())
saved_country = st.session_state.get("country", "All")
default_country = saved_country if saved_country in country_options else "All"
with c3:
    sel_country = st.selectbox("Country", country_options, index=country_options.index(default_country), key="country")

def filt_wb_single_year(df: pd.DataFrame, year_any) -> tuple[pd.DataFrame, int]:
    yy = int(year_any) if (isinstance(year_any, int) and year_any in years_wb) else max(years_wb)
    out = df[df["year"] == yy].copy()
    if sel_cont != "All":
        out = out[out["continent"] == sel_cont]
    if sel_country != "All":
        out = out[out["country"] == sel_country]
    return out, yy

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_scoring, tab_eda, tab_sectors = st.tabs(["Scoring", "EDA", "Sectors"])

# =============================================================================
# SCORING TAB (UNCHANGED)
# =============================================================================
with tab_scoring:
    st.caption("Scoring • (World Bank–based)")
    where_title = sel_country if sel_country != "All" else (sel_cont if sel_cont != "All" else "Worldwide")
    st.subheader(where_title)

    wb_year_df, scoring_year = filt_wb_single_year(wb, sel_year_any)

    if sel_country != "All":
        rows = wb[(wb["year"] == scoring_year) & (wb["country"] == sel_country)]
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
        if sel_country != "All":
            base = wb[wb["country"] == sel_country]; title = f"Year-over-Year Viability Score — {sel_country}"
        elif sel_cont != "All":
            base = wb[wb["continent"] == sel_cont]; title = f"Year-over-Year Viability Score — {sel_cont}"
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
            fig_map = px.choropleth(
                map_df, locations="country", locationmode="country names",
                color="score", color_continuous_scale="Blues",
                title=f"Global Performance Map — {scoring_year}"
            )
            fig_map.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(sel_cont, "world")
            fig_map.update_geos(scope=current_scope, projection_type="natural earth",
                                showcountries=True, showcoastlines=True)
            if sel_cont != "All" or sel_country != "All":
                fig_map.update_geos(fitbounds="locations")
            fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410)
            st.plotly_chart(fig_map, use_container_width=True)

    if sel_country == "All":
        b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")
        with b1:
            top10 = wb_year_df[["country", "score"]].dropna().sort_values("score", ascending=False).head(10)
            if top10.empty:
                st.info("No countries available for Top 10 with this filter.")
            else:
                fig_top = px.bar(
                    top10.sort_values("score"),
                    x="score", y="country", orientation="h",
                    color="score", color_continuous_scale="Blues",
                    labels={"score": "", "country": ""},
                    title=f"Top 10 Performing Countries — {scoring_year}"
                )
                fig_top.update_coloraxes(showscale=False)
                fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_top, use_container_width=True)
        with b2:
            donut_base = wb_year_df.copy()
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
                shades = [px.colors.sequential.Blues[-1-i] for i in range(5)]
                cmap = {g: c for g, c in zip(grades, shades)}
                fig_donut = px.pie(
                    donut, names="grade", values="count", hole=0.55,
                    title=f"Grade Distribution — {scoring_year}",
                    color="grade", color_discrete_map=cmap
                )
                fig_donut.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, showlegend=True)
                st.plotly_chart(fig_donut, use_container_width=True)
        with b3:
            cont_base = wb[wb["year"] == scoring_year].copy()
            if sel_cont != "All":
                cont_base = cont_base[cont_base["continent"] == sel_cont]
            cont_bar = cont_base.groupby("continent", as_index=False)["score"].mean().sort_values("score", ascending=True)
            if cont_bar.empty:
                st.info("No continent data for this selection.")
            else:
                fig_cont = px.bar(
                    cont_bar,
                    x="score", y="continent", orientation="h",
                    color="score", color_continuous_scale="Blues",
                    labels={"score": "", "continent": ""},
                    title=f"Continent Viability Score — {scoring_year}"
                )
                fig_cont.update_coloraxes(showscale=False)
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_cont, use_container_width=True)

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
# EDA TAB (UNCHANGED)
# =============================================================================
with tab_eda:
    st.caption("Exploratory Data Analysis • (CAPEX)")

    grade_options = ["All", "A+", "A", "B", "C", "D"]
    auto_grade = st.session_state.get("grade_eda", "All")
    if sel_country != "All" and isinstance(sel_year_any, int):
        g_rows = wb[(wb["year"] == sel_year_any) & (wb["country"] == sel_country)]
        if not g_rows.empty and g_rows["grade"].notna().any():
            gval = str(g_rows["grade"].dropna().iloc[0])
            if gval in grade_options:
                auto_grade = gval
    sel_grade_eda = st.selectbox(
        "Grade (EDA)", grade_options,
        index=grade_options.index(auto_grade if auto_grade in grade_options else "All"),
        key="grade_eda"
    )

    capx_eda = capx_enriched.copy()
    if sel_cont != "All":
        capx_eda = capx_eda[capx_eda["continent"] == sel_cont]
    if sel_country != "All":
        capx_eda = capx_eda[capx_eda["country"] == sel_country]
    if sel_grade_eda != "All" and "grade" in capx_eda.columns:
        capx_eda = capx_eda[capx_eda["grade"] == sel_grade_eda]
    if isinstance(sel_year_any, int):
        capx_eda = capx_eda[capx_eda["year"] == sel_year_any]

    country_selected = sel_country != "All"
    specific_year_selected = isinstance(sel_year_any, int)

    e1, e2 = st.columns([1.2, 2], gap="large")

    with e1:
        if country_selected and specific_year_selected:
            cap_val = float(capx_eda["capex"].sum()) if not capx_eda.empty else np.nan
            value = "-" if np.isnan(cap_val) else f"{cap_val:,.1f}"

            st.markdown(
                f"""
                <div style="padding:22px 8px;">
                  <div style="font-weight:800; font-size:22px; text-align:left;">
                    {sel_country} CAPEX — {sel_year_any}
                  </div>
                  <div style="
                      display:flex;
                      flex-direction:column;
                      align-items:center;
                      justify-content:center;
                      margin-top:8px;">
                    <div style="font-weight:900; font-size:76px; line-height:1;">
                      {value}
                    </div>
                    <div style="opacity:0.7; font-size:14px;">$B</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            trend = capx_eda.groupby("year", as_index=False)["capex"].sum().sort_values("year")
            if trend.empty:
                st.info("No CAPEX data for the selected filters.")
            else:
                if country_selected:
                    trend_title = f"{sel_country} CAPEX Trend"
                elif sel_cont != "All":
                    trend_title = f"{sel_cont} CAPEX Trend"
                else:
                    trend_title = "Global CAPEX Trend"

                trend["year_str"] = trend["year"].astype(int).astype(str)
                fig = px.line(
                    trend, x="year_str", y="capex", markers=True,
                    labels={"year_str": "", "capex": "CAPEX ($B)"},
                    title=trend_title
                )
                fig.update_xaxes(type="category", categoryorder="array",
                                 categoryarray=trend["year_str"].tolist(), showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True)

    with e2:
        if isinstance(sel_year_any, int):
            map_df = capx_eda.copy()
            map_title = f"CAPEX Map — {sel_year_any}"
        else:
            map_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
            map_title = "CAPEX Map — All Years (aggregated)"
        if map_df.empty:
            st.info("No CAPEX data for this selection.")
        else:
            fig = px.choropleth(
                map_df, locations="country", locationmode="country names",
                color="capex", color_continuous_scale="Blues", title=map_title
            )
            fig.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(sel_cont, "world")
            fig.update_geos(scope=current_scope, projection_type="natural earth",
                            showcountries=True, showcoastlines=True)
            if sel_cont != "All" or sel_country != "All":
                fig.update_geos(fitbounds="locations")
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)

    if not country_selected:
        show_grade_trend = (sel_grade_eda == "All")

        if show_grade_trend:
            b1, b2, b3 = st.columns([1.2, 1.2, 1.6], gap="large")
        else:
            b1, b3 = st.columns([1.2, 1.6], gap="large")

        with b1:
            if isinstance(sel_year_any, int):
                level_df = capx_eda.copy()
                title_top10 = f"Top 10 Countries by CAPEX — {sel_year_any}"
            else:
                level_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
                title_top10 = "Top 10 Countries by CAPEX — All Years (aggregated)"
            top10 = level_df.dropna(subset=["capex"]).sort_values("capex", ascending=False).head(10)
            if top10.empty:
                st.info("No CAPEX data for Top 10 with this filter.")
            else:
                fig = px.bar(
                    top10.sort_values("capex"),
                    x="capex", y="country", orientation="h",
                    color="capex", color_continuous_scale="Blues",
                    labels={"capex": "", "country": ""},
                    title=title_top10
                )
                fig.update_coloraxes(showscale=False)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)

        if show_grade_trend:
            with b2:
                if "grade" in capx_eda.columns and not capx_eda.empty:
                    tg = (
                        capx_eda.assign(grade=capx_eda["grade"].astype(str))
                        .groupby(["year", "grade"], as_index=False, observed=True)["capex"]
                        .sum()
                        .sort_values("year")
                    )
                    if tg.empty:
                        st.info("No CAPEX data for grade trend.")
                    else:
                        tg["year_str"] = tg["year"].astype(int).astype(str)
                        blues = px.colors.sequential.Blues
                        shades = [blues[-1], blues[-2], blues[-3], blues[-4], blues[-5]]
                        grades = ["A+", "A", "B", "C", "D"]
                        cmap = {g: c for g, c in zip(grades, shades)}
                        fig = px.line(
                            tg, x="year_str", y="capex", color="grade",
                            color_discrete_map=cmap,
                            labels={"year_str": "", "capex": "CAPEX ($B)", "grade": "Grade"},
                            title="CAPEX Trend by Grade"
                        )
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
                    fig = px.bar(
                        joined.sort_values("growth_abs").tail(10),
                        x="growth_abs", y="country", orientation="h",
                        color="growth_abs", color_continuous_scale="Blues",
                        labels={"growth_abs": "", "country": ""},
                        title=f"Top 10 Countries by CAPEX Growth {label_grade} [{first_year} → {last_year}]"
                    )
                    fig.update_coloraxes(showscale=False)
                    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SECTORS TAB (UPDATED WITH KPI MODE)
# =============================================================================
with tab_sectors:
    st.caption("Sectors Analysis")

    # Restrict to the 10 countries present in the sectors dataset
    sector_countries = sorted(sectors_df["country"].dropna().unique().tolist())
    if len(sector_countries) > 10:
        sector_countries = sector_countries[:10]

    sectors_list = ["All"] + sorted(sectors_df["sector"].dropna().unique().tolist())

    sc1, sc2 = st.columns([1, 1.2], gap="small")
    with sc1:
        sel_sector = st.selectbox("Sector", sectors_list, index=0, key="sectors_sector")
    with sc2:
        default_sector_country = sector_countries[0] if sector_countries else None
        sel_sector_country = st.selectbox(
            "Country (Sectors)", sector_countries,
            index=sector_countries.index(default_sector_country) if default_sector_country in sector_countries else 0,
            key="sectors_country"
        )

    metric_choice = st.radio(
        "Metric",
        ["Companies", "Jobs Created", "Capex", "Projects"],
        horizontal=True,
        index=0,
        key="sectors_metric"
    )
    metric_map = {
        "Companies":   ("companies", "Companies"),
        "Jobs Created":("jobs", "Jobs Created"),
        "Capex":       ("capex", "CAPEX ($B)"),
        "Projects":    ("projects", "Projects"),
    }
    metric_col, metric_label = metric_map[metric_choice]

    # ── KPI mode: specific sector AND specific country
    if sel_sector != "All" and sel_sector_country:
        kdf = sectors_df[(sectors_df["country"] == sel_sector_country) & (sectors_df["sector"] == sel_sector)]
        if kdf.empty:
            st.info("No sector data for this selection.")
        else:
            val = float(kdf[metric_col].sum())
            # Format & unit
            if metric_col == "capex":
                val_str = f"{val:,.1f}"
                unit = "$B"
            else:
                val_str = f"{int(round(val)):,}"
                unit = ""

            st.markdown(
                f"""
                <div style="padding:22px 8px;">
                  <div style="font-weight:800; font-size:22px; text-align:left;">
                    {sel_sector_country} — {sel_sector} • {metric_label}
                  </div>
                  <div style="
                      display:flex;
                      flex-direction:column;
                      align-items:center;
                      justify-content:center;
                      margin-top:8px;">
                    <div style="font-weight:900; font-size:76px; line-height:1;">
                      {val_str}
                    </div>
                    <div style="opacity:0.7; font-size:14px;">{unit}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # ── Bar chart modes (unchanged behaviour)
        if sel_sector == "All":
            filtered = sectors_df[sectors_df["country"] == sel_sector_country]
            grp_dim = "sector"
            title = f"{metric_label} by Sector — {sel_sector_country}"
        else:
            filtered = sectors_df[sectors_df["sector"] == sel_sector]
            filtered = filtered[filtered["country"].isin(sector_countries)]
            grp_dim = "country"
            title = f"{metric_label} in {sel_sector} Sector — 10 Countries"

        if filtered.empty:
            st.info("No sector data for this selection.")
        else:
            plot_df = filtered.groupby(grp_dim, as_index=False)[metric_col].sum()
            plot_df = plot_df.sort_values(metric_col, ascending=True)

            fig = px.bar(
                plot_df,
                x=metric_col, y=grp_dim, orientation="h",
                color=metric_col, color_continuous_scale="Blues",
                labels={metric_col: "", grp_dim: ""},
                title=title
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=520)
            st.plotly_chart(fig, use_container_width=True)
