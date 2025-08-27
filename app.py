# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import quote

# ──────────────────────────────────────────────────────────────────────────────
# Page & styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FDI Analytics", layout="wide")
st.title("FDI Analytics Dashboard")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .metric-value { font-weight: 700 !important; }
      .kpi-big { font-size: 64px; font-weight: 800; line-height: 1; }
      .kpi-sub { font-size: 14px; color: #8a93a6; margin-top: -0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Data sources
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    # scoring/eda data
    "wb": "world_bank_data_with_scores_and_continent.csv",
    "cap_csv": "capex_EDA_cleaned_filled.csv",
    "cap_csv_alt": "capex_EDA_cleaned_filled.csv",
    "cap_xlsx": "capex_EDA.xlsx",

    # sectors & destinations
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
}

def gh_raw_url(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

# robust column finder
def find_col(cols, *cands):
    low = {str(c).lower().strip(): c for c in cols}
    for c in cands:
        key = c.lower().strip()
        if key in low:
            return low[key]
    for c in cands:
        wanted = c.lower()
        for col in cols:
            if wanted in str(col).lower():
                return col
    return None

# sectors shown on bar charts (as in your notebook)
SELECTED_SECTORS = [
    "Software & IT services", "Business services", "Communications",
    "Financial services", "Transportation & Warehousing", "Real estate",
    "Consumer products", "Food and Beverages", "Automotive OEM",
    "Automotive components", "Chemicals", "Pharmaceuticals", "Metals",
    "Coal, oil & gas", "Space & defence", "Leisure & entertainment"
]

# optional normalization map
SECTOR_NAME_MAP = {
    "Software & IT Services": "Software & IT services",
    "IT & Software": "Software & IT services",
    "Business Services": "Business services",
    "Communication": "Communications",
    "Financial Services": "Financial services",
    "Transport & Warehousing": "Transportation & Warehousing",
    "Food & Beverages": "Food and Beverages",
    "Automotive Oem": "Automotive OEM",
    "Automotive Components": "Automotive components",
    "Coal, Oil & Gas": "Coal, oil & gas",
    "Space & Defense": "Space & defence",
}

# ──────────────────────────────────────────────────────────────────────────────
# SCORING / EDA loaders (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_world_bank() -> pd.DataFrame:
    url = gh_raw_url(FILES["wb"])
    df = pd.read_csv(url)
    country = find_col(df.columns, "country", "country_name", "country name")
    year    = find_col(df.columns, "year")
    cont    = find_col(df.columns, "continent", "region")
    score   = find_col(df.columns, "score", "viability_score", "composite_score")
    grade   = find_col(df.columns, "grade", "letter_grade")
    if country is None or year is None or cont is None:
        raise ValueError("World Bank CSV missing required columns.")

    df = df.rename(columns={
        country: "country", year: "year", cont: "continent",
        **({score: "score"} if score else {}), **({grade: "grade"} if grade else {}),
    })
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "score" not in df.columns: df["score"] = np.nan
    if "grade" not in df.columns: df["grade"] = np.nan
    df["country"]   = df["country"].astype(str).str.strip()
    df["continent"] = df["continent"].astype(str).str.strip()

    order = ["A+", "A", "B", "C", "D"]
    df["grade"] = (df["grade"].astype(str).str.strip()
                   .where(lambda s: s.isin(order)))
    df["grade"] = pd.Categorical(df["grade"], categories=order, ordered=True)
    return df

def _melt_capex_wide(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    src = find_col(cols, "Source Country", "Source Co", "Country", "country")
    if not src:
        raise ValueError("CAPEX: 'Source Country' column not found.")
    year_cols = [c for c in cols if str(c).isdigit() and len(str(c)) == 4]
    id_vars = [src]
    grade_col = find_col(cols, "Grade")
    if grade_col: id_vars.append(grade_col)

    melted = df.melt(
        id_vars=id_vars, value_vars=year_cols, var_name="year", value_name="capex"
    ).rename(columns={src: "country", grade_col if grade_col else "": "grade"})
    melted["year"] = pd.to_numeric(melted["year"], errors="coerce").astype("Int64")

    def numify(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
        s = str(x).replace(",", "").strip()
        try: return float(s)
        except Exception: return np.nan

    melted["capex"] = melted["capex"].map(numify)
    melted["country"] = melted["country"].astype(str).str.strip()
    if "grade" in melted.columns:
        order = ["A+", "A", "B", "C", "D"]
        melted["grade"] = (melted["grade"].astype(str).str.strip()
                           .where(lambda s: s.isin(order)))
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
            pass
    url = gh_raw_url(FILES["cap_xlsx"])
    df_x = pd.read_excel(url, sheet_name=0)
    return _melt_capex_wide(df_x)

# Load Scoring/EDA sources
wb   = load_world_bank()
capx = load_capex_long()
wb_year_cc = wb[["year", "country", "continent"]].dropna()
capx_enriched = capx.merge(wb_year_cc, on=["year", "country"], how="left")

years_wb  = sorted(wb["year"].dropna().astype(int).unique().tolist())
years_cap = sorted(capx_enriched["year"].dropna().astype(int).unique().tolist())
years_all = ["All"] + sorted(set(years_wb).union(years_cap))

# ──────────────────────────────────────────────────────────────────────────────
# Aggregated (Sectors/Destinations) loader & helpers — robust
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_agg_raw(file_key: str) -> pd.DataFrame:
    """
    Reads merged_sectors_data.csv or merged_destinations_data.csv and returns a
    normalized dataframe with columns:
      country, sector, companies, projects, capex, jobs
    """
    fname = FILES[file_key]
    try:
        df_raw = pd.read_csv(gh_raw_url(fname))
    except Exception:
        df_raw = pd.read_csv(f"/mnt/data/{fname}")

    cols = list(df_raw.columns)

    # liberal aliases for safety
    country_col = find_col(
        cols,
        "source country", "country", "home country", "origin country",
        "destination country", "host country", "target country"
    )
    sector_col = find_col(
        cols, "sector", "destination", "destination sector", "sector name",
        "industry", "target sector", "category", "segment", "function"
    )
    companies_col = find_col(cols, "companies", "number of companies", "num companies",
                             "company count", "no. of companies")
    projects_col  = find_col(cols, "projects", "number of projects", "num projects",
                             "project count", "no. of projects")
    capex_col     = find_col(cols, "capex", "capital expenditure", "capex (in million usd)",
                             "capex_usd", "capex (usd)", "capex ($m)", "capex (m usd)")
    jobs_col      = find_col(cols, "jobs created", "jobs", "jobs_created", "new jobs",
                             "job creation", "jobs (estimated)")

    needs = {"country": country_col, "sector/destination": sector_col,
             "companies": companies_col, "projects": projects_col,
             "capex": capex_col, "jobs": jobs_col}
    missing = [k for k, v in needs.items() if v is None]
    if missing:
        raise ValueError(f"{file_key} CSV missing column for {missing}. Found: {list(df_raw.columns)}")

    out = pd.DataFrame({
        "country": df_raw[country_col].astype(str).str.strip(),
        "sector":  df_raw[sector_col].astype(str).str.strip().map(lambda x: SECTOR_NAME_MAP.get(x, x)),
        "companies": pd.to_numeric(df_raw[companies_col], errors="coerce"),
        "projects":  pd.to_numeric(df_raw[projects_col], errors="coerce"),
        "capex":     pd.to_numeric(df_raw[capex_col], errors="coerce"),
        "jobs":      pd.to_numeric(df_raw[jobs_col], errors="coerce"),
    }).fillna(0)

    return out

sectors_df       = load_agg_raw("sectors")
destinations_df  = load_agg_raw("destinations")

def available_countries_for(d: pd.DataFrame):
    # EXACTLY the countries in the provided dataframe (no 'All')
    return sorted(d["country"].dropna().unique().tolist())

def filter_selected_sectors(d: pd.DataFrame) -> pd.DataFrame:
    z = (d.groupby("sector", as_index=False)[["companies","projects","capex","jobs"]]
         .sum())
    z["sector"] = z["sector"].map(lambda s: SECTOR_NAME_MAP.get(s, s))
    cat = pd.Categorical(z["sector"], categories=SELECTED_SECTORS, ordered=True)
    z = z[cat.notna()].copy()
    z["sector"] = z["sector"].astype("category").cat.set_categories(SELECTED_SECTORS, ordered=True)
    z = z.sort_values("sector")
    return z

def bar_for_metric(df_in: pd.DataFrame, metric: str, title: str, height=460):
    fig = px.bar(df_in.sort_values(metric, ascending=True),
                 x=metric, y="sector", orientation="h",
                 color=metric, color_continuous_scale="Blues",
                 labels={metric: "", "sector": ""}, title=title, height=height)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

def kpi_centered(value, title, unit=""):
    if pd.isna(value): value, unit = "-", ""
    elif isinstance(value, (int,float,np.integer,np.floating)): value = f"{value:,.0f}"
    st.markdown(f"<div style='text-align:center' class='kpi-big'>{value}{unit}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center' class='kpi-sub'>{title}</div>", unsafe_allow_html=True)

def country_download_button(df_src: pd.DataFrame, country: str, filename: str, label: str):
    d = df_src[df_src["country"] == country].copy()
    if d.empty:
        st.info("No rows for this country.")
        return
    st.download_button(label=label,
                       file_name=filename, mime="text/csv",
                       data=d.to_csv(index=False).encode("utf-8"))

# ──────────────────────────────────────────────────────────────────────────────
# Global selectors (for Scoring/EDA; unchanged)
# ──────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1,1,2], gap="small")
with c1:
    sel_year_any = st.selectbox("Year", years_all, index=0, key="year_any")
with c2:
    valid_year_for_wb = sel_year_any if (isinstance(sel_year_any, int) and sel_year_any in years_wb) else max(years_wb)
    cont_options = ["All"] + sorted(wb.loc[wb["year"] == valid_year_for_wb, "continent"].dropna().unique().tolist())
    sel_cont = st.selectbox("Continent", cont_options, index=0, key="continent")
with c3:
    wb_scope = wb[wb["year"] == valid_year_for_wb].copy()
    if sel_cont != "All":
        wb_scope = wb_scope[wb_scope["continent"] == sel_cont]
    country_options = ["All"] + sorted(wb_scope["country"].unique().tolist())
    sel_country = st.selectbox("Country", country_options, index=0, key="country")

def filt_wb_single_year(df: pd.DataFrame, year_any) -> tuple[pd.DataFrame, int]:
    yy = int(year_any) if (isinstance(year_any, int) and year_any in years_wb) else max(years_wb)
    out = df[df["year"] == yy].copy()
    if sel_cont != "All": out = out[out["continent"] == sel_cont]
    if sel_country != "All": out = out[out["country"] == sel_country]
    return out, yy

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_scoring, tab_eda, tab_sectors, tab_dest = st.tabs(["Scoring", "EDA", "Sectors", "Destinations"])

# =============================================================================
# SCORING TAB (unchanged)
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
        with k1: st.metric("Country Score", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
        with k2: st.metric("Grade", country_grade)
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
        if not map_df.empty:
            fig_map = px.choropleth(map_df, locations="country", locationmode="country names",
                                    color="score", color_continuous_scale="Blues",
                                    title=f"Global Performance Map — {scoring_year}")
            fig_map.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(sel_cont, "world")
            fig_map.update_geos(scope=current_scope, projection_type="natural earth",
                                showcountries=True, showcoastlines=True)
            if sel_cont != "All" or sel_country != "All": fig_map.update_geos(fitbounds="locations")
            fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No data for this selection.")

    if sel_country == "All":
        b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")
        with b1:
            top10 = wb_year_df[["country", "score"]].dropna().sort_values("score", ascending=False).head(10)
            if not top10.empty:
                fig_top = px.bar(top10.sort_values("score"), x="score", y="country", orientation="h",
                                 color="score", color_continuous_scale="Blues",
                                 labels={"score": "", "country": ""},
                                 title=f"Top 10 Performing Countries — {scoring_year}")
                fig_top.update_coloraxes(showscale=False)
                fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_top, use_container_width=True)
        with b2:
            donut_base = wb_year_df.copy()
            if not donut_base.empty and donut_base["grade"].notna().any():
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
            cont_bar = (wb[wb["year"] == scoring_year]
                        .groupby("continent", as_index=False)["score"].mean()
                        .sort_values("score", ascending=True))
            if not cont_bar.empty:
                fig_cont = px.bar(cont_bar, x="score", y="continent", orientation="h",
                                  color="score", color_continuous_scale="Blues",
                                  labels={"score": "", "continent": ""},
                                  title=f"Continent Viability Score — {scoring_year}")
                fig_cont.update_coloraxes(showscale=False)
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig_cont, use_container_width=True)

# =============================================================================
# EDA TAB (unchanged)
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
    sel_grade_eda = st.selectbox("Grade (EDA)", grade_options,
                                 index=grade_options.index(auto_grade if auto_grade in grade_options else "All"),
                                 key="grade_eda")

    capx_eda = capx_enriched.copy()
    if sel_cont != "All":    capx_eda = capx_eda[capx_eda["continent"] == sel_cont]
    if sel_country != "All": capx_eda = capx_eda[capx_eda["country"] == sel_country]
    if sel_grade_eda != "All" and "grade" in capx_eda.columns:
        capx_eda = capx_eda[capx_eda["grade"] == sel_grade_eda]
    if isinstance(sel_year_any, int):
        capx_eda = capx_eda[capx_eda["year"] == sel_year_any]

    e1, e2 = st.columns([1.6, 2], gap="large")
    with e1:
        trend = capx_eda.groupby("year", as_index=False)["capex"].sum().sort_values("year")
        if trend.empty:
            st.info("No CAPEX data for the selected filters.")
        else:
            trend["year_str"] = trend["year"].astype(int).astype(str)
            title = f"{sel_country} CAPEX Trend" if sel_country != "All" else "Global CAPEX Trend"
            fig = px.line(trend, x="year_str", y="capex", markers=True,
                          labels={"year_str": "", "capex": "Global CAPEX ($B)"},
                          title=title)
            fig.update_xaxes(type="category", categoryorder="array",
                             categoryarray=trend["year_str"].tolist(), showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

    with e2:
        if isinstance(sel_year_any, int):
            map_df = capx_eda.copy(); map_title = f"CAPEX Map — {sel_year_any}"
        else:
            map_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
            map_title = "CAPEX Map — All Years (aggregated)"
        if map_df.empty:
            st.info("No CAPEX data for this selection.")
        else:
            fig = px.choropleth(map_df, locations="country", locationmode="country names",
                                color="capex", color_continuous_scale="Blues", title=map_title)
            fig.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(sel_cont, "world")
            fig.update_geos(scope=current_scope, projection_type="natural earth",
                            showcountries=True, showcoastlines=True)
            if sel_cont != "All" or sel_country != "All": fig.update_geos(fitbounds="locations")
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)

    show_grade_trend = (sel_grade_eda == "All")
    if show_grade_trend:
        b1, b2, b3 = st.columns([1.2, 1.2, 1.6], gap="large")
    else:
        b1, b3 = st.columns([1.2, 1.6], gap="large")

    with b1:
        if isinstance(sel_year_any, int):
            level_df = capx_eda.copy(); title_top10 = f"Top 10 Countries by CAPEX — {sel_year_any}"
        else:
            level_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
            title_top10 = "Top 10 Countries by CAPEX — All Years (aggregated)"
        top10 = level_df.dropna(subset=["capex"]).sort_values("capex", ascending=False).head(10)
        if not top10.empty:
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
                if not tg.empty:
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
        if not growth_base.empty:
            agg = growth_base.groupby(["country", "year"], as_index=False)["capex"].sum()
            first_year = int(agg["year"].min()) if not agg.empty else None
            last_year  = int(agg["year"].max()) if not agg.empty else None
            if first_year is not None and last_year is not None and first_year != last_year:
                start = agg[agg["year"] == first_year][["country", "capex"]].rename(columns={"capex": "capex_start"})
                end   = agg[agg["year"] == last_year][["country", "capex"]].rename(columns={"capex": "capex_end"})
                joined = start.merge(end, on="country", how="inner")
                joined["growth_abs"] = joined["capex_end"] - joined["capex_start"]
                fig = px.bar(joined.sort_values("growth_abs").tail(10),
                             x="growth_abs", y="country", orientation="h",
                             color="growth_abs", color_continuous_scale="Blues",
                             labels={"growth_abs": "", "country": ""},
                             title=f"Top 10 Countries by CAPEX Growth (All Grades) [{first_year} → {last_year}]")
                fig.update_coloraxes(showscale=False)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough years to compute growth.")
        else:
            st.info("No CAPEX data for growth ranking.")

# =============================================================================
# SECTORS TAB (unchanged; country filter = dataset countries only)
# =============================================================================
with tab_sectors:
    st.caption("Sectors Analysis")

    s1, s2 = st.columns([1, 3], gap="large")
    with s1:
        sector_opt = ["All"] + sorted(sectors_df["sector"].dropna().unique().tolist())
        sel_sector = st.selectbox("Sector", sector_opt, index=0, key="sec_sector")
    with s2:
        country_list = available_countries_for(sectors_df)  # EXACT list from dataset
        sel_country_sec = st.selectbox("Country (Sectors)", country_list,
                                       index=0 if country_list else None,
                                       key="sec_country")

    metric = st.radio("Metric", ["Companies", "Jobs Created", "Capex", "Projects"],
                      horizontal=True, key="sec_metric")
    metric_col = {"Companies":"companies", "Jobs Created":"jobs", "Capex":"capex", "Projects":"projects"}[metric]
    title_suffix = {"companies":"Number of Companies", "jobs":"Jobs Created",
                    "capex":"CAPEX ($M)", "projects":"Projects"}[metric_col]

    if sel_country_sec:
        country_download_button(sectors_df, sel_country_sec,
                                filename=f"{sel_country_sec.lower().replace(' ','_')}_sectors_data.csv",
                                label=f"Download CSV for {sel_country_sec}")

    if sel_sector != "All" and sel_country_sec:
        v = (sectors_df[(sectors_df["country"] == sel_country_sec) &
                        (sectors_df["sector"] == sel_sector)][metric_col]
             .sum())
        kpi_centered(v, f"{sel_country_sec} — {sel_sector} • {metric}")
    else:
        df_view = sectors_df.copy()
        if sel_country_sec:
            df_view = df_view[df_view["country"] == sel_country_sec]
        df_view = filter_selected_sectors(df_view)
        bar_for_metric(df_view, metric_col,
                       f"{title_suffix} by Selected Sectors — {sel_country_sec if sel_country_sec else 'Global'}")

# =============================================================================
# DESTINATIONS TAB — EXACT same behavior as Sectors (country filter = dataset)
# =============================================================================
with tab_dest:
    st.caption("Destinations Analysis")

    d1, d2 = st.columns([1, 3], gap="large")
    with d1:
        dest_opt = ["All"] + sorted(destinations_df["sector"].dropna().unique().tolist())
        sel_dest = st.selectbox("Sector", dest_opt, index=0, key="dest_sector")
    with d2:
        country_list_d = available_countries_for(destinations_df)  # EXACT list from merged_destinations_data.csv
        sel_country_dest = st.selectbox("Country (Destinations)", country_list_d,
                                        index=0 if country_list_d else None,
                                        key="dest_country")

    metric_d = st.radio("Metric", ["Companies", "Jobs Created", "Capex", "Projects"],
                        horizontal=True, key="dest_metric")
    metric_col_d = {"Companies":"companies", "Jobs Created":"jobs", "Capex":"capex", "Projects":"projects"}[metric_d]
    title_suffix_d = {"companies":"Number of Companies", "jobs":"Jobs Created",
                      "capex":"CAPEX ($M)", "projects":"Projects"}[metric_col_d]

    if sel_country_dest:
        country_download_button(destinations_df, sel_country_dest,
                                filename=f"{sel_country_dest.lower().replace(' ','_')}_destinations_data.csv",
                                label=f"Download CSV for {sel_country_dest}")

    if sel_dest != "All" and sel_country_dest:
        v = (destinations_df[(destinations_df["country"] == sel_country_dest) &
                             (destinations_df["sector"] == sel_dest)][metric_col_d]
             .sum())
        kpi_centered(v, f"{sel_country_dest} — {sel_dest} • {metric_d}")
    else:
        df_view_d = destinations_df.copy()
        if sel_country_dest:
            df_view_d = df_view_d[df_view_d["country"] == sel_country_dest]
        df_view_d = filter_selected_sectors(df_view_d)
        bar_for_metric(df_view_d, metric_col_d,
                       f"{title_suffix_d} by Selected Sectors — {sel_country_dest if sel_country_dest else 'Global'}")

# ──────────────────────────────────────────────────────────────────────────────
# Indicator weights table (Scoring tab only — unchanged)
# ──────────────────────────────────────────────────────────────────────────────
with tab_scoring:
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
