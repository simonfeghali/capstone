# app.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from urllib.parse import quote
from urllib.error import URLError, HTTPError

st.set_page_config(page_title="FDI Analytics", layout="wide")
st.title("FDI Analytics Dashboard")

# ──────────────────────────────────────────────────────────────────────────────
# Small style helpers
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .kpi-wrap {display:flex;align-items:center;justify-content:center;height:260px;}
      .kpi-num  {font-size:72px; font-weight:800; letter-spacing:1px;}
      .kpi-unit {font-size:18px; font-weight:600; margin-left:6px; opacity:.8;}
      .title-left {font-weight:700; font-size:18px; margin:12px 0 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Remote data (Scoring/EDA) — unchanged
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    "cap_csv": "capex_EDA_cleaned_filled.csv",
    "cap_csv_alt": "capex_EDA_cleaned_filled.csv",
    "cap_xlsx": "capex_EDA.xlsx",
}
def gh_raw_url(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def _find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    for c in cands:
        for col in cols:
            if c.lower() in col.lower(): return col
    return None

@st.cache_data(show_spinner=True)
def load_world_bank() -> pd.DataFrame:
    url = gh_raw_url(FILES["wb"])
    df = pd.read_csv(url)

    country = _find_col(df.columns, "country", "country_name", "Country Name")
    year    = _find_col(df.columns, "year")
    cont    = _find_col(df.columns, "continent", "region")
    score   = _find_col(df.columns, "score", "viability_score", "composite_score")
    grade   = _find_col(df.columns, "grade", "letter_grade")

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
    df["country"]   = df["country"].astype(str).str.strip()
    df["continent"] = df["continent"].astype(str).str.strip()
    if "score" not in df.columns: df["score"] = np.nan
    if "grade" not in df.columns: df["grade"] = np.nan

    order = ["A+", "A", "B", "C", "D"]
    df["grade"] = df["grade"].astype(str).str.strip()
    df.loc[~df["grade"].isin(order), "grade"] = np.nan
    df["grade"] = pd.Categorical(df["grade"], categories=order, ordered=True)
    return df

def _melt_capex_wide(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    src = _find_col(df.columns, "Source Country", "Source Co", "Country")
    if not src:
        raise ValueError("CAPEX: 'Source Country' column not found.")
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    id_vars = [src]
    grade_col = _find_col(df.columns, "Grade")
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
            df = pd.read_csv(gh_raw_url(FILES[key]))
            return _melt_capex_wide(df)
        except Exception:
            continue
    df_x = pd.read_excel(gh_raw_url(FILES["cap_xlsx"]), sheet_name=0)
    return _melt_capex_wide(df_x)

wb   = load_world_bank()
capx = load_capex_long()
wb_for_merge = wb[["year", "country", "continent"]].dropna()
capx_enriched = capx.merge(wb_for_merge, on=["year", "country"], how="left")

# ──────────────────────────────────────────────────────────────────────────────
# Global filters (used by Scoring/EDA) — unchanged
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
    rows = wb[(wb["country"] == prev_country)] if sel_year_any == "All" else wb[(wb["year"] == sel_year_any) & (wb["country"] == prev_country)]
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

def filt_wb_year(df, year_any):
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
# SCORING (unchanged)
# =============================================================================
with tab_scoring:
    st.caption("Scoring • (World Bank–based)")
    where_title = sel_country if sel_country != "All" else (sel_cont if sel_cont != "All" else "Worldwide")
    st.subheader(where_title)

    wb_year_df, scoring_year = filt_wb_year(wb, sel_year_any)

    if sel_country != "All":
        rows = wb[(wb["year"] == scoring_year) & (wb["country"] == sel_country)]
        country_score = float(rows["score"].mean()) if not rows.empty else np.nan
        country_grade = rows["grade"].astype(str).dropna().iloc[0] if not rows.empty and rows["grade"].notna().any() else "-"
        ctry_cont = rows["continent"].dropna().iloc[0] if not rows.empty and rows["continent"].notna().any() else None
        cont_avg = float(wb[(wb["year"] == scoring_year) & (wb["continent"] == ctry_cont)]["score"].mean()) if ctry_cont else np.nan

        c1_, c2_, c3_ = st.columns(3, gap="large")
        c1_.metric("Country Score", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
        c2_.metric("Grade", country_grade)
        c3_.metric(f"{ctry_cont} Avg Score" if ctry_cont else "Continent Avg Score", "-" if np.isnan(cont_avg) else f"{cont_avg:,.3f}")

    t1, t2 = st.columns([1, 2], gap="large")
    with t1:
        base = wb.copy()
        title = "Year-over-Year Viability Score — Global"
        if sel_country != "All":
            base = base[base["country"] == sel_country]
            title = f"Year-over-Year Viability Score — {sel_country}"
        elif sel_cont != "All":
            base = base[base["continent"] == sel_cont]
            title = f"Year-over-Year Viability Score — {sel_cont}"

        yoy_df = base.groupby("year", as_index=False)["score"].mean().sort_values("year")
        yoy_df["year_str"] = yoy_df["year"].astype(int).astype(str)
        fig = px.line(yoy_df, x="year_str", y="score", markers=True,
                      labels={"year_str": "", "score": "Mean score"}, title=title)
        fig.update_xaxes(type="category", categoryorder="array",
                         categoryarray=yoy_df["year_str"].tolist(), showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=340)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        map_df = wb_year_df[["country", "score"]].copy()
        if map_df.empty:
            st.info("No data for this selection.")
        else:
            fig = px.choropleth(map_df, locations="country", locationmode="country names",
                                color="score", color_continuous_scale="Blues",
                                title=f"Global Performance Map — {scoring_year}")
            fig.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            fig.update_geos(scope=scope_map.get(sel_cont, "world"), projection_type="natural earth",
                            showcountries=True, showcoastlines=True)
            if sel_cont != "All" or sel_country != "All": fig.update_geos(fitbounds="locations")
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=410)
            st.plotly_chart(fig, use_container_width=True)

    if sel_country == "All":
        b1,b2,b3 = st.columns([1.2,1,1.2], gap="large")
        with b1:
            top10 = wb_year_df[["country","score"]].dropna().sort_values("score", ascending=False).head(10)
            if top10.empty: st.info("No countries available for Top 10 with this filter.")
            else:
                fig = px.bar(top10.sort_values("score"), x="score", y="country", orientation="h",
                             color="score", color_continuous_scale="Blues",
                             labels={"score":"", "country":""},
                             title=f"Top 10 Performing Countries — {scoring_year}")
                fig.update_coloraxes(showscale=False)
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)
        with b2:
            base = wb_year_df.copy()
            if base.empty or base["grade"].isna().all(): st.info("No grade data for this selection.")
            else:
                grades = ["A+","A","B","C","D"]
                donut = (base.assign(grade=base["grade"].astype(str))
                              .loc[lambda d: d["grade"].isin(grades)]
                              .groupby("grade", as_index=False)["country"].nunique()
                              .rename(columns={"country":"count"})
                        ).set_index("grade").reindex(grades, fill_value=0).reset_index()
                shades = [px.colors.sequential.Blues[-1-i] for i in range(5)]
                cmap = {g:c for g,c in zip(grades, shades)}
                fig = px.pie(donut, names="grade", values="count", hole=0.55,
                             title=f"Grade Distribution — {scoring_year}",
                             color="grade", color_discrete_map=cmap)
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=420, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        with b3:
            base = wb[wb["year"] == scoring_year].copy()
            if sel_cont != "All": base = base[base["continent"] == sel_cont]
            cont_bar = base.groupby("continent", as_index=False)["score"].mean().sort_values("score", ascending=True)
            if cont_bar.empty: st.info("No continent data for this selection.")
            else:
                fig = px.bar(cont_bar, x="score", y="continent", orientation="h",
                             color="score", color_continuous_scale="Blues",
                             labels={"score":"","continent":""},
                             title=f"Continent Viability Score — {scoring_year}")
                fig.update_coloraxes(showscale=False)
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# EDA (unchanged)
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
        if trend.empty: st.info("No CAPEX data for the selected filters.")
        else:
            trend["year_str"] = trend["year"].astype(int).astype(str)
            fig = px.line(trend, x="year_str", y="capex", markers=True,
                          labels={"year_str": "", "capex": "Global CAPEX ($B)"},
                          title="Global CAPEX Trend")
            fig.update_xaxes(type="category", categoryorder="array",
                             categoryarray=trend["year_str"].tolist(), showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

    with e2:
        if isinstance(sel_year_any, int):
            map_df = capx_eda.copy(); map_title = f"CAPEX Map — {sel_year_any}"
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
            fig.update_geos(scope=scope_map.get(sel_cont, "world"), projection_type="natural earth",
                            showcountries=True, showcoastlines=True)
            if sel_cont != "All" or sel_country != "All": fig.update_geos(fitbounds="locations")
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=420)
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
        if top10.empty: st.info("No CAPEX data for Top 10 with this filter.")
        else:
            fig = px.bar(top10.sort_values("capex"), x="capex", y="country", orientation="h",
                         color="capex", color_continuous_scale="Blues",
                         labels={"capex": "", "country": ""}, title=title_top10)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=420)
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
                    fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=420)
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
                fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=420)
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Helpers for Sectors & Destinations (fail-safe file loading)
# =============================================================================
SELECTED_SECTORS_ORDER = [
    "Software & IT services","Business services","Communications","Financial services",
    "Food and Beverages","Transportation & Warehousing","Real estate","Consumer products",
    "Automotive OEM","Automotive components","Chemicals","Pharmaceuticals","Metals",
    "Coal, oil & gas","Space & defence","Leisure & entertainment"
]

def normalize_sector_name(s: str) -> str:
    if pd.isna(s): return np.nan
    t = str(s).strip()
    repl = {
        "Software and IT services": "Software & IT services",
        "IT & Software services": "Software & IT services",
        "Business Services": "Business services",
        "Food & Beverages": "Food and Beverages",
        "Transport & Warehousing": "Transportation & Warehousing",
        "Automotive Components": "Automotive components",
        "Automotive O.E.M.": "Automotive OEM",
        "Coal, Oil & Gas": "Coal, oil & gas",
        "Space and defence": "Space & defence",
        "Leisure and entertainment": "Leisure & entertainment",
    }
    return repl.get(t, t)

@st.cache_data(show_spinner=False)
def try_load_vertical_csv(path: str):
    """
    Return (df, error_message). If file is missing or invalid, df=None with msg.
    """
    p = Path(path)
    if not p.exists():
        return None, f"Dataset not found: {path}"

    try:
        raw = pd.read_csv(p)
    except Exception as e:
        return None, f"Failed to read {path}: {e}"

    cols = list(raw.columns)
    low = {c.lower(): c for c in cols}
    def need(name, *alts):
        for k in (name, *alts):
            if k in low: return low[k]
        for c in cols:
            if name in c.lower(): return c
        return None

    country = need("country")
    sector  = need("sector")
    comp    = need("companies","company")
    jobs    = need("jobs_created","jobs","jobs created")
    capex   = need("capex")
    projs   = need("projects","project_count","nb_projects")

    if not all([country, sector, comp, jobs, capex, projs]):
        return None, f"{path} is missing required columns. Found: {cols}"

    df = raw.rename(columns={
        country:"country", sector:"sector", comp:"companies",
        jobs:"jobs_created", capex:"capex", projs:"projects"
    }).copy()

    df["country"] = df["country"].astype(str).str.strip()
    df["sector"]  = df["sector"].astype(str).map(normalize_sector_name)
    for c in ["companies","jobs_created","capex","projects"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df[df["sector"].isin(SELECTED_SECTORS_ORDER)].copy()
    df["sector"] = pd.Categorical(df["sector"], categories=SELECTED_SECTORS_ORDER, ordered=True)
    return df, None

def sectors_bar_for_country(df: pd.DataFrame, country: str, metric: str, title_prefix: str):
    metric_labels = {
        "companies": "Number of Companies",
        "jobs_created": "Jobs Created",
        "capex": "CAPEX (in million USD)",
        "projects": "Number of Projects",
    }
    nice = metric_labels.get(metric, metric)
    base = df[df["country"] == country].groupby("sector", as_index=False)[metric].sum()
    base = base.set_index("sector").reindex(SELECTED_SECTORS_ORDER, fill_value=0).reset_index()
    fig = px.bar(
        base.sort_values(metric, ascending=True),
        x=metric, y="sector", orientation="h",
        labels={metric: "", "sector": ""},
        color=metric, color_continuous_scale="Blues",
        title=f"{nice} by Sector — {title_prefix}"
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=480)
    return fig

def kpi_card(value: float, unit: str):
    st.markdown(f'<div class="kpi-wrap"><div class="kpi-num">{value:,.0f}</div><div class="kpi-unit">{unit}</div></div>', unsafe_allow_html=True)

def download_country_csv_button(df: pd.DataFrame, country: str, label_prefix: str):
    export = (df[df["country"] == country]
              .groupby("sector", as_index=False)
              .agg({"companies":"sum","jobs_created":"sum","capex":"sum","projects":"sum"})
              .set_index("sector").reindex(SELECTED_SECTORS_ORDER, fill_value=0).reset_index())
    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download CSV for {country}",
        data=csv,
        file_name=f"{label_prefix}_{country.lower().replace(' ','_')}.csv",
        mime="text/csv",
        key=f"dl_{label_prefix}_{country}"
    )

def sector_dest_tab_ui(csv_path: str, who: str, key_prefix: str):
    df, err = try_load_vertical_csv(csv_path)
    if df is None:
        st.warning(err)
        return

    countries = sorted(df["country"].unique().tolist())
    left, right = st.columns([1, 3], gap="small")
    with left:
        sect_opt = ["All"] + SELECTED_SECTORS_ORDER
        sel_sector = st.selectbox("Sector", sect_opt, index=0, key=f"{key_prefix}_sector")
    with right:
        sel_country_local = st.selectbox(f"Country ({who})", countries, index=0, key=f"{key_prefix}_country")

    metric_map = {"Companies":"companies","Jobs Created":"jobs_created","Capex":"capex","Projects":"projects"}
    pretty = st.radio("Metric", list(metric_map.keys()), horizontal=True, key=f"{key_prefix}_metric")
    metric = metric_map[pretty]

    download_country_csv_button(df, sel_country_local, f"{who.lower()}_sectors_data")

    if sel_sector == "All":
        fig = sectors_bar_for_country(df, sel_country_local, metric, sel_country_local)
        st.plotly_chart(fig, use_container_width=True)
    else:
        row = df[(df["country"] == sel_country_local) & (df["sector"] == sel_sector)].agg({metric:"sum"})
        val = float(row.iloc[0]) if not row.empty else 0.0
        st.markdown(f'<div class="title-left">{sel_country_local} — {sel_sector} • {pretty}</div>', unsafe_allow_html=True)
        unit = {"companies":"", "jobs_created":"", "capex":"$M", "projects":""}[metric]
        kpi_card(val, unit)

# =============================================================================
# SECTORS (UNCHANGED features; now fail-safe if file is missing)
# =============================================================================
with tab_sectors:
    st.caption("Sectors Analysis")
    sector_dest_tab_ui("/mnt/data/merged_sectors_data.csv", "Sectors", "sec")

# =============================================================================
# DESTINATIONS (mirror of Sectors; separate dataset; now fail-safe)
# =============================================================================
with tab_dest:
    st.caption("Destinations Analysis")
    sector_dest_tab_ui("/mnt/data/merged_destinations_data.csv", "Destinations", "dest")
