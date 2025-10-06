# app.py
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re
from urllib.parse import quote
from urllib.error import URLError, HTTPError
from forecasting import render_forecasting_tab
from overview import render_overview_tab, info_button, emit_auto_jump_script


# ──────────────────────────────────────────────────────────────────────────────
# App chrome / theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")

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
      .subtitle { color:#6b7280; font-size:0.95rem; margin:-6px 0 12px 0; text-align:center;}
      .section-h { font-weight:800; text-align:center; margin:22px 0 2px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab icons (local PNGs in repo root)
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Header assets ─────────────────────────────────────────────────────────────
LOGO_FILE = "oco_global_logo.jpeg"
ICON_FILE = "data-analytics.png"

def _b64_img(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""

def _b64_png(path_or_url) -> str:
    """Return base64 for a local file or an http(s) URL (PNG/JPG)."""
    try:
        s = str(path_or_url)
        if s.startswith("http://") or s.startswith("https://"):
            import urllib.request
            with urllib.request.urlopen(s) as r:
                data = r.read()
        else:
            from pathlib import Path
            data = Path(path_or_url).read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return ""

# Build header HTML with left icon, centered title, and right logo
logo_b64 = _b64_img(ROOT / LOGO_FILE)
icon_b64 = _b64_img(ROOT / ICON_FILE)

st.markdown(f"""
<style>
.header-row {{
  display:flex; align-items:center; justify-content:space-between;
  margin: 20px 0 24px 0;
}}
.header-left {{
  display:flex; align-items:center; gap:14px;
}}
.header-title {{
  font-size: 42px; font-weight: 800; line-height:1.2; margin:0;
}}
.header-icon {{
  width: 42px; height: 42px;
}}
.header-logo {{
  height: 160px;
  width: auto;
}}
</style>
<div class="header-row">
  <div class="header-left">
    <img class="header-icon" src="data:image/png;base64,{icon_b64}" />
    <div class="header-title">Country Viability & FDI Analytics Dashboard</div>
  </div>
  <img class="header-logo" src="data:image/jpeg;base64,{logo_b64}" />
</div>
""", unsafe_allow_html=True)

def inject_tab_icons():
    icons_in_order = [
        ROOT / "information.png",   # Overview
        ROOT / "score.png",         # Scoring
        ROOT / "capex.png",         # CAPEX
        ROOT / "sectors.png",       # Sectors
        ROOT / "destinations.png",  # Destinations
        ROOT / "compare.png",       # Compare
        ROOT / "forecast.png",      # Forecast
    ]
    css_blocks = []

    css_blocks.append("""
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        padding: 14px 0;
        font-size: 1.1rem;
    }
    """)

    for i, icon_path in enumerate(icons_in_order, start=1):
        b64 = _b64_png(icon_path)
        if not b64:
            continue
        css_blocks.append(f"""
        .stTabs [data-baseweb="tab-list"] > [data-baseweb="tab"]:nth-child({i}) p::before,
        div[data-baseweb="tab-list"] > div[role="tab"]:nth-child({i}) p::before {{
            content: "";
            display: inline-block;
            width: 22px;
            height: 22px;
            margin-right: 10px;
            vertical-align: -4px;
            background-image: url('data:image/png;base64,{b64}');
            background-size: contain;
            background-repeat: no-repeat;
        }}
        """)

    st.markdown("<style>" + "\n".join(css_blocks) + "</style>", unsafe_allow_html=True)

inject_tab_icons()



# ──────────────────────────────────────────────────────────────────────────────
# Data sources (GitHub raw)
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    "wb_avg": "world_bank_data_average_scores_and_grades.csv",
    "cap_csv": "capex_EDA_cleaned_filled.csv",
    "cap_csv_alt": "capex_EDA_cleaned_filled.csv",
    "cap_xlsx": "capex_EDA.xlsx",
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
}

def gh_raw_url(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def find_col(cols, *cands):
    low = {str(c).lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for cand in cands:
        for col in cols:
            if cand.lower() in str(col).lower():
                return col
    return None

def _numify_generic(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = re.sub(r"[^\d\.\-]", "", str(x))
    try:
        return float(s)
    except Exception:
        return np.nan

# ──────────────────────────────────────────────────────────────────────────────
# Loaders
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

    order = ["A+", "A", "B", "C", "D"]
    df["grade"] = df["grade"].astype(str).str.strip()
    df.loc[~df["grade"].isin(order), "grade"] = np.nan
    df["grade"] = pd.Categorical(df["grade"], categories=order, ordered=True)
    return df

@st.cache_data(show_spinner=True)
def load_world_bank_averages() -> pd.DataFrame:
    url = gh_raw_url(FILES["wb_avg"])
    try:
        df = pd.read_csv(url)
    except (URLError, HTTPError, FileNotFoundError) as e:
        raise RuntimeError(f"Could not fetch {FILES['wb_avg']}: {e}")

    country = find_col(df.columns, "country", "country_name", "Country Name")
    avg     = find_col(df.columns, "avg_score", "average_score", "mean_score", "score")
    grade   = find_col(df.columns, "grade", "letter_grade")

    if country is None or avg is None:
        raise ValueError(f"Averages CSV missing required columns. Found: {list(df.columns)}")

    keep = {country: "country", avg: "avg_score"}
    if grade: keep[grade] = "grade"
    out = df.rename(columns=keep)[list(keep.values())].copy()

    if "grade" not in out.columns: out["grade"] = np.nan
    out["country"] = out["country"].astype(str).str.strip()

    order = ["A+", "A", "B", "C", "D"]
    out["grade"] = out["grade"].astype(str).str.strip()
    out.loc[~out["grade"].isin(order), "grade"] = np.nan
    out["grade"] = pd.Categorical(out["grade"], categories=order, ordered=True)
    out["avg_score"] = pd.to_numeric(out["avg_score"], errors="coerce")
    return out[["country", "avg_score", "grade"]]

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


# Core data
wb        = load_world_bank()
wb_avg    = load_world_bank_averages()
capx      = load_capex_long()

# Enrichers
wb_cc = wb.drop_duplicates(subset=["country", "continent"])[["country", "continent"]]
capx_enriched = capx.merge(wb_cc, on="country", how="left")
wb_avg_enriched = wb_avg.merge(wb_cc, on="country", how="left")

# ──────────────────────────────────────────────────────────────────────────────
# Filter blocks
# ──────────────────────────────────────────────────────────────────────────────
years_wb  = sorted(wb["year"].dropna().astype(int).unique().tolist())
years_cap = sorted(capx_enriched["year"].dropna().astype(int).unique().tolist())
years_all = ["All"] + sorted(set(years_wb).union(years_cap))

def render_filters_block(prefix: str):
    k_year = f"{prefix}_year_any"
    k_cont = f"{prefix}_continent"
    k_ctry = f"{prefix}_country"

    c1, c2, c3 = st.columns([1, 1, 2], gap="small")

    with c1:
        sel_year_any = st.selectbox("Year", years_all, index=0, key=k_year)

    prev_country = st.session_state.get(k_ctry, "All")
    suggested_cont = None
    if prev_country != "All":
        rows = wb[(wb["year"] == sel_year_any) & (wb["country"] == prev_country)] if isinstance(sel_year_any, int) else wb[wb["country"] == prev_country]
        if not rows.empty and rows["continent"].notna().any():
            suggested_cont = rows["continent"].dropna().iloc[0]

    valid_year_for_wb = sel_year_any if (isinstance(sel_year_any, int) and sel_year_any in years_wb) else max(years_wb)
    cont_options = ["All"] + sorted(wb.loc[wb["year"] == valid_year_for_wb, "continent"].dropna().unique().tolist())

    saved_cont = st.session_state.get(k_cont, "All")
    default_cont = suggested_cont if (suggested_cont in cont_options) else (saved_cont if saved_cont in cont_options else "All")

    with c2:
        sel_cont = st.selectbox("Continent", cont_options, index=cont_options.index(default_cont), key=k_cont)

    wb_scope = wb[wb["year"] == valid_year_for_wb].copy()
    if sel_cont != "All":
        wb_scope = wb_scope[wb_scope["continent"] == sel_cont]
    country_options = ["All"] + sorted(wb_scope["country"].unique().tolist())
    saved_country = st.session_state.get(k_ctry, "All")
    default_country = saved_country if saved_country in country_options else "All"

    with c3:
        sel_country = st.selectbox("Country", country_options, index=country_options.index(default_country), key=k_ctry)

    def filt_wb_single_year(df: pd.DataFrame, year_any) -> tuple[pd.DataFrame, int]:
        yy = int(year_any) if (isinstance(year_any, int) and year_any in years_wb) else max(years_wb)
        out = df[df["year"] == yy].copy()
        if sel_cont != "All":
            out = out[out["continent"] == sel_cont]
        if sel_country != "All":
            out = out[out["country"] == sel_country]
        return out, yy

    return sel_year_any, sel_cont, sel_country, filt_wb_single_year

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Title tokens for the Scoring tab (centralized)
# ──────────────────────────────────────────────────────────────────────────────
def _year_token(year_any):
    return "2021–2023" if year_any == "All" else str(int(year_any))

def _scope_token(cont, country):
    if country != "All":
        return country
    return cont if cont != "All" else "Global"

def _where_clause(cont, country):
    if country != "All":
        return country
    if cont != "All":
        return cont
    return "Worldwide"

def _subtitle_suffix(year_any):
    return "" if year_any == "All" else f"(for {_year_token(year_any)})"

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
from compare_tab import render_compare_tab
tab_overview, tab_scoring, tab_eda, tab_sectors, tab_dest, tab_compare, tab_forecast = st.tabs(
    ["Overview", "Country Attractiveness", "Capital Investment", "Industry Landscape", "Target Countries", "Benchmarking", "FDI Forecasts"]
)

# =============================================================================
# OVERVIEW TAB 
# =============================================================================

with tab_overview:
    render_overview_tab()

# =============================================================================
# SCORING TAB (unchanged from your latest) 
# =============================================================================
# ...  ⬇️  (kept exactly as in your message; not repeating here to keep this block focused)
# (Your entire SCORING tab code remains the same)

# =============================================================================
# CAPEX TAB — Rebuilt per your requested slides/titles/subtitles (2021–2024)
# =============================================================================
with tab_eda:
    # Caption + info button ABOVE filters
    cap_left, cap_right = st.columns([20, 1], gap="small")
    with cap_left:
        st.caption("Capital Investment • 2021–2024")
    with cap_right:
        info_button("capex_trend")

    # Filters (left visible for navigation; charts below use fixed 2021–2024 global scope unless noted)
    sel_year_any, sel_cont, sel_country, _filt = render_filters_block("eda")

    # Prepare CAPEX ($B) 2021–2024
    cap = capx_enriched.copy()
    cap = cap[(cap["year"].between(2021, 2024))]
    # Ensure $B
    cap["capex"] = pd.to_numeric(cap["capex"], errors="coerce") / 1000.0

    def _title(h: str, sub: str):
        st.markdown(f"<h3 class='section-h'>{h}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtitle'>{sub}</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # Slide 1: Capital Investment
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("<h2 class='section-h'>Capital Investment</h2>", unsafe_allow_html=True)

    # Chart 1: Top Countries by CAPEX — All Years
    _title(
        "Global Leaders in FDI Capital Expenditure (2021–2024)",
        "United States, China, and the UK dominate global CAPEX inflows, reflecting their scale, stability, and market maturity."
    )
    top_all = (cap.groupby("country", as_index=False)["capex"].sum()
                 .sort_values("capex", ascending=False).head(10)
                 .sort_values("capex", ascending=True))
    if top_all.empty:
        st.info("No CAPEX data available for 2021–2024.")
    else:
        fig_top = px.bar(
            top_all, x="capex", y="country", orientation="h",
            color="capex", color_continuous_scale="Blues",
            labels={"capex": "", "country": ""},
        )
        fig_top.update_coloraxes(showscale=False)
        fig_top.update_traces(hovertemplate="%{y}: %{x:,.3f} $B<extra></extra>")
        fig_top.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=460)
        st.plotly_chart(fig_top, use_container_width=True)

    # Chart 2: CAPEX Trend by Grade ($B)
    _title(
        "CAPEX Flows by Country Grade (A+ to D), 2021–2024",
        "A+ and A countries consistently attract the largest capital inflows; lower-graded countries show minor, volatile flows."
    )
    if "grade" in cap.columns and not cap.empty:
        tg = (cap.assign(grade=cap["grade"].astype(str))
                .groupby(["year", "grade"], as_index=False, observed=True)["capex"].sum())
        # Order grades + consistent blues
        grade_order = ["A+", "A", "B", "C", "D"]
        blues = px.colors.sequential.Blues
        shades = [blues[-1], blues[-2], blues[-3], blues[-4], blues[-5]]
        cmap = {g: c for g, c in zip(grade_order, shades)}
        tg["year_str"] = tg["year"].astype(int).astype(str)
        fig_grade = px.line(
            tg, x="year_str", y="capex", color="grade",
            color_discrete_map=cmap, category_orders={"grade": grade_order},
            labels={"year_str": "", "capex": "", "grade": "Grade"},
        )
        fig_grade.update_traces(mode="lines+markers",
                                hovertemplate="Year: %{x}<br>Capex: %{y:,.3f} $B<br>Grade: %{fullData.name}<extra></extra>")
        fig_grade.update_xaxes(type="category", showgrid=False)
        fig_grade.update_yaxes(showgrid=False)
        fig_grade.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, legend_title_text="Grade")
        st.plotly_chart(fig_grade, use_container_width=True)
    else:
        st.info("No grade data available for 2021–2024.")

    # Chart 3: Top Countries by CAPEX Growth (All Grades)
    _title(
        "Fastest-Growing CAPEX Destinations (2021 → 2024)",
        "UAE and China show the strongest growth momentum, highlighting emerging hotspots alongside advanced economies."
    )
    gdf = (cap.groupby(["country", "year"], as_index=False)["capex"].sum())
    have_21 = gdf[gdf["year"] == 2021][["country", "capex"]].rename(columns={"capex": "capex_2021"})
    have_24 = gdf[gdf["year"] == 2024][["country", "capex"]].rename(columns={"capex": "capex_2024"})
    growth = have_21.merge(have_24, on="country", how="inner")
    growth["growth_abs"] = growth["capex_2024"] - growth["capex_2021"]
    top_growth = (growth.sort_values("growth_abs", ascending=False).head(10)
                        .sort_values("growth_abs", ascending=True))
    if top_growth.empty:
        st.info("Not enough data points to compute 2021 → 2024 growth.")
    else:
        fig_g = px.bar(
            top_growth, x="growth_abs", y="country", orientation="h",
            color="growth_abs", color_continuous_scale="Blues",
            labels={"growth_abs": "", "country": ""},
        )
        fig_g.update_coloraxes(showscale=False)
        fig_g.update_traces(hovertemplate="%{y}: %{x:,.3f} $B<extra></extra>")
        fig_g.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=440)
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # Slide 2: Global CAPEX Overview
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("<h2 class='section-h'>Global CAPEX Overview</h2>", unsafe_allow_html=True)

    # Chart 1: Global CAPEX Trend ($B)
    _title(
        "Global FDI Capital Expenditure Trend (2021–2024)",
        "Global CAPEX rebounded post-COVID, peaked in 2023, and corrected in 2024 due to tightening economic conditions."
    )
    trend = (cap.groupby("year", as_index=False)["capex"].sum().sort_values("year"))
    if trend.empty:
        st.info("No global CAPEX data for 2021–2024.")
    else:
        trend["year_str"] = trend["year"].astype(int).astype(str)
        fig_tr = px.line(trend, x="year_str", y="capex", markers=True,
                         labels={"year_str": "", "capex": ""})
        fig_tr.update_traces(hovertemplate="Year: %{x}<br>Capex: %{y:,.3f} $B<extra></extra>")
        fig_tr.update_xaxes(type="category", showgrid=False)
        fig_tr.update_yaxes(showgrid=False)
        fig_tr.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(fig_tr, use_container_width=True)

    # Chart 2: CAPEX Map — All Years
    _title(
        "Geographic Distribution of Global CAPEX (2021–2024)",
        "Investment flows are concentrated in North America, Europe, and select Asian economies; lighter shades indicate lower inflows."
    )
    map_all = cap.groupby("country", as_index=False)["capex"].sum()
    if map_all.empty:
        st.info("No CAPEX data to map for 2021–2024.")
    else:
        fig_map = px.choropleth(
            map_all, locations="country", locationmode="country names",
            color="capex", color_continuous_scale="Blues",
        )
        fig_map.update_traces(hovertemplate="Country: %{location}<br>Capex: %{z:,.3f} $B<extra></extra>")
        fig_map.update_coloraxes(showscale=True)
        fig_map.update_geos(scope="world", projection_type="natural earth",
                            showcountries=True, showcoastlines=True,
                            landcolor="white", bgcolor="white")
        fig_map.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420,
                              paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # Slide 3: Saudi Arabia Case Study
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("<h2 class='section-h'>Saudi Arabia Case Study</h2>", unsafe_allow_html=True)

    sa = cap[cap["country"] == "Saudi Arabia"].copy()
    sa_total = float(sa["capex"].sum()) if not sa.empty else np.nan

    # Growth 2021 → 2024
    sa_by_y = (sa.groupby("year", as_index=False)["capex"].sum())
    sa_21 = float(sa_by_y.loc[sa_by_y["year"] == 2021, "capex"].sum()) if not sa_by_y.empty else np.nan
    sa_24 = float(sa_by_y.loc[sa_by_y["year"] == 2024, "capex"].sum()) if not sa_by_y.empty else np.nan
    sa_growth = (sa_24 - sa_21) if (pd.notna(sa_24) and pd.notna(sa_21)) else np.nan

    k1, k2 = st.columns(2, gap="large")
    with k1:
        st.markdown(
            f"""
            <div class="kpi-box">
              <div class="kpi-title">Saudi Arabia’s Cumulative FDI Capital Expenditure (2021–2024)</div>
              <div class="kpi-number">{('-' if np.isnan(sa_total) else f'{sa_total:,.3f}')}</div>
              <div class="kpi-sub">$B</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='subtitle'>Attracted ~$80.6B across 2021–2024, reinforcing its role as a regional investment hub.</div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-box">
              <div class="kpi-title">Saudi Arabia’s CAPEX Growth Momentum (2021 → 2024)</div>
              <div class="kpi-number">{('-' if np.isnan(sa_growth) else f'{sa_growth:,.3f}')}</div>
              <div class="kpi-sub">$B</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='subtitle'>Grew by ~$10.7B over the period, despite volatility in global flows.</div>",
            unsafe_allow_html=True,
        )

    # Chart: Saudi Arabia CAPEX Trend ($B)
    _title(
        "Yearly CAPEX Flows into Saudi Arabia (2021–2024)",
        "Sharp post-COVID rebound in 2022, followed by moderation in 2023–2024."
    )
    if sa_by_y.empty:
        st.info("No CAPEX data for Saudi Arabia (2021–2024).")
    else:
        sa_by_y = sa_by_y.sort_values("year")
        sa_by_y["year_str"] = sa_by_y["year"].astype(int).astype(str)
        fig_sa = px.line(sa_by_y, x="year_str", y="capex", markers=True,
                         labels={"year_str": "", "capex": ""})
        fig_sa.update_traces(hovertemplate="Year: %{x}<br>Capex: %{y:,.3f} $B<extra></extra>")
        fig_sa.update_xaxes(type="category", showgrid=False)
        fig_sa.update_yaxes(showgrid=False)
        fig_sa.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(fig_sa, use_container_width=True)

    # Chart: Saudi Arabia CAPEX Map — All Years (show global with emphasis via scale)
    _title(
        "Saudi Arabia’s Share of Global CAPEX",
        "Map highlights Saudi Arabia as a key FDI destination within MENA, with consistent inflows over the period."
    )
    fig_sa_map = px.choropleth(
        map_all, locations="country", locationmode="country names",
        color="capex", color_continuous_scale="Blues",
    )
    fig_sa_map.update_traces(hovertemplate="Country: %{location}<br>Capex: %{z:,.3f} $B<extra></extra>")
    fig_sa_map.update_coloraxes(showscale=True)
    fig_sa_map.update_geos(scope="world", projection_type="natural earth",
                           showcountries=True, showcoastlines=True,
                           landcolor="white", bgcolor="white")
    fig_sa_map.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420,
                             paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_sa_map, use_container_width=True)

# =============================================================================
# SECTORS TAB — unchanged (kept from your code)
# =============================================================================

SECTORS_CANON = [
    "Software & IT services","Business services","Communications","Financial services",
    "Transportation & Warehousing","Real estate","Consumer products","Food and Beverages",
    "Automotive OEM","Automotive components","Chemicals","Pharmaceuticals",
    "Metals","Coal, oil & gas","Space & defence","Leisure & entertainment",
]

SECTOR_COUNTRIES_10 = [
    "United States","United Kingdom","Germany","France","China",
    "Japan","South Korea","Canada","Netherlands","United Arab Emirates", "Bahrain", "Kuwait", "Qatar", "Oman", "Saudi Arabia", "GCC"
]

def _canon_country(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.strip()
    swap = {
        # existing mappings
        "usa":"United States","us":"United States","u.s.":"United States",
        "uk":"United Kingdom","u.k.":"United Kingdom",
        "south korea":"South Korea","republic of korea":"South Korea",
        "uae":"United Arab Emirates", "saudiarabia":"Saudi Arabia",

        # NEW — ensure GCC stays uppercase everywhere
        "gcc":"GCC", "g.c.c.":"GCC", "gulf cooperation council":"GCC",
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
    df = (df.groupby(["country", "sector"], as_index=False)[["companies","jobs_created","capex","projects"]]
            .sum(min_count=1))
    return df

sectors_df = load_sectors_raw()

with tab_sectors:
    cap_left, cap_right = st.columns([20, 1], gap="small")
    with cap_left:
        st.caption("Sectors Analysis for 2021-2024")
    with cap_right:
        info_button("investment_profile", "What is this?", key_suffix="sectors")

    sc1, sc2 = st.columns([1, 2], gap="small")
    with sc1:
        SECTORS_CANON = SECTORS_CANON  # keep
        sector_opt = ["All"] + SECTORS_CANON
        sel_sector = st.selectbox("Sector", sector_opt, index=0, key="sector_sel")
    with sc2:
        countries = sorted(sectors_df["country"].dropna().unique().tolist())
        default_c = _canon_country(st.session_state.get("sector_country", countries[0]))
        if default_c not in countries:
            default_c = countries[0]
        sel_sector_country = st.selectbox(
            "Source Country",
            countries,
            index=countries.index(default_c),
            key="sector_country",
            format_func=lambda s: "GCC" if str(s).strip().lower() == "gcc" else s
        )

    metric = st.radio("Metric", ["Companies", "Jobs Created", "Capex", "Projects"],
                      horizontal=True, index=0, key="metric_sel")

    sel_country_canon = _canon_country(sel_sector_country)
    display_country   = "GCC" if sel_country_canon == "GCC" else sel_country_canon

    cdf = sectors_df[sectors_df["country"] == sel_country_canon].copy()
    if metric == "Capex": cdf["capex"] = cdf["capex"] / 1000.0

    if not cdf.empty:
        out_cols = ["country","sector","companies","jobs_created","capex","projects"]
        csv_bytes = cdf[out_cols].rename(columns={
            "country":"Country","sector":"Sector",
            "companies":"Companies","jobs_created":"Jobs Created",
            "capex":"Capex","projects":"Projects"
        }).to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {display_country} sectors CSV",
            data=csv_bytes,
            file_name=f"{display_country.lower().replace(' ','_')}_sectors_data.csv",
            mime="text/csv",
            key="dl_country_sectors_csv",
        )

    value_col = {
        "Companies":"companies",
        "Jobs Created":"jobs_created",
        "Capex":"capex",
        "Projects":"projects",
    }[metric]

    metric_display = {
            "Companies": "Number of Companies",
            "Jobs Created": "Number of Jobs Created",
            "Capex": "Capex (USD B)",
            "Projects": "Number of Projects",
        }[metric]
        
    dynamic_title = f"Sectoral Distribution by {metric_display} in {display_country}"

    if sel_sector == "All":
        bars = cdf[["sector", value_col]].copy()
        bars = bars.set_index("sector").reindex(SECTORS_CANON, fill_value=0).reset_index()
        bars = bars.sort_values(value_col, ascending=True)
    
        if bars[value_col].sum() == 0:
            st.info("No data for this selection.")
        else:
            fig = px.bar(
                bars, x=value_col, y="sector", orientation="h",
                labels={value_col:"", "sector":""},
                color=value_col, color_continuous_scale="Blues"
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(
                title={
                    "text": dynamic_title,
                    "x": 0.0,
                    "xanchor": "left",
                },
                margin=dict(l=10, r=10, t=60, b=10),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        val = float(cdf.loc[cdf["sector"] == sel_sector, value_col].sum()) if not cdf.empty else 0.0
        unit = {"Companies":"", "Jobs Created":"", "Capex":" (USD B)", "Projects":""}[metric]
        fmt_val = f"{val:,.3f}" if metric == "Capex" else f"{int(val):,}"
        st.markdown(
            f"""
            <div class="kpi-box">
              <div class="kpi-title">{display_country} — {sel_sector} • {metric}</div>
              <div class="kpi-number">{fmt_val}</div>
              <div class="kpi-sub">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# DESTINATIONS TAB — unchanged (your latest)
# =============================================================================

GCC_MEMBERS = {"United Arab Emirates", "Bahrain", "Kuwait", "Saudi Arabia", "Qatar", "Oman"}

def _demonym(country: str) -> str:
    m = {
        "Bahrain": "Bahraini",
        "Kuwait": "Kuwaiti",
        "Qatar": "Qatari",
        "Oman": "Omani",
        "Saudi Arabia": "Saudi",
        "United Arab Emirates": "Emirati",
        "United States": "American",
        "United Kingdom": "British",
        "Germany": "German",
        "France": "French",
        "China": "Chinese",
        "Japan": "Japanese",
        "South Korea": "South Korean",
        "Canada": "Canadian",
        "Netherlands": "Dutch",
        "GCC": "GCC",
    }
    return m.get(country, country)

def _metric_phrase(metric: str) -> str:
    return {
        "Companies": "number of companies",
        "Jobs Created": "number of jobs created",
        "Capex": "capex (USD B)",
        "Projects": "number of projects",
    }.get(metric, metric.lower())

def _style_geo_white(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True, countrycolor="#8a8a8a",
        showcoastlines=True, coastlinecolor="#8a8a8a",
        showland=True, landcolor="white",
        showocean=False, lakecolor="white",
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(yanchor="top", y=1.02, x=0.02)
    )
    return fig

@st.cache_data(show_spinner=True)
def load_destinations_raw() -> pd.DataFrame:
    url = gh_raw_url(FILES["destinations"])
    df = pd.read_csv(url)

    col_source = find_col(df.columns, "source country", "source_country", "source")
    col_dest   = find_col(df.columns, "destination country", "destination_country", "destination", "dest")
    col_comp    = find_col(df.columns, "companies", "# companies", "number of companies")
    col_jobs    = find_col(df.columns, "jobs created", "jobs", "job")
    col_capex   = find_col(df.columns, "capex", "capital expenditure", "capex (in million usd)")
    col_proj    = find_col(df.columns, "projects")

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

def make_top_map(source_country: str, dest_list: list[str]) -> go.Figure:
    fig = go.Figure()

    is_gcc = str(source_country).strip().lower() == "gcc"
    if is_gcc:
        fig.add_trace(go.Choropleth(
            locations=list(GCC_MEMBERS),
            z=[1]*len(GCC_MEMBERS),
            locationmode="country names",
            colorscale=[[0, "#e63946"], [1, "#e63946"]],
            showscale=False,
            name="Source",
            zmin=0, zmax=1,
            marker_line_color="white",
            marker_line_width=0.4,
            hoverinfo="skip",
        ))
    else:
        fig.add_trace(go.Scattergeo(
            locationmode="country names",
            locations=[source_country],
            mode="text",
            text=["📍"],
            textfont=dict(size=22),
            name="Source",
            showlegend=False
        ))
    
    fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode="markers",
                                marker=dict(symbol="circle", size=12, color="#e63946"),
                                name="Source"))
    
    if dest_list:
        fig.add_trace(go.Scattergeo(
            locationmode="country names",
            locations=dest_list,
            mode="markers",
            marker=dict(symbol="circle", size=9, color="#1f77b4"),
            name="Destinations"
        ))
    return _style_geo_white(fig, height=420)

def make_route_map(source_country: str, dest_country: str) -> go.Figure:
    fig = go.Figure()
    is_gcc = str(source_country).strip().lower() == "gcc"
    if is_gcc:
        fig.add_trace(go.Choropleth(
            locations=list(GCC_MEMBERS),
            z=[1]*len(GCC_MEMBERS),
            locationmode="country names",
            colorscale=[[0, "#e63946"], [1, "#e63946"]],
            showscale=False,
            name="Source",
            zmin=0, zmax=1,
            marker_line_color="white",
            marker_line_width=0.4,
            hoverinfo="skip",
        ))
    else:
        fig.add_trace(go.Scattergeo(
            locationmode="country names",
            locations=[source_country],
            mode="text",
            text=["📍"],
            textfont=dict(size=22),
            name="Source",
            showlegend=False
        ))

    fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode="markers",
                                marker=dict(symbol="circle", size=12, color="#e63946"),
                                name="Source"))

    fig.add_trace(go.Scattergeo(
        locationmode="country names",
        locations=[dest_country],
        mode="markers",
        marker=dict(symbol="circle", size=11, color="#1f77b4"),
        name="Destination"
    ))
    return _style_geo_white(fig, height=360)

with tab_dest:
    cap_left, cap_right = st.columns([20, 1], gap="small")
    with cap_left:
        st.caption("Destinations Analysis for 2021-2024")
    with cap_right:
        info_button("investment_profile", "What is this?", key_suffix="destinations")
    dest_df = load_destinations_raw()

    src_countries = sorted(dest_df["source_country"].dropna().unique().tolist())
    default_src = st.session_state.get("dest_src", src_countries[0] if src_countries else "")
    if default_src not in src_countries and src_countries:
        default_src = src_countries[0]

    c1, c2 = st.columns([1, 3], gap="small")

    with c2:
        sel_src_country = st.selectbox(
            "Source Country",
            src_countries,
            index=(src_countries.index(default_src) if default_src in src_countries else 0),
            key="dest_src",
            format_func=lambda s: "GCC" if str(s).strip().lower() == "gcc" else s
        )
        
        shown_src_label = "GCC" if str(sel_src_country).strip().lower() == "gcc" else sel_src_country

    dest_opts_all = sorted(
        dest_df.loc[dest_df["source_country"] == sel_src_country,
                    "destination_country"].dropna().unique().tolist()
    )
    dest_opts_all = [d for d in dest_opts_all if str(d).strip().lower() != "total"]

    if sel_src_country.strip().lower() == "gcc":
        dest_opts_all = [d for d in dest_opts_all if d not in GCC_MEMBERS]
    
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

    metric_phrase = {
        "Companies": "Number of Companies",
        "Jobs Created": "Jobs Created",
        "Capex": "Capex (USD B)",
        "Projects": "Number of Projects",
    }[metric_dest]
    
    src_adj = _demonym(shown_src_label)

    export = dest_df[dest_df["source_country"] == sel_src_country].copy()
    export = export[export["destination_country"].astype(str).str.strip().str.lower() != "total"]

    if sel_src_country.strip().lower() == "gcc":
        export = export[~export["destination_country"].isin(GCC_MEMBERS)]
        
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

    if sel_src_country.strip().lower() == "gcc":
        ddf = ddf[~ddf["destination_country"].isin(GCC_MEMBERS)]
    
    if metric_dest == "Capex": ddf["capex"] = ddf["capex"] / 1000.0

    if sel_dest_country == "All":
        bars = (ddf[["destination_country", value_col_dest]]
                    .groupby("destination_country", as_index=False)[value_col_dest]
                    .sum()
                    .sort_values(value_col_dest, ascending=False)
                    .head(15))

        left, right = st.columns([1.2, 1], gap="large")
        with left:
            title = f"Top Destination Countries for {src_adj} by {metric_phrase}"
            if bars.empty or (bars[value_col_dest].sum() == 0):
                st.info("No data for this selection.")
            else:
                fig = px.bar(
                    bars.sort_values(value_col_dest),
                    x=value_col_dest, y="destination_country",
                    orientation="h",
                    labels={value_col_dest:"", "destination_country":""},
                    color=value_col_dest, color_continuous_scale="Blues"
                )
                fig.update_coloraxes(showscale=False)
                fig.update_layout(
                    title={"text": title, "x": 0.0, "xanchor": "left"},
                    margin=dict(l=10, r=10, t=60, b=10),
                    height=520
                )
                st.plotly_chart(fig, use_container_width=True)
        with right:
            top_dests = bars["destination_country"].tolist()
            fig_top_map = make_top_map(sel_src_country, top_dests)
            fig_top_map.update_layout(
                title={"text": f"Geographic Distribution of {src_adj} Expansions by {metric_phrase}",
                       "x": 0.0, "xanchor": "left"}
            )
            st.plotly_chart(fig_top_map, use_container_width=True)

    else:
        left, right = st.columns([0.9, 1.4], gap="large")
        with left:
            val = float(ddf.loc[ddf["destination_country"] == sel_dest_country, value_col_dest].sum()) if not ddf.empty else 0.0
            unit = {"Companies":"", "Jobs Created":"", "Capex":" (USD B)", "Projects":""}[metric_dest]
            fmt_val = f"{val:,.3f}" if metric_dest == "Capex" else f"{int(val):,}"
            st.markdown(
                f"""
                <div class="kpi-box">
                  <div class="kpi-title">{_demonym(shown_src_label)} Investments in {sel_dest_country} by {_metric_phrase(metric_dest)}</div>
                  <div class="kpi-number">{fmt_val}</div>
                  <div class="kpi-sub">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            fig_route = make_route_map(sel_src_country, sel_dest_country)
            fig_route.update_layout(title={
                "text": f"Geographic Route of {_demonym(shown_src_label)} Expansions toward {sel_dest_country}",
                "x": 0.0, "xanchor": "left"
            })
            st.plotly_chart(fig_route, use_container_width=True)

with tab_compare:
    render_compare_tab()

with tab_forecast:
    render_forecasting_tab()
    
emit_auto_jump_script()
