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

def _sync_continent_from_country():
    """Update continent to match the selected country (if possible)."""
    country = st.session_state.get("sc_country", "All")
    year = st.session_state.get("sc_year", "All")
    if country == "All":
        return

    if year == "All":
        lookup = wb[wb["country"] == country]
    else:
        lookup = wb[(wb["year"] == int(year)) & (wb["country"] == country)]

    if not lookup.empty and lookup["continent"].notna().any():
        cont = str(lookup["continent"].dropna().iloc[0])
        cont_opts = st.session_state.get("_sc_cont_opts", [])
        if cont in cont_opts:
            st.session_state["sc_cont"] = cont

# SCORING-only filters — UPDATED to auto-sync continent to selected country
def scoring_filters_block(wb: pd.DataFrame):
    years_sc = sorted(wb["year"].dropna().astype(int).unique().tolist())
    years_sc = [y for y in years_sc if y <= 2023]
    year_opts_sc = ["All"] + years_sc

    c1, c2, c3 = st.columns([1, 1, 2], gap="small")
    with c1:
        sel_year_sc = st.selectbox("Year", year_opts_sc, index=0, key="sc_year")

    cont_pool = wb["continent"] if sel_year_sc == "All" else wb.loc[wb["year"] == int(sel_year_sc), "continent"]
    cont_opts_sc = ["All"] + sorted(cont_pool.dropna().unique().tolist())
    st.session_state["_sc_cont_opts"] = cont_opts_sc

    cur_country = st.session_state.get("sc_country", "All")
    suggested_cont = None
    if cur_country != "All":
        if sel_year_sc == "All":
            lk = wb[wb["country"] == cur_country]
        else:
            lk = wb[(wb["year"] == int(sel_year_sc)) & (wb["country"] == cur_country)]
        if not lk.empty and lk["continent"].notna().any():
            cand = str(lk["continent"].dropna().iloc[0])
            if cand in cont_opts_sc:
                suggested_cont = cand

    current_cont = st.session_state.get("sc_cont", "All")
    if suggested_cont and suggested_cont in cont_opts_sc:
        current_cont = suggested_cont
    if current_cont not in cont_opts_sc:
        current_cont = "All"

    with c2:
        sel_cont_sc = st.selectbox(
            "Continent", cont_opts_sc,
            index=cont_opts_sc.index(current_cont),
            key="sc_cont",
        )

    pool = wb if sel_year_sc == "All" else wb[wb["year"] == int(sel_year_sc)]
    if sel_cont_sc != "All":
        pool = pool[pool["continent"] == sel_cont_sc]
    country_opts_sc = ["All"] + sorted(pool["country"].dropna().unique().tolist())

    cur_country = st.session_state.get("sc_country", "All")
    country_index = country_opts_sc.index(cur_country) if cur_country in country_opts_sc else 0

    with c3:
        sel_country_sc = st.selectbox(
            "Country", country_opts_sc, index=country_index, key="sc_country",
        )

    return sel_year_sc, sel_cont_sc, sel_country_sc

def filt_wb_scoping(df: pd.DataFrame, year_any, cont, country):
    out = df.copy()
    if year_any != "All":
        out = out[out["year"] == int(year_any)]
    if cont != "All":
        out = out[out["continent"] == cont]
    if country != "All":
        out = out[out["country"] == country]
    return out

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
    return "(averaged across 2021–2023)" if year_any == "All" else f"(for {_year_token(year_any)})"

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
# SCORING TAB (Country Attractiveness) — TITLES FULLY DYNAMIC BY YEAR/FILTER
# =============================================================================
with tab_scoring:
    # Caption + info button ABOVE the filters
    cap1, cap2 = st.columns([20, 1])
    with cap1:
        # Use the current session value (defaults to "All" on first run)
        st.caption(
            f"Country Attractiveness (World Bank–based) • "
            f"{_year_token(st.session_state.get('sc_year', 'All'))}"
        )
    with cap2:
        info_button("score_trend")

    # Now render the filters
    sel_year_sc, sel_cont_sc, sel_country_sc = scoring_filters_block(wb)


    # ── Dynamic main title + subtitle, fully reflecting Year/Scope
    scope_txt = _scope_token(sel_cont_sc, sel_country_sc)
    year_txt  = _year_token(sel_year_sc)

    if sel_country_sc != "All":
        # Country-focused
        main_title = f"{sel_country_sc} — Investment Attractiveness {_subtitle_suffix(sel_year_sc)}"
        subtitle   = f"Viability score, grade, and comparison to {wb_cc.loc[wb_cc['country']==sel_country_sc,'continent'].dropna().iloc[0] if not wb_cc.loc[wb_cc['country']==sel_country_sc].empty else 'regional'} and global averages."
    elif sel_cont_sc != "All":
        # Continent-focused
        main_title = f"{sel_cont_sc} — Investment Attractiveness Overview {_subtitle_suffix(sel_year_sc)}"
        subtitle   = "Scores summarising economic, governance, and infrastructure indicators across countries in the region."
    else:
        # Global
        main_title = f"Global Country Attractiveness Rankings {_subtitle_suffix(sel_year_sc)}"
        subtitle   = "Cross-country comparison of the composite viability score."

    st.markdown(
        f"""
        <h3 style='text-align:center; margin-bottom:4px; font-weight:800'>{main_title}</h3>
        <p style='text-align:center; font-size:0.95rem; color:#555;'>{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )

    use_avg = (sel_year_sc == "All")

    if use_avg:
        # =============== ALL YEARS (2021–2023) ===============
        avg_scope = wb_avg_enriched.copy()
        if sel_cont_sc != "All":
            avg_scope = avg_scope[avg_scope["continent"] == sel_cont_sc]
        if sel_country_sc != "All":
            avg_scope = avg_scope[avg_scope["country"] == sel_country_sc]

        # KPIs
        if sel_country_sc != "All":
            country_row = avg_scope[avg_scope["country"] == sel_country_sc]
            country_score = float(country_row["avg_score"].mean()) if not country_row.empty else np.nan
            country_grade = country_row["grade"].astype(str).dropna().iloc[0] if not country_row.empty and country_row["grade"].notna().any() else "-"

            ctry_cont = wb_cc.loc[wb_cc["country"] == sel_country_sc, "continent"]
            ctry_cont = ctry_cont.dropna().iloc[0] if not ctry_cont.empty else None
            cont_rows = wb_avg_enriched[wb_avg_enriched["continent"] == ctry_cont] if ctry_cont else pd.DataFrame()
            if sel_cont_sc != "All":
                cont_rows = cont_rows[cont_rows["continent"] == sel_cont_sc]
            cont_avg = float(cont_rows["avg_score"].mean()) if not cont_rows.empty else np.nan

            k1, k2, k3 = st.columns(3, gap="large")
            with k1:
                st.metric(f"{sel_country_sc} — Average Score ({year_txt})", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
            with k2:
                st.metric("Overall Grade", country_grade)
            with k3:
                label = f"{ctry_cont} Average ({year_txt})" if ctry_cont else f"Continent Average ({year_txt})"
                st.metric(label, "-" if np.isnan(cont_avg) else f"{cont_avg:,.3f}")

        # Trend + Map
        t1, t2 = st.columns([1, 2], gap="large")
        with t1:
            if sel_country_sc != "All":
                base = wb[wb["country"] == sel_country_sc]; title = f"Viability Score Trend — {sel_country_sc} ({year_txt})"
            elif sel_cont_sc != "All":
                base = wb[wb["continent"] == sel_cont_sc]; title = f"Viability Score Trend — {sel_cont_sc} ({year_txt})"
            else:
                base = wb.copy(); title = f"Viability Score Trend — Global ({year_txt})"

            yoy_df = base.groupby("year", as_index=False)["score"].mean().sort_values("year")
            if yoy_df.empty:
                st.info("No data for this selection.")
            else:
                yoy_df["year_str"] = yoy_df["year"].astype(int).astype(str)
                y = yoy_df["score"].astype(float)
                pad = max((y.max() - y.min()) * 0.12, 0.002)
                
                fig_line = px.line(
                    yoy_df, x="year_str", y="score", markers=True,
                    labels={"year_str": "", "score": ""}, title=title
                )
                fig_line.update_xaxes(
                    type="category",
                    categoryorder="array",
                    categoryarray=yoy_df["year_str"].tolist(),
                    showgrid=False,
                    ticks="",
                    title_text=""
                )
                fig_line.update_yaxes(
                    visible=False,
                    range=[float(y.min() - pad), float(y.max() + pad)]
                )
                fig_line.update_traces(
                    mode="lines+markers+text",
                    text=[f"{v:.3f}" for v in y],
                    textposition="top center",
                    cliponaxis=False,
                    hovertemplate="Year: %{x}<br>Score: %{y:.3f}<extra></extra>"
                )
                fig_line.update_layout(height=410, margin=dict(l=10, r=10, t=60, b=30))
                st.plotly_chart(fig_line, use_container_width=True)

        with t2:
            if avg_scope.empty:
                st.info("No data for this selection.")
            else:
                map_df = avg_scope.rename(columns={"avg_score": "score"})[["country", "score"]].copy()
                fig_map = px.choropleth(
                    map_df, locations="country", locationmode="country names",
                    color="score", color_continuous_scale="Blues",
                    title=f"Country Scores Map — {scope_txt} ({year_txt})",
                )
                fig_map.update_traces(hovertemplate="Country: %{location}<br>Score: %{z:.3f}<extra></extra>")
                fig_map.update_coloraxes(showscale=True)
                scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                             "North America":"north america","South America":"south america",
                             "Oceania":"world","All":"world"}
                current_scope = scope_map.get(sel_cont_sc, "world")
                fig_map.update_geos(scope=current_scope, projection_type="natural earth",
                                    showcountries=True, showcoastlines=True,
                                    landcolor="white", bgcolor="white")
                if sel_cont_sc != "All" or sel_country_sc != "All":
                    fig_map.update_geos(fitbounds="locations")
                fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410,
                                      paper_bgcolor="white", plot_bgcolor="white")
                st.plotly_chart(fig_map, use_container_width=True)

        # Bottom row: only if NO specific country is selected
        if sel_country_sc == "All":
            b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")
            with b1:
                base = avg_scope[["country", "avg_score"]].rename(columns={"avg_score": "score"})
                title_top = (
                    f"Top 10 Countries by Investment Attractiveness — {year_txt}"
                    "<br><span style='font-size:0.9em;color:#555;'>Average scores across key investment factors.</span>"
                )
                top10 = base.dropna().sort_values("score", ascending=False).head(10)
                if top10.empty:
                    st.info("No countries available for Top 10 with this filter.")
                else:
                    fig_top = px.bar(top10.sort_values("score"), x="score", y="country", orientation="h",
                                     color="score", color_continuous_scale="Blues",
                                     labels={"score": "", "country": ""}, title=title_top)
                    fig_top.update_coloraxes(showscale=False)
                    fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                    fig_top.update_traces(hovertemplate="%{y}: %{x:.3f}<extra></extra>",text=None, texttemplate=None, textposition=None)
                    st.plotly_chart(fig_top, use_container_width=True)

            with b2:
                donut_base = avg_scope.copy()
                if donut_base.empty or donut_base["grade"].isna().all():
                    st.info("No grade data for this selection.")
                else:
                    grades = ["A+", "A", "B", "C", "D"]
                    donut = (donut_base.assign(grade=donut_base["grade"].astype(str))
                                         .loc[lambda d: d["grade"].isin(grades)]
                                         .groupby("grade", as_index=False)["country"].nunique()
                                         .rename(columns={"country": "count"})
                            ).set_index("grade").reindex(grades, fill_value=0).reset_index()
                    shades = [px.colors.sequential.Blues[-1-i] for i in range(5)]
                    cmap = {g:c for g, c in zip(grades, shades)}
                    donut_title = (
                        f"Grade Distribution — {year_txt}"
                        "<br><span style='font-size:0.9em;color:#555;'>Share of countries by letter grade (A+ to D).</span>"
                    )
                    fig_donut = px.pie(donut, names="grade", values="count", hole=0.55,
                                       title=donut_title,
                                       color="grade", color_discrete_map=cmap)
                    fig_donut.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, showlegend=True)
                    fig_donut.update_traces(hovertemplate="Grade: %{label}<br>Countries: %{value} (%{percent:.1%})<extra></extra>")
                    st.plotly_chart(fig_donut, use_container_width=True)

            with b3:
                cont_bar = (wb_avg_enriched.copy()
                            .pipe(lambda d: d if sel_cont_sc == "All" else d[d["continent"] == sel_cont_sc])
                            .groupby("continent", as_index=False)["avg_score"].mean()
                            .rename(columns={"avg_score": "score"})
                            .sort_values("score", ascending=True))
                if cont_bar.empty:
                    st.info("No continent data for this selection.")
                elif cont_bar.shape[0] == 1:
                    label = str(cont_bar["continent"].iloc[0])
                    val = float(cont_bar["score"].iloc[0])
                    st.markdown(
                        f"""
                        <div class="kpi-box">
                          <div class="kpi-title">Average Score by Continent — {label} ({year_txt})</div>
                          <div class="kpi-number">{val:,.3f}</div>
                          <div class="kpi-sub"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    title_cont = (
                        f"Average Score by Continent — {year_txt}"
                        "<br><span style='font-size:0.9em;color:#555;'>Regional comparison of mean attractiveness scores.</span>"
                    )
                    fig_cont = px.bar(cont_bar, x="score", y="continent", orientation="h",
                                      color="score", color_continuous_scale="Blues",
                                      labels={"score": "", "continent": ""}, title=title_cont)
                    fig_cont.update_coloraxes(showscale=False)
                    fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                    fig_cont.update_traces(hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                                           text=None, texttemplate=None, textposition=None)
                    st.plotly_chart(fig_cont, use_container_width=True)

    else:
        # =============== SINGLE YEAR ===============
        wb_scope = filt_wb_scoping(wb, sel_year_sc, sel_cont_sc, sel_country_sc)

        if sel_country_sc != "All":
            rows = filt_wb_scoping(wb, sel_year_sc, "All", sel_country_sc)
            country_score = float(rows["score"].mean()) if not rows.empty else np.nan
            country_grade = rows["grade"].astype(str).dropna().iloc[0] if not rows.empty and rows["grade"].notna().any() else "-"
            ctry_cont = rows["continent"].dropna().iloc[0] if not rows.empty and rows["continent"].notna().any() else None

            cont_rows = filt_wb_scoping(wb, sel_year_sc, ctry_cont, "All") if ctry_cont else pd.DataFrame()
            cont_avg = float(cont_rows["score"].mean()) if not cont_rows.empty else np.nan

            k1, k2, k3 = st.columns(3, gap="large")
            with k1:
                st.metric(f"{sel_country_sc} — Score ({year_txt})", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
            with k2:
                st.metric("Grade", country_grade)
            with k3:
                label = f"{ctry_cont} Average ({year_txt})" if ctry_cont else f"Continent Average ({year_txt})"
                st.metric(label, "-" if np.isnan(cont_avg) else f"{cont_avg:,.3f}")

        # LEFT: KPI only when NO specific country is selected
        t1, t2 = st.columns([1, 2], gap="large")
        with t1:
            if sel_country_sc == "All":
                year_i = int(sel_year_sc)
                rows = wb[wb["year"] == year_i].copy()
                if sel_cont_sc != "All":
                    rows = rows[rows["continent"] == sel_cont_sc]

                scope_label = sel_cont_sc if sel_cont_sc != "All" else "Global"
                val = float(rows["score"].mean()) if not rows.empty else np.nan

                st.markdown(
                    f"""
                    <div class="kpi-box">
                      <div class="kpi-title">{scope_label} — Mean Score ({year_txt})</div>
                      <div class="kpi-number">{'-' if np.isnan(val) else f'{val:,.3f}'}</div>
                      <div class="kpi-sub"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            # else: intentionally show nothing when a country is selected

        with t2:
            map_df = wb_scope[["country", "score"]].copy()
            map_title = f"Country Scores Map — {scope_txt} ({year_txt})"
            if map_df.empty:
                st.info("No data for this selection.")
            else:
                fig_map = px.choropleth(map_df, locations="country", locationmode="country names",
                                        color="score", color_continuous_scale="Blues", title=map_title)
                fig_map.update_coloraxes(showscale=True)
                scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                             "North America":"north america","South America":"south america",
                             "Oceania":"world","All":"world"}
                current_scope = scope_map.get(sel_cont_sc, "world")
                fig_map.update_geos(scope=current_scope, projection_type="natural earth",
                                    showcountries=True, showcoastlines=True,
                                    landcolor="white", bgcolor="white")
                if sel_cont_sc != "All" or sel_country_sc != "All":
                    fig_map.update_geos(fitbounds="locations")
                fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410,
                                      paper_bgcolor="white", plot_bgcolor="white")
                st.plotly_chart(fig_map, use_container_width=True)

        # Bottom row:
        if sel_country_sc == "All":
            b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")
            with b1:
                base = wb_scope[["country", "score"]]
                title_top = f"Top 10 Countries by Investment Attractiveness — {year_txt}"
                top10 = base.dropna().sort_values("score", ascending=False).head(10)
                if top10.empty: st.info("No countries available for Top 10 with this filter.")
                else:
                    fig_top = px.bar(top10.sort_values("score"), x="score", y="country", orientation="h",
                                     color="score", color_continuous_scale="Blues",
                                     labels={"score": "", "country": ""}, title=title_top)
                    fig_top.update_coloraxes(showscale=False)
                    fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                    st.plotly_chart(fig_top, use_container_width=True)

            with b2:
                donut_base = wb_scope.copy()
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
                                       title=f"Grade Distribution — {year_txt}",
                                       color="grade", color_discrete_map=cmap)
                    fig_donut.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, showlegend=True)
                    st.plotly_chart(fig_donut, use_container_width=True)

            with b3:
                cont_base = wb[wb["year"] == int(sel_year_sc)].copy()
                if sel_cont_sc != "All":
                    cont_base = cont_base[cont_base["continent"] == sel_cont_sc]
                cont_bar = cont_base.groupby("continent", as_index=False)["score"].mean().sort_values("score", ascending=True)
                if cont_bar.empty:
                    st.info("No continent data for this selection.")
                elif cont_bar.shape[0] == 1:
                    label = str(cont_bar["continent"].iloc[0])
                    val = float(cont_bar["score"].iloc[0])
                    st.markdown(
                        f"""
                        <div class="kpi-box">
                          <div class="kpi-title">Average Score by Continent — {label} ({year_txt})</div>
                          <div class="kpi-number">{val:,.3f}</div>
                          <div class="kpi-sub"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    title_cont = f"Average Score by Continent — {year_txt}"
                    fig_cont = px.bar(cont_bar, x="score", y="continent", orientation="h",
                                      color="score", color_continuous_scale="Blues",
                                      labels={"score": "", "continent": ""}, title=title_cont)
                    fig_cont.update_coloraxes(showscale=False)
                    fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                    st.plotly_chart(fig_cont, use_container_width=True)

# =============================================================================
# CAPEX TAB — show only your requested KPIs on TOP; suppress grade KPI when year+country chosen
# =============================================================================
with tab_eda:
    sel_year_any, sel_cont, sel_country, _filt = render_filters_block("eda")
    
    cap_left, cap_right = st.columns([20, 1], gap="small")
    with cap_left:
        st.caption("CAPEX Analysis for 2021-2024")
    with cap_right:
        info_button("capex_trend")


    # De-dup helpers (CAPEX tab only)
    shown_kpi_keys: set = set()
    shown_series_keys: set = set()

    def _kpi_block(title: str, value: float, unit: str = ""):
        """Render single-value KPIs *on top* in the two slots."""
        key = ("KPI", title)
        if key in shown_kpi_keys:
            return
        shown_kpi_keys.add(key)

        slot = _top_slots[_top_i[0] % len(_top_slots)]
        with slot:
            st.markdown(
                f"""
                <div class="kpi-box">
                  <div class="kpi-title">{title}</div>
                  <div class="kpi-number">{('-' if value is None or pd.isna(value) else f'{value:,.3f}')}</div>
                  <div class="kpi-sub">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        _top_i[0] += 1

    def _series_key(kind: str, x, y):
        x_tuple = tuple([str(v) for v in x])
        y_tuple = tuple([None if pd.isna(v) else round(float(v), 4) for v in y])
        return (kind, x_tuple, y_tuple)

    def _plotly_line_once(x_vals, y_vals, title, labels_x, labels_y, height=360, color=None):
        sig = _series_key("LINE", x_vals, y_vals)
        if sig in shown_series_keys:
            return
        shown_series_keys.add(sig)
        fig = px.line(pd.DataFrame({"x": x_vals, "y": y_vals}), x="x", y="y", markers=True, title=title)
        if color:
            fig.update_traces(line=dict(color=color))
        fig.update_traces(hovertemplate="Year: %{x}<br>Capex: %{y:,.3f} $B<extra></extra>")
        fig.update_xaxes(title=labels_x, type="category", showgrid=False)
        fig.update_yaxes(title=labels_y, showgrid=False)
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=height)
        st.plotly_chart(fig, use_container_width=True)

    # Contextual KPI title mapper
    def _pretty_kpi_title(orig_title: str, label: str) -> str:
        t = str(orig_title).strip()
        m = re.search(r"Top Countries by CAPEX Growth(.*)", t, flags=re.IGNORECASE)
        if m:
            tail = m.group(1).strip()
            tail = tail.lstrip("—").strip()
            return f"{label} — CAPEX Growth{(' ' + tail) if tail else ''}"
        m = re.search(r"Top Countries by CAPEX\s*—\s*(.*)", t, flags=re.IGNORECASE)
        if m:
            tail = m.group(1).strip()
            return f"{label} — Total CAPEX — {tail}" if tail else f"{label} — Total CAPEX"
        return f"{label} — {t}"

    def _bars_or_kpi(df: pd.DataFrame, value_col: str, name_col: str, title: str,
                     unit: str, height: int = 420, ascending_for_hbar: bool = False):
        valid = df.copy()
        valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
        valid = valid[valid[value_col].notna()]
        if valid.empty:
            st.info("No data for this selection.")
            return
    
        if valid.shape[0] == 1:
            label = str(valid[name_col].iloc[0])
            val = float(valid[value_col].iloc[0])
            kpi_title = _pretty_kpi_title(title, label)
            _kpi_block(kpi_title, val, unit)
            return
    
        ordered = valid.sort_values(value_col, ascending=ascending_for_hbar)
        sig = _series_key("BAR",
                          ordered[name_col].astype(str).tolist(),
                          ordered[value_col].astype(float).tolist())
        if sig in shown_series_keys:
            return
        shown_series_keys.add(sig)
    
        fig = px.bar(
            ordered, x=value_col, y=name_col, orientation="h",
            color=value_col, color_continuous_scale="Blues",
            labels={value_col: "", name_col: ""}, title=title
        )
        h = f"%{{y}}: %{{x:,.0f}} {unit}<extra></extra>" if unit else "%{y}: %{x:,.0f}<extra></extra>"
        fig.update_traces(hovertemplate=h, text=None, texttemplate=None, textposition=None)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=height)
        st.plotly_chart(fig, use_container_width=True)

    # filters applied to CAPEX
    grade_options = ["All", "A+", "A", "B", "C", "D"]
    auto_grade = st.session_state.get("grade_eda", "All")
    if sel_country != "All" and isinstance(sel_year_any, int):
        g_rows = wb[(wb["year"] == sel_year_any) & (wb["country"] == sel_country)]
        if not g_rows.empty and g_rows["grade"].notna().any():
            gval = str(g_rows["grade"].dropna().iloc[0])
            if gval in grade_options:
                auto_grade = gval
    sel_grade_eda = st.selectbox("Grade", grade_options,
                                 index=grade_options.index(auto_grade if auto_grade in grade_options else "All"),
                                 key="grade_eda")
    
    # --- TOP KPI slots (only the small KPIs you trigger will render here) ---
    top_k1, top_k2 = st.columns([1, 1], gap="large")
    _top_slots = [top_k1, top_k2]
    _top_i = [0]  # mutable index
    # A placeholder right under the Grade filter for the single KPI
    below_grade_kpi = st.empty()

    capx_eda = capx_enriched.copy()
    if sel_cont != "All":    capx_eda = capx_eda[capx_eda["continent"] == sel_cont]
    if sel_country != "All": capx_eda = capx_eda[capx_eda["country"] == sel_country]
    if sel_grade_eda != "All" and "grade" in capx_eda.columns:
        capx_eda = capx_eda[capx_eda["grade"] == sel_grade_eda]
    if isinstance(sel_year_any, int):
        capx_eda = capx_eda[capx_eda["year"] == sel_year_any]
        
    # Scale to $B
    capx_eda["capex"], capx_enriched["capex"] = capx_eda["capex"] / 1000.0, capx_enriched["capex"] / 1000.0

    # --- Special rule: when Year + Country chosen, show ONLY this KPI on top ---
    if isinstance(sel_year_any, int) and sel_country != "All":
        total_capex = float(capx_eda["capex"].sum()) if not capx_eda.empty else 0.0
        _kpi_block(f"{sel_country} — Total CAPEX — {sel_year_any}", total_capex, "$B")

    # ── Main 2-up area ────────────────────────────────────────────────────────
    e1, e2 = st.columns([1.6, 2], gap="large")
    with e1:
        # If a specific year is picked, we skip the left trend (you only want top KPI)
        if not isinstance(sel_year_any, int):
            trend = capx_eda.groupby("year", as_index=False)["capex"].sum().sort_values("year")
            if trend.empty: st.info("No CAPEX data for the selected filters.")
            else:
                x_vals = trend["year"].astype(int).astype(str).tolist()
                y_vals = trend["capex"].astype(float).tolist()
                title = (f"{sel_country} CAPEX Trend ($B)" if sel_country != "All"
                         else "Global CAPEX Trend ($B)")
                _plotly_line_once(x_vals, y_vals, title, labels_x="", labels_y="", height=360)

    with e2:
        if isinstance(sel_year_any, int):
            map_df = capx_eda.copy(); map_title = f"CAPEX Map — {sel_year_any}"
        else:
            map_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
            map_title = "CAPEX Map — All Years"
        if map_df.empty: st.info("No CAPEX data for this selection.")
        else:
            fig = px.choropleth(map_df,locations="country",locationmode="country names",color="capex",color_continuous_scale="Blues",title=map_title,)
            fig.update_traces(hovertemplate="Country: %{location}<br>Capex: %{z:,.3f} $B<extra></extra>")
            fig.update_coloraxes(showscale=True)
            scope_map = {"Africa":"africa","Asia":"asia","Europe":"europe",
                         "North America":"north america","South America":"south america",
                         "Oceania":"world","All":"world"}
            current_scope = scope_map.get(sel_cont, "world")
            fig.update_geos(scope=current_scope, projection_type="natural earth",
                            showcountries=True, showcoastlines=True,
                            landcolor="white", bgcolor="white")
            if sel_cont != "All" or sel_country != "All": fig.update_geos(fitbounds="locations")
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420,
                              paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # Grade views
    # Hide the entire grade view when Year + Country are chosen (per your instruction)
    hide_grade_view = isinstance(sel_year_any, int) and sel_country != "All"
    show_grade_trend = (sel_grade_eda == "All") and (not hide_grade_view)

    # Top-10 block: also skip when Year + Country chosen (it would collapse to 1 KPI)
    if not (isinstance(sel_year_any, int) and sel_country != "All"):
        if show_grade_trend:
            b1, b2, b3 = st.columns([1.2, 1.2, 1.6], gap="large")
        else:
            b1, b3 = st.columns([1.2, 1.6], gap="large")

        with b1:
            if isinstance(sel_year_any, int):
                level_df = capx_eda.copy(); title_top10 = f"Top Countries by CAPEX — {sel_year_any}"
            else:
                level_df = capx_eda.groupby("country", as_index=False)["capex"].sum()
                title_top10 = "Top Countries by CAPEX — All Years"
            top10 = level_df.dropna(subset=["capex"]).sort_values("capex", ascending=False).head(10)
            if top10.empty:
                st.info("No CAPEX data for Top 10 with this filter.")
            else:
                _bars_or_kpi(
                    df=top10.sort_values("capex"),
                    value_col="capex",
                    name_col="country",
                    title=title_top10,
                    unit="$B",
                    height=420,
                    ascending_for_hbar=True
                )

        if show_grade_trend:
            with b2:
                if "grade" in capx_eda.columns and not capx_eda.empty:
                    if isinstance(sel_year_any, int):
                        gb = (capx_enriched.copy()
                              .pipe(lambda d: d[(d["year"] == sel_year_any) &
                                                ((d["continent"] == sel_cont) if sel_cont != "All" else True) &
                                                ((d["country"] == sel_country) if sel_country != "All" else True)])
                              .assign(grade=lambda d: d["grade"].astype(str))
                              .groupby("grade", as_index=False)["capex"].sum())

                        gb_sorted = gb.sort_values("capex", ascending=True)
                        nonzero = gb_sorted.loc[gb_sorted["capex"].fillna(0) != 0, ["grade", "capex"]]
                        if nonzero.shape[0] <= 1:
                            if nonzero.empty:
                                st.info("No CAPEX data for grade view.")
                            else:
                                _kpi_block(f"CAPEX by Grade — {sel_year_any} — {nonzero['grade'].iloc[0]}",
                                           float(nonzero["capex"].iloc[0]), "$B")
                        else:
                            fig = px.bar(gb_sorted, x="capex", y="grade", orientation="h",
                                         labels={"capex": "", "grade": ""},
                                         title=f"CAPEX by Grade — {sel_year_any}",
                                         color="capex", color_continuous_scale="Blues")
                            fig.update_coloraxes(showscale=False)
                            fig.update_yaxes(categoryorder="array", categoryarray=gb_sorted["grade"].tolist())
                            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        tg = (capx_eda.assign(grade=capx_eda["grade"].astype(str))
                                       .groupby(["year", "grade"], as_index=False, observed=True)["capex"]
                                       .sum()
                                       .sort_values("year"))
                        if tg.empty:
                            st.info("No CAPEX data for grade trend.")
                        else:
                            tg["year_str"] = tg["year"].astype(int).astype(str)
                            grades_present = tg["grade"].dropna().unique().tolist()
                            if len(grades_present) == 1:
                                x_vals = tg["year_str"].tolist()
                                y_vals = tg["capex"].astype(float).tolist()
                                sig = _series_key("LINE", x_vals, y_vals)
                                if sig not in shown_series_keys:
                                    fig_single = px.line(
                                        tg, x="year_str", y="capex", color="grade",
                                        labels={"year_str": "", "capex": "", "grade": "Grade"},
                                        title="CAPEX Trend by Grade ($B)"
                                    )
                                    fig_single.update_xaxes(type="category",
                                                            categoryorder="array",
                                                            categoryarray=sorted(tg["year_str"].unique().tolist() ),
                                                            showgrid=False)
                                    fig_single.update_yaxes(showgrid=False)
                                    fig_single.update_layout(margin=dict(l=10, r=10, t=60, b=10),
                                                             height=420, legend_title_text="Grade")
                                    st.plotly_chart(fig_single, use_container_width=True)
                            else:
                                blues = px.colors.sequential.Blues
                                shades = [blues[-1], blues[-2], blues[-3], blues[-4], blues[-5]]
                                grade_order = ["A+", "A", "B", "C", "D"]
                                cmap = {g:c for g,c in zip(grade_order, shades)}
                                fig = px.line(
                                    tg, x="year_str", y="capex", color="grade",
                                    color_discrete_map=cmap,
                                    category_orders={"grade": grade_order},
                                    labels={"year_str": "", "capex": "", "grade": "Grade"},
                                    title="CAPEX Trend by Grade ($B)"
                                )
                                fig.update_traces(mode="lines+markers",hovertemplate="Year: %{x}<br>Capex: %{y:,.3f} $B<br>Grade: %{fullData.name}<extra></extra>")
                                fig.update_xaxes(type="category",
                                                 categoryorder="array",
                                                 categoryarray=sorted(tg["year_str"].unique().tolist()),
                                                 showgrid=False)
                                fig.update_yaxes(showgrid=False)
                                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10),
                                                  height=420, legend_title_text="Grade")
                                st.plotly_chart(fig, use_container_width=True)

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
                    top_growth = joined.sort_values("growth_abs").tail(10)
                    if top_growth.empty:
                        st.info("No CAPEX data for growth ranking.")
                    else:
                        _bars_or_kpi(
                            df=top_growth.sort_values("growth_abs"),
                            value_col="growth_abs",
                            name_col="country",
                            title=f"Top Countries by CAPEX Growth {label_grade} [{first_year} → {last_year}]",
                            unit="$B",
                            height=420,
                            ascending_for_hbar=True
                        )

# =============================================================================
# SECTORS TAB — UNCHANGED
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
        sector_opt = ["All"] + SECTORS_CANON
        sel_sector = st.selectbox("Sector", sector_opt, index=0, key="sector_sel")
    with sc2:
        countries = sorted(sectors_df["country"].dropna().unique().tolist())
        # Normalize whatever might be in session state (e.g., old "Gcc") to the canonical form
        default_c = _canon_country(st.session_state.get("sector_country", countries[0]))
        if default_c not in countries:
            default_c = countries[0]
    
        # Display label fix: always show "GCC" (value stored remains canonical from the list)
        sel_sector_country = st.selectbox(
            "Source Country",
            countries,
            index=countries.index(default_c),
            key="sector_country",
            format_func=lambda s: "GCC" if str(s).strip().lower() == "gcc" else s
        )

    metric = st.radio("Metric", ["Companies", "Jobs Created", "Capex", "Projects"],
                      horizontal=True, index=0, key="metric_sel")

    # Normalize the chosen country for data + display
    sel_country_canon = _canon_country(sel_sector_country)         # e.g., "GCC"
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
    
            # ⬇️ Left-align the dynamic title with the y-axis
            fig.update_layout(
                title={
                    "text": dynamic_title,
                    "x": 0.0,            # left
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
# DESTINATIONS TAB — minor KPI duplicate fix
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
        "GCC": "GCC",  # leave as-is
    }
    return m.get(country, country)

def _metric_phrase(metric: str) -> str:
    """Return a readable phrase for each metric for chart titles."""
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

    # Phrase used in titles per metric
    metric_phrase = {
        "Companies": "Number of Companies",
        "Jobs Created": "Jobs Created",
        "Capex": "Capex (USD B)",
        "Projects": "Number of Projects",
    }[metric_dest]
    
    # Use the label you already show in the UI (GCC stays "GCC")
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
