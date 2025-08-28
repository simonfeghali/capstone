# app.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from urllib.parse import quote

st.set_page_config(page_title="FDI Analytics", layout="wide")
st.title("FDI Analytics Dashboard")

# ──────────────────────────────────────────────────────────────────────────────
# Styling
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
# Data locations (GitHub)
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    "cap_csv": "capex_EDA_cleaned_filled.csv",
    "cap_csv_alt": "capex_EDA_cleaned_filled.csv",
    "cap_xlsx": "capex_EDA.xlsx",
    # Aggregated datasets
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
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

# ──────────────────────────────────────────────────────────────────────────────
# World Bank + CAPEX (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
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
# Top row (global filters) — unchanged
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
    # ... (unchanged scoring code from your working version)
    # (Omitted here for brevity — keep your last working Scoring section exactly)
    pass

# =============================================================================
# EDA (unchanged)
# =============================================================================
with tab_eda:
    # ... (unchanged EDA code from your working version)
    # (Omitted here for brevity — keep your last working EDA section exactly)
    pass

# =============================================================================
# Shared helpers for Sectors/Destinations
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

# ---------- Sectors loader ----------
@st.cache_data(show_spinner=True)
def load_sectors_raw() -> pd.DataFrame:
    url = gh_raw_url(FILES["sectors"])
    raw = pd.read_csv(url)

    cols = list(raw.columns)
    low = {c.lower(): c for c in cols}
    def need_or_none(name, *alts):
        for k in (name, *alts):
            if k in low: return low[k]
        for col in cols:
            if name in col.lower(): return col
        return None

    country = need_or_none("country", "source_country", "home_country", "origin")
    sector  = need_or_none("sector", "industry", "sector_name")
    comp    = need_or_none("companies", "company")
    jobs    = need_or_none("jobs_created", "jobs", "jobs created")
    capex   = need_or_none("capex")
    projs   = need_or_none("projects", "project_count", "nb_projects")

    if country is None or sector is None:
        raise ValueError(f"Sectors CSV missing required columns. Found: {cols}")

    df = raw.rename(columns={
        country:"country", sector:"sector",
        **({comp:"companies"} if comp else {}),
        **({jobs:"jobs_created"} if jobs else {}),
        **({capex:"capex"} if capex else {}),
        **({projs:"projects"} if projs else {}),
    }).copy()

    for m in ["companies","jobs_created","capex","projects"]:
        if m not in df.columns: df[m] = 0

    df["country"] = df["country"].astype(str).str.strip()
    df["sector"]  = df["sector"].astype(str).map(normalize_sector_name)
    for c in ["companies","jobs_created","capex","projects"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df[df["sector"].isin(SELECTED_SECTORS_ORDER)].copy()
    df["sector"] = pd.Categorical(df["sector"], categories=SELECTED_SECTORS_ORDER, ordered=True)
    return df

# ---------- Destinations loader ----------
@st.cache_data(show_spinner=True)
def load_destinations_raw() -> pd.DataFrame:
    url = gh_raw_url(FILES["destinations"])
    raw = pd.read_csv(url)

    cols = list(raw.columns)
    low = {c.lower(): c for c in cols}
    def find(name, *alts):
        for k in (name, *alts):
            if k in low: return low[k]
        for col in cols:
            if name in col.lower(): return col
        return None

    src  = find("source_country", "country", "origin", "home_country", "from_country", "source")
    dest = find("destination_country", "destination", "target_country", "host_country", "to_country")
    comp = find("companies", "company")
    jobs = find("jobs_created", "jobs", "jobs created")
    capx = find("capex")
    proj = find("projects", "project_count", "nb_projects")

    if src is None or dest is None:
        raise ValueError(f"Destinations CSV missing required columns. Found: {cols}")

    df = raw.rename(columns={
        src: "source_country",
        dest:"destination_country",
        **({comp:"companies"} if comp else {}),
        **({jobs:"jobs_created"} if jobs else {}),
        **({capx:"capex"} if capx else {}),
        **({proj:"projects"} if proj else {}),
    }).copy()

    for m in ["companies","jobs_created","capex","projects"]:
        if m not in df.columns: df[m] = 0

    df["source_country"]      = df["source_country"].astype(str).str.strip()
    df["destination_country"] = df["destination_country"].astype(str).str.strip()
    for c in ["companies","jobs_created","capex","projects"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

# ---------- Shared viz helpers ----------
def kpi_card(value: float, unit: str):
    st.markdown(
        f'<div class="kpi-wrap"><div class="kpi-num">{value:,.0f}</div>'
        f'<div class="kpi-unit">{unit}</div></div>',
        unsafe_allow_html=True
    )

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

def destinations_bar_for_source(df: pd.DataFrame, source: str, metric: str):
    metric_labels = {
        "companies": "Number of Companies",
        "jobs_created": "Jobs Created",
        "capex": "CAPEX (in million USD)",
        "projects": "Number of Projects",
    }
    nice = metric_labels.get(metric, metric)
    base = (df[df["source_country"] == source]
            .groupby("destination_country", as_index=False)[metric].sum())
    fig = px.bar(
        base.sort_values(metric, ascending=True),
        x=metric, y="destination_country", orientation="h",
        labels={metric: "", "destination_country": ""},
        color=metric, color_continuous_scale="Blues",
        title=f"{nice} by Destination — {source}"
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), height=520)
    return fig

def download_sectors_csv(df: pd.DataFrame, country: str):
    export = (df[df["country"] == country]
              .groupby("sector", as_index=False)
              .agg({"companies":"sum","jobs_created":"sum","capex":"sum","projects":"sum"})
              .set_index("sector").reindex(SELECTED_SECTORS_ORDER, fill_value=0).reset_index())
    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download CSV for {country}",
        data=csv,
        file_name=f"sectors_{country.lower().replace(' ','_')}.csv",
        mime="text/csv",
        key=f"dl_sectors_{country}"
    )

def download_destinations_csv(df: pd.DataFrame, source: str):
    export = (df[df["source_country"] == source]
              .groupby("destination_country", as_index=False)
              .agg({"companies":"sum","jobs_created":"sum","capex":"sum","projects":"sum"}))
    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download CSV for {source}",
        data=csv,
        file_name=f"destinations_{source.lower().replace(' ','_')}.csv",
        mime="text/csv",
        key=f"dl_dest_{source}"
    )

# ---------- Sectors tab UI ----------
def sectors_tab_ui():
    df = load_sectors_raw()

    countries = sorted(df["country"].unique().tolist())  # 10 source countries
    left, right = st.columns([1, 3], gap="small")
    with left:
        sect_opt = ["All"] + SELECTED_SECTORS_ORDER
        sel_sector = st.selectbox("Sector", sect_opt, index=0, key="sec_sector")
    with right:
        sel_country_local = st.selectbox("Source Country (Sectors)", countries, index=0, key="sec_country")

    metric_map = {"Companies":"companies","Jobs Created":"jobs_created","Capex":"capex","Projects":"projects"}
    pretty = st.radio("Metric", list(metric_map.keys()), horizontal=True, key="sec_metric")
    metric = metric_map[pretty]

    download_sectors_csv(df, sel_country_local)

    if sel_sector == "All":
        fig = sectors_bar_for_country(df, sel_country_local, metric, sel_country_local)
        st.plotly_chart(fig, use_container_width=True)
    else:
        row = df[(df["country"] == sel_country_local) & (df["sector"] == sel_sector)].agg({metric:"sum"})
        val = float(row.iloc[0]) if not row.empty else 0.0
        st.markdown(f'<div class="title-left">{sel_country_local} — {sel_sector} • {pretty}</div>', unsafe_allow_html=True)
        unit = {"companies":"", "jobs_created":"", "capex":"$M", "projects":""}[metric]
        kpi_card(val, unit)

# ---------- Destinations tab UI ----------
def destinations_tab_ui():
    df = load_destinations_raw()

    src_countries = sorted(df["source_country"].unique().tolist())  # the 10 sources
    c1, c2 = st.columns([3, 2], gap="small")
    with c1:
        sel_source = st.selectbox("Source Country (Destinations)", src_countries, index=0, key="dest_source")
    # Destination options depend on selected source
    dest_opts = ["All"] + sorted(df[df["source_country"] == sel_source]["destination_country"].unique().tolist())
    with c2:
        sel_dest = st.selectbox("Destination Country", dest_opts, index=0, key="dest_destination")

    metric_map = {"Companies":"companies","Jobs Created":"jobs_created","Capex":"capex","Projects":"projects"}
    pretty = st.radio("Metric", list(metric_map.keys()), horizontal=True, key="dest_metric")
    metric = metric_map[pretty]

    download_destinations_csv(df, sel_source)

    if sel_dest == "All":
        fig = destinations_bar_for_source(df, sel_source, metric)
        st.plotly_chart(fig, use_container_width=True)
    else:
        row = df[(df["source_country"] == sel_source) & (df["destination_country"] == sel_dest)].agg({metric:"sum"})
        val = float(row.iloc[0]) if not row.empty else 0.0
        st.markdown(f'<div class="title-left">{sel_source} → {sel_dest} • {pretty}</div>', unsafe_allow_html=True)
        unit = {"companies":"", "jobs_created":"", "capex":"$M", "projects":""}[metric]
        kpi_card(val, unit)

# =============================================================================
# SECTORS (uses sectors_tab_ui)
# =============================================================================
with tab_sectors:
    st.caption("Sectors Analysis")
    sectors_tab_ui()

# =============================================================================
# DESTINATIONS (uses destinations_tab_ui)
# =============================================================================
with tab_dest:
    st.caption("Destinations Analysis")
    destinations_tab_ui()
