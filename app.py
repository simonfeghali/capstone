import io
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote

# ──────────────────────────────────────────────────────────────────────────────
# PAGE
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FDI Analytics Dashboard", layout="wide")
st.title("FDI Analytics Dashboard")
st.caption("EDA • Viability Scoring • Forecasting • Comparisons • Scenarios")

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOCATIONS (GitHub RAW). No uploads needed.
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"

FILES = {
    "world_bank": "world_bank_data_with_scores_and_continent.csv",
    "capex_eda": "capex_EDA (3).xlsx",  # note: space + parentheses
}

def url_for(fname: str) -> str:
    # Percent‑encode the path segment so spaces () etc. work with raw.githubusercontent
    return f"{RAW_BASE}/{quote(fname)}"

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def find_col(cols, *candidates):
    """Find a column by case-insensitive exact then contains match."""
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    for c in candidates:
        key = c.lower()
        for col in cols:
            if key in col.lower():
                return col
    return None

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_world_bank() -> pd.DataFrame:
    df = pd.read_csv(url_for(FILES["world_bank"]))
    # Standardize
    country = find_col(df.columns, "country", "country_name", "Country Name")
    year    = find_col(df.columns, "year")
    cont    = find_col(df.columns, "continent", "region")
    grade   = find_col(df.columns, "grade", "letter_grade")

    missing = [k for k,v in {"country":country,"year":year,"continent":cont}.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns in world_bank CSV: {missing}")

    df = df.rename(columns={
        country: "country",
        year: "year",
        cont: "continent",
        **({grade: "grade"} if grade else {})
    })
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "grade" not in df.columns:
        df["grade"] = np.nan
    return df

@st.cache_data(show_spinner=True)
def load_capex() -> pd.DataFrame:
    # Find a sheet that contains Year + CAPEX (any name containing 'capex')
    xls = pd.ExcelFile(url_for(FILES["capex_eda"]))
    chosen = None
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sh)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        year = find_col(df.columns, "year")
        capex = next((c for c in df.columns if "capex" in c.lower()), None)
        if year and capex:
            country = find_col(df.columns, "country", "country_name")
            cont    = find_col(df.columns, "continent", "region")
            df = df.rename(columns={
                year: "year",
                capex: "capex",
                **({country: "country"} if country else {}),
                **({cont: "continent"} if cont else {}),
            })
            df["year"]  = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["capex"] = pd.to_numeric(df["capex"], errors="coerce")
            chosen = df
            break
    if chosen is None:
        raise ValueError("Could not find a (Year, CAPEX*) table in 'capex_EDA (3).xlsx'.")
    return chosen

wb   = load_world_bank()
capx = load_capex()

# ──────────────────────────────────────────────────────────────────────────────
# FILTERS (Year • Continent • Country)
# ──────────────────────────────────────────────────────────────────────────────
years = sorted([int(y) for y in wb["year"].dropna().unique()])
c1, c2, c3 = st.columns([1, 1, 2], gap="small")

with c1:
    sel_year = st.selectbox("Year", years, index=len(years)-1 if years else 0)

with c2:
    conts = ["All"] + sorted(wb.loc[wb["year"]==sel_year, "continent"].dropna().astype(str).unique().tolist())
    sel_cont = st.selectbox("Continent", conts, index=0)

with c3:
    wb_scope = wb[wb["year"]==sel_year].copy()
    if sel_cont != "All":
        wb_scope = wb_scope[wb_scope["continent"].astype(str)==sel_cont]
    countries = ["All"] + sorted(wb_scope["country"].dropna().astype(str).unique().tolist())
    sel_country = st.selectbox("Country", countries, index=0)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        out = out[out["year"] == sel_year]
    if sel_cont != "All" and "continent" in out.columns:
        out = out[out["continent"].astype(str) == sel_cont]
    if sel_country != "All" and "country" in out.columns:
        out = out[out["country"].astype(str) == sel_country]
    return out

wb_f   = apply_filters(wb)
capx_f = apply_filters(capx)

# ──────────────────────────────────────────────────────────────────────────────
# METRICS (tiles)
# ──────────────────────────────────────────────────────────────────────────────
def metric_global_capex(df: pd.DataFrame) -> float:
    if "country" in df.columns and df["country"].notna().any():
        return float(np.nansum(df["capex"]))
    return float(np.nanmax(df["capex"]))

def metric_countries_tracked(df: pd.DataFrame) -> int:
    if "country" in df.columns:
        return int(df["country"].dropna().nunique())
    # Fallback to world-bank scope if CAPEX sheet has no per-country rows
    return int(wb_f["country"].dropna().nunique())

def metric_aa_plus(df: pd.DataFrame) -> int:
    if "grade" not in df.columns:
        return 0
    return int(df[df["grade"].astype(str).isin(["A", "A+"])]["country"].nunique())

m1, m2, m3 = st.columns(3, gap="large")
with m1:
    st.metric("Global CAPEX (latest, $B)", f"{metric_global_capex(capx_f):,.0f}")
with m2:
    st.metric("# Countries Tracked", f"{metric_countries_tracked(capx_f):,}")
with m3:
    st.metric("A/A+ Countries", f"{metric_aa_plus(wb_f):,}")

st.markdown(
    "<div style='color:#94a3b8; font-size:0.9rem;'>"
    "Metrics reflect current filters • Data loaded from GitHub.</div>",
    unsafe_allow_html=True,
)
