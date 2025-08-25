# app.py
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote
from urllib.error import HTTPError, URLError

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
    # Use your exact filenames from the repo
    "world_bank": "world_bank_data_with_scores_and_continent (1).csv",
    "capex_eda": "capex_EDA (3).xlsx",
}

def url_for(fname: str) -> str:
    # percent‑encode spaces/parentheses etc.
    return f"{RAW_BASE}/{quote(fname)}"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def find_col(cols, *candidates):
    """Return the first matching column (case-insensitive exact, then contains)."""
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

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

# ──────────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_world_bank() -> pd.DataFrame:
    """Load CSV with country, year, continent, grade (A/A+ used for a metric)."""
    url = url_for(FILES["world_bank"])
    try:
        df = pd.read_csv(url)
    except (HTTPError, URLError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to fetch world bank CSV at {url}: {e}")

    # Normalize key columns
    country = find_col(df.columns, "country", "country_name", "Country Name")
    year    = find_col(df.columns, "year")
    cont    = find_col(df.columns, "continent", "region")
    grade   = find_col(df.columns, "grade", "letter_grade")

    missing = [k for k, v in {"country": country, "year": year, "continent": cont}.items() if v is None]
    if missing:
        raise ValueError(f"World bank CSV missing columns: {missing}. Found: {list(df.columns)}")

    df = df.rename(columns={
        country: "country",
        year: "year",
        cont: "continent",
        **({grade: "grade"} if grade else {})
    })
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "grade" not in df.columns:
        df["grade"] = np.nan
    # ensure string columns are clean
    df["country"] = df["country"].astype(str).str.strip()
    df["continent"] = df["continent"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=True)
def load_capex_wide_to_long() -> pd.DataFrame:
    """
    Your Excel is wide:
    - 'Source Country' | 2021 | 2022 | 2023 | 2024 | Total | Grade
    We melt the 4-digit year columns to a tidy long table: (country, year, capex).
    """
    xls = pd.ExcelFile(url_for(FILES["capex_eda"]))
    # read first sheet (or change index if needed)
    df = pd.read_excel(xls, sheet_name=0)
    if df is None or df.empty:
        raise ValueError("CAPEX sheet is empty.")

    df.columns = [str(c).strip() for c in df.columns]
    source_country = find_col(df.columns, "Source Country", "Country", "country_name", "country")
    if source_country is None:
        raise ValueError("Expected a 'Source Country' column in the CAPEX sheet.")

    # identify year columns by 'YYYY'
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    if not year_cols:
        raise ValueError("Could not find year columns (e.g., 2021, 2022, 2023, 2024) in CAPEX sheet.")

    melted = df.melt(
        id_vars=[source_country],
        value_vars=year_cols,
        var_name="year",
        value_name="capex"
    ).rename(columns={source_country: "country"})

    melted["year"] = pd.to_numeric(melted["year"], errors="coerce").astype("Int64")
    melted["capex"] = melted["capex"].map(numify)
    melted["country"] = melted["country"].astype(str).str.strip()

    # optional: drop rows where country is blank
    melted = melted[melted["country"].str.len() > 0]
    return melted

wb = load_world_bank()
capx = load_capex_wide_to_long()

# ──────────────────────────────────────────────────────────────────────────────
# FILTERS (Year • Continent • Country)
# ──────────────────────────────────────────────────────────────────────────────
years = sorted([int(y) for y in wb["year"].dropna().unique()])
c1, c2, c3 = st.columns([1, 1, 2], gap="small")

with c1:
    sel_year = st.selectbox("Year", years, index=len(years)-1 if years else 0)

with c2:
    conts = ["All"] + sorted(
        wb.loc[wb["year"] == sel_year, "continent"].dropna().astype(str).unique().tolist()
    )
    sel_cont = st.selectbox("Continent", conts, index=0)

with c3:
    wb_scope = wb[wb["year"] == sel_year].copy()
    if sel_cont != "All":
        wb_scope = wb_scope[wb_scope["continent"].astype(str) == sel_cont]
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

# Note: CAPEX file has no continent—filters are Year/Country only.
capx_f = capx[(capx["year"] == sel_year)]
if sel_country != "All":
    capx_f = capx_f[capx_f["country"].astype(str) == sel_country]

wb_f = apply_filters(wb)

# ──────────────────────────────────────────────────────────────────────────────
# METRICS (tiles)
# ──────────────────────────────────────────────────────────────────────────────
def metric_global_capex(df: pd.DataFrame) -> float:
    # sum of country CAPEX for the filtered year (ignores NaNs)
    return float(np.nansum(df["capex"])) if not df.empty else 0.0

def metric_countries_tracked(df: pd.DataFrame) -> int:
    # number of distinct countries with a CAPEX value for the filtered year
    if df.empty:
        return 0
    return int(df.loc[df["capex"].notna(), "country"].nunique())

def metric_aa_plus(df: pd.DataFrame) -> int:
    if df.empty or "grade" not in df.columns:
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
    "Metrics reflect current filters • Data is loaded directly from GitHub.</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# (Optional) Debug toggle
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Debug (optional)"):
    st.write("World Bank rows (filtered):", len(wb_f))
    st.write("CAPEX rows (filtered):", len(capx_f))
    st.dataframe(capx_f.head(10))
