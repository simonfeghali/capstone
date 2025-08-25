# app.py
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote
from urllib.error import HTTPError, URLError
import plotly.express as px

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
    "wb":  "world_bank_data_with_scores_and_continent (1).csv",
    "cap": "capex_EDA_cleaned_filled (9).csv",
}

def url_for(fname: str) -> str:
    # Percent‑encode spaces and parentheses so raw.githubusercontent works
    return f"{RAW_BASE}/{quote(fname)}"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def find_col(cols, *candidates):
    """Return the first matching column (case-insensitive exact first, then contains)."""
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
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = str(x).replace(",", "").strip()
    try: return float(s)
    except Exception: return np.nan

# ──────────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_world_bank() -> pd.DataFrame:
    url = url_for(FILES["wb"])
    try:
        df = pd.read_csv(url)
    except (HTTPError, URLError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to fetch World Bank CSV at {url}: {e}")

    # Expected: country, year, continent, (score?), grade
    country = find_col(df.columns, "country", "country_name", "Country Name")
    year    = find_col(df.columns, "year")
    cont    = find_col(df.columns, "continent", "region")
    # score column could have many names
    score   = find_col(df.columns, "score", "viability_score", "composite_score", "overall_score")
    grade   = find_col(df.columns, "grade", "letter_grade")

    missing = [k for k,v in {"country":country, "year":year, "continent":cont}.items() if v is None]
    if missing:
        raise ValueError(f"World Bank CSV missing required columns: {missing}. Found: {list(df.columns)}")

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
    return df

@st.cache_data(show_spinner=True)
def load_capex() -> pd.DataFrame:
    """
    Reads the cleaned wide CSV and melts year columns into (country, year, capex).
    Columns like:
    'Source Country' | 2021 | 2022 | 2023 | 2024 | Total | Grade | Calculated Total
    """
    url = url_for(FILES["cap"])
    try:
        df = pd.read_csv(url)
    except (HTTPError, URLError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to fetch CAPEX CSV at {url}: {e}")

    df.columns = [str(c).strip() for c in df.columns]
    src = find_col(df.columns, "Source Country", "Source Co", "Country")
    if not src:
        raise ValueError("CAPEX CSV must include a 'Source Country' column.")

    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    if not year_cols:
        raise ValueError("CAPEX CSV has no 4‑digit year columns (e.g., 2021, 2022…).")

    melted = df.melt(
        id_vars=[src],
        value_vars=year_cols,
        var_name="year",
        value_name="capex"
    ).rename(columns={src: "country"})

    melted["year"]    = pd.to_numeric(melted["year"], errors="coerce").astype("Int64")
    melted["capex"]   = melted["capex"].map(numify)
    melted["country"] = melted["country"].astype(str).str.strip()
    return melted

wb   = load_world_bank()
capx = load_capex()

# ──────────────────────────────────────────────────────────────────────────────
# FILTERS (Year • Continent • Country)
# Years from UNION of both datasets so 2024 appears (CAPEX has it)
# ──────────────────────────────────────────────────────────────────────────────
years_union = sorted(set(wb["year"].dropna().astype(int)).union(set(capx["year"].dropna().astype(int))))
c1, c2, c3 = st.columns([1, 1, 2], gap="small")

with c1:
    sel_year = st.selectbox("Year", years_union, index=len(years_union)-1 if years_union else 0)

with c2:
    conts = ["All"] + sorted(wb.loc[wb["year"] == sel_year, "continent"].dropna().astype(str).unique().tolist())
    sel_cont = st.selectbox("Continent", conts, index=0)

with c3:
    wb_scope = wb[wb["year"] == sel_year].copy()
    if sel_cont != "All":
        wb_scope = wb_scope[wb_scope["continent"].astype(str) == sel_cont]
    countries = ["All"] + sorted(wb_scope["country"].dropna().astype(str).unique().tolist())
    sel_country = st.selectbox("Country", countries, index=0)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns: out = out[out["year"] == sel_year]
    if sel_cont != "All" and "continent" in out.columns:
        out = out[out["continent"].astype(str) == sel_cont]
    if sel_country != "All" and "country" in out.columns:
        out = out[out["country"].astype(str) == sel_country]
    return out

wb_f = apply_filters(wb)

# CAPEX by selected year; map continent via WB when filtering by continent
capx_f = capx[capx["year"] == sel_year]
if sel_cont != "All":
    valid_countries = set(wb_scope["country"].unique())
    capx_f = capx_f[capx_f["country"].isin(valid_countries)]
if sel_country != "All":
    capx_f = capx_f[capx_f["country"].astype(str) == sel_country]

# ──────────────────────────────────────────────────────────────────────────────
# KPI TILES
# ──────────────────────────────────────────────────────────────────────────────
def kpi_global_capex(df: pd.DataFrame) -> float:
    return float(np.nansum(df["capex"])) if not df.empty else 0.0

def kpi_countries_tracked(df: pd.DataFrame) -> int:
    if df.empty: return 0
    return int(df.loc[df["capex"].notna(), "country"].nunique())

def kpi_aa_plus(df: pd.DataFrame) -> int:
    if df.empty or "grade" not in df.columns: return 0
    return int(df[df["grade"].astype(str).isin(["A", "A+"])]["country"].nunique())

m1, m2, m3 = st.columns(3, gap="large")
with m1:
    st.metric("Global CAPEX (latest, $B)", f"{kpi_global_capex(capx_f):,.0f}")
with m2:
    st.metric("# Countries Tracked", f"{kpi_countries_tracked(capx_f):,}")
with m3:
    st.metric("A/A+ Countries", f"{kpi_aa_plus(wb_f):,}")

# ──────────────────────────────────────────────────────────────────────────────
# VISUALS: Global CAPEX Trend • Grade Distribution
# ──────────────────────────────────────────────────────────────────────────────
v1, v2 = st.columns((3, 2), gap="large")

with v1:
    # CAPEX trend over all years, filtered by continent/country
    capx_for_trend = capx.copy()
    if sel_cont != "All":
        valid_countries_all = set(wb[wb["continent"].astype(str) == sel_cont]["country"].unique())
        capx_for_trend = capx_for_trend[capx_for_trend["country"].isin(valid_countries_all)]
    if sel_country != "All":
        capx_for_trend = capx_for_trend[capx_for_trend["country"].astype(str) == sel_country]

    trend = (capx_for_trend.groupby("year", as_index=False)["capex"].sum().sort_values("year"))
    fig = px.line(trend, x="year", y="capex", markers=True,
                  labels={"year": "", "capex": "Global CAPEX ($B)"},
                  title="Global CAPEX Trend")
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

with v2:
    grades = ["A+", "A", "B", "C", "D"]
    if wb_f.empty or "grade" not in wb_f.columns:
        st.info("No grade data for this year/selection.")
    else:
        dist = (
            wb_f.assign(grade=wb_f["grade"].astype(str))
               .loc[wb_f["grade"].astype(str).isin(grades)]
               .groupby("grade", as_index=False)["country"].nunique()
               .rename(columns={"country": "count"})
        )
        dist = dist.set_index("grade").reindex(grades, fill_value=0).reset_index()
        bar = px.bar(dist, x="grade", y="count",
                     labels={"grade": "", "count": "Number of countries"},
                     title="Grade Distribution")
        bar.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=380)
        st.plotly_chart(bar, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Top Countries table (Country • Grade • Score • YoY % • CAPEX ($B))
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Top Countries  \n_Score vs. CAPEX_")

# Build YoY% from CAPEX
prev_year = sel_year - 1
cap_curr = capx[capx["year"] == sel_year][["country", "capex"]].rename(columns={"capex": "capex_curr"})
cap_prev = capx[capx["year"] == prev_year][["country", "capex"]].rename(columns={"capex": "capex_prev"})

cap_join = cap_curr.merge(cap_prev, on="country", how="left")
cap_join["yoy_pct"] = np.where(
    cap_join["capex_prev"].abs() > 0,
    (cap_join["capex_curr"] - cap_join["capex_prev"]) / cap_join["capex_prev"],
    np.nan
)

# Take WB for selected year: country, grade, score (if exists)
wb_cols = ["country"]
if "grade" in wb.columns: wb_cols.append("grade")
if "score" in wb.columns: wb_cols.append("score")

wb_year = wb[wb["year"] == sel_year][wb_cols].copy()

# Merge WB with CAPEX; filter by selected continent/country
if sel_cont != "All":
    wb_year = wb_year.merge(wb_scope[["country"]].drop_duplicates(), on="country", how="inner")
if sel_country != "All":
    wb_year = wb_year[wb_year["country"].astype(str) == sel_country]

top_tbl = wb_year.merge(cap_join, on="country", how="left")

# Choose ranking: prefer by 'score' desc; if score missing, by capex_curr desc
if "score" in top_tbl.columns and top_tbl["score"].notna().any():
    top_tbl = top_tbl.sort_values(["score", "capex_curr"], ascending=[False, False])
else:
    top_tbl = top_tbl.sort_values("capex_curr", ascending=False)

# Format and display (limit to 6 like your mock)
def fmt_pct(x):
    return "" if pd.isna(x) else f"{x*100:,.1f}%"

def fmt_num(x):
    return "" if pd.isna(x) else f"{x:,.0f}"

show_cols = []
show_cols.append("country")
if "grade" in top_tbl.columns: show_cols.append("grade")
if "score" in top_tbl.columns: show_cols.append("score")
show_cols += ["yoy_pct", "capex_curr"]

nice = top_tbl[show_cols].head(6).rename(columns={
    "country": "Country",
    "grade": "Grade",
    "score": "Score",
    "yoy_pct": "YoY %",
    "capex_curr": "CAPEX ($B)"
})

# Apply simple formatting
if "Score" in nice.columns:
    nice["Score"] = nice["Score"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}" if abs(v) > 1 else f"{v:,.2f}")
nice["YoY %"] = nice["YoY %"].map(fmt_pct)
nice["CAPEX ($B)"] = nice["CAPEX ($B)"].map(fmt_num)

st.dataframe(
    nice,
    hide_index=True,
    use_container_width=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Optional debug
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Debug (optional)"):
    st.write("Years available (union):", years_union)
    st.write("WB rows (filtered):", len(wb_f))
    st.write("CAPEX rows (filtered):", len(capx_f))
    st.dataframe(capx_f.head(10))
