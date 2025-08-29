# compare_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import re

# ───────────── Config ─────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    "cap": "capex_EDA_cleaned_filled.csv",
}

def _raw(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def _find_col(cols, *cands):
    """Return a matching column from 'cols' for any of the candidates (case-insensitive, also fuzzy contains)."""
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for cand in cands:
        for col in cols:
            if cand.lower() in str(col).lower():
                return col
    return None

# ───────────── Loaders ─────────────
@st.cache_data(show_spinner=True)
def _load_wb():
    try:
        df = pd.read_csv(_raw(FILES["wb"]))
    except (URLError, HTTPError, FileNotFoundError) as e:
        st.error(f"Could not load World Bank file: {e}")
        return pd.DataFrame(columns=["country","year","continent","score","grade"])

    # Find columns robustly
    country = _find_col(df.columns, "country", "country_name", "Country Name")
    year    = _find_col(df.columns, "year", "Year")
    cont    = _find_col(df.columns, "continent", "region")
    score   = _find_col(df.columns, "score", "viability_score", "composite_score")
    grade   = _find_col(df.columns, "grade", "letter_grade")

    missing = [n for n, c in [("country", country), ("year", year), ("continent", cont)] if c is None]
    if missing:
        st.error(f"World Bank CSV missing columns: {', '.join(missing)}. Found: {list(df.columns)}")
        return pd.DataFrame(columns=["country","year","continent","score","grade"])

    keep = {country: "country", year: "year", cont: "continent"}
    if score: keep[score] = "score"
    if grade: keep[grade] = "grade"

    df = df.rename(columns=keep)[list(keep.values())].copy()
    if "score" not in df.columns: df["score"] = np.nan
    if "grade" not in df.columns: df["grade"] = np.nan

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["country"]   = df["country"].astype(str).str.strip()
    df["continent"] = df["continent"].astype(str).str.strip()

    order = ["A+", "A", "B", "C", "D"]
    df["grade"] = df["grade"].astype(str).str.strip()
    df.loc[~df["grade"].isin(order), "grade"] = np.nan
    df["grade"] = pd.Categorical(df["grade"], categories=order, ordered=True)
    return df

@st.cache_data(show_spinner=True)
def _load_capex():
    try:
        df = pd.read_csv(_raw(FILES["cap"]))
    except (URLError, HTTPError, FileNotFoundError) as e:
        st.error(f"Could not load CAPEX file: {e}")
        return pd.DataFrame(columns=["country","year","capex","grade"])

    # Identify columns
    src = _find_col(df.columns, "Source Country", "Country", "Source Co")
    grade_col = _find_col(df.columns, "Grade")
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]

    if not src or not year_cols:
        st.error("CAPEX CSV must have a source-country column and 4-digit year columns.")
        return pd.DataFrame(columns=["country","year","capex","grade"])

    id_vars = [src] + ([grade_col] if grade_col else [])
    m = df.melt(id_vars=id_vars, value_vars=year_cols,
                var_name="year", value_name="capex").rename(columns={src: "country"})
    if grade_col: m = m.rename(columns={grade_col: "grade"})
    else: m["grade"] = np.nan

    m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
    # numeric capex
    def _numify(x):
        if pd.isna(x): return np.nan
        s = str(x).replace(",", "")
        s = re.sub(r"[^\d\.\-]", "", s)
        try: return float(s)
        except Exception: return np.nan
    m["capex"] = m["capex"].map(_numify)

    m["country"] = m["country"].astype(str).str.strip()
    order = ["A+", "A", "B", "C", "D"]
    m["grade"] = m["grade"].astype(str).str.strip()
    m.loc[~m["grade"].isin(order), "grade"] = np.nan
    m["grade"] = pd.Categorical(m["grade"], categories=order, ordered=True)
    return m[["country","year","capex","grade"]]

# ───────────── Small UI helpers ─────────────
def _kpi(title, value, unit=""):
    disp = "-" if pd.isna(value) else f"{float(value):,.1f}"
    st.markdown(
        f"""
        <div style="text-align:center;padding:14px 0">
          <div style="font-weight:700">{title}</div>
          <div style="font-size:48px;font-weight:800;line-height:1;margin:8px 0 2px">{disp}</div>
          <div style="opacity:.75">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _line(df: pd.DataFrame, x: str, y: str, title: str, ylab: str, height=300):
    if df.empty: 
        st.info("No data for this selection.")
        return
    d = df.copy().sort_values(x)
    d["x"] = d[x].astype(int).astype(str)
    fig = px.line(d, x="x", y=y, markers=True, title=title,
                  labels={"x":"", y:ylab})
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=height)
    st.plotly_chart(fig, use_container_width=True)

def _bars(df: pd.DataFrame, x: str, y: str, title: str, height=320):
    if df.empty or df[y].fillna(0).sum() == 0:
        st.info("No data for this selection.")
        return
    d = df.copy().sort_values(y)
    fig = px.bar(d, x=y, y=x, orientation="h", title=title,
                 labels={x:"", y:""}, color=y, color_continuous_scale="Blues")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=height)
    st.plotly_chart(fig, use_container_width=True)

# ───────────── Public entrypoint ─────────────
def render_compare_tab():
    wb = _load_wb()
    cap = _load_capex()

    if wb.empty:
        st.stop()

    # Year & countries
    wb_years  = sorted(wb["year"].dropna().astype(int).unique().tolist())
    cap_years = sorted(cap["year"].dropna().astype(int).unique().tolist())
    years = sorted(set(wb_years).union(cap_years))
    year_opts = ["All"] + years

    c1, c2, c3 = st.columns([1,1,2], gap="small")
    with c1:
        year_any = st.selectbox("Year", year_opts, index=0, key="cmp_year")
    latest_year = max(wb_years) if wb_years else None
    if latest_year is None:
        st.error("No World Bank years found.")
        st.stop()

    countries = sorted(wb.loc[wb["year"] == latest_year, "country"].dropna().unique().tolist())
    if len(countries) < 2:
        st.error("Need at least two countries to compare.")
        st.stop()

    with c2:
        a = st.selectbox("Country A", countries, index=0, key="cmp_a")
    with c3:
        b = st.selectbox("Country B", countries, index=1, key="cmp_b")

    if a == b:
        st.warning("Choose two different countries.")
        st.stop()

    def _score_meta(country):
        s = wb[wb["country"] == country]
        cont = s["continent"].dropna().iloc[0] if not s.empty and s["continent"].notna().any() else "-"
        if year_any == "All":
            score = float(s["score"].mean()) if not s.empty else np.nan
            grade = "-"
        else:
            row = s[s["year"] == int(year_any)]
            score = float(row["score"].mean()) if not row.empty else np.nan
            grade = row["grade"].astype(str).dropna().iloc[0] if not row.empty and row["grade"].notna().any() else "-"
        return score, grade, cont

    st.markdown("### Overview")
    colA, colB = st.columns(2, gap="large")

    # Left
    with colA:
        st.subheader(a)
        sc, gr, cont = _score_meta(a)
        k1,k2,k3 = st.columns(3)
        with k1: _kpi("Score", sc)
        with k2: st.markdown(f"**Grade:** {gr}")
        with k3: st.markdown(f"**Continent:** {cont}")

        s_trend = wb[wb["country"] == a].groupby("year", as_index=False)["score"].mean()
        if year_any == "All":
            _line(s_trend, "year", "score", f"{a} • Viability Score Trend", "Mean score")
        cap_scope = cap[cap["country"] == a]
        if year_any != "All":
            cap_now = cap_scope[cap_scope["year"] == int(year_any)]
            _kpi(f"{a} CAPEX — {year_any}", cap_now["capex"].sum(), "$B")
        else:
            t = cap_scope.groupby("year", as_index=False)["capex"].sum()
            _line(t, "year", "capex", f"{a} • CAPEX Trend", "CAPEX ($B)")
        # Grade view
        if year_any != "All":
            gb = (cap_scope[cap_scope["year"] == int(year_any)]
                  .assign(grade=lambda d: d["grade"].astype(str))
                  .groupby("grade", as_index=False)["capex"].sum())
            grades = ["A+","A","B","C","D"]
            gb = gb.set_index("grade").reindex(grades, fill_value=0).reset_index()
            _bars(gb.rename(columns={"grade":"Grade"}), "Grade", "capex",
                  f"{a} • CAPEX by Grade — {year_any}")

    # Right
    with colB:
        st.subheader(b)
        sc, gr, cont = _score_meta(b)
        k1,k2,k3 = st.columns(3)
        with k1: _kpi("Score", sc)
        with k2: st.markdown(f"**Grade:** {gr}")
        with k3: st.markdown(f"**Continent:** {cont}")

        s_trend = wb[wb["country"] == b].groupby("year", as_index=False)["score"].mean()
        if year_any == "All":
            _line(s_trend, "year", "score", f"{b} • Viability Score Trend", "Mean score")
        cap_scope = cap[cap["country"] == b]
        if year_any != "All":
            cap_now = cap_scope[cap_scope["year"] == int(year_any)]
            _kpi(f"{b} CAPEX — {year_any}", cap_now["capex"].sum(), "$B")
        else:
            t = cap_scope.groupby("year", as_index=False)["capex"].sum()
            _line(t, "year", "capex", f"{b} • CAPEX Trend", "CAPEX ($B)")
        if year_any != "All":
            gb = (cap_scope[cap_scope["year"] == int(year_any)]
                  .assign(grade=lambda d: d["grade"].astype(str))
                  .groupby("grade", as_index=False)["capex"].sum())
            grades = ["A+","A","B","C","D"]
            gb = gb.set_index("grade").reindex(grades, fill_value=0).reset_index()
            _bars(gb.rename(columns={"grade":"Grade"}), "Grade", "capex",
                  f"{b} • CAPEX by Grade — {year_any}")

    # Head-to-head
    st.markdown("### Head-to-Head")
    scA, grA, _ = _score_meta(a)
    scB, grB, _ = _score_meta(b)
    if year_any == "All":
        capA = cap[cap["country"] == a]["capex"].sum()
        capB = cap[cap["country"] == b]["capex"].sum()
    else:
        capA = cap[(cap["country"] == a) & (cap["year"] == int(year_any))]["capex"].sum()
        capB = cap[(cap["country"] == b) & (cap["year"] == int(year_any))]["capex"].sum()

    d1,d2,d3 = st.columns(3)
    with d1: _kpi("Score Δ (A − B)", (0 if pd.isna(scA) else scA) - (0 if pd.isna(scB) else scB))
    with d2: st.markdown(f"**Grade A/B:** `{grA}` / `{grB}`")
    with d3: _kpi("CAPEX Δ (A − B)", capA - capB, "$B")
