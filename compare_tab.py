# compare_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import re

# ================= Config =================
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    "cap": "capex_EDA_cleaned_filled.csv",
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
}

SECT_DEST_ALLOWED = {
    "Canada","China","France","Germany","Japan","Netherlands",
    "South Korea","United Arab Emirates","United Kingdom","United States"
}

def _raw(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def _find_col(cols, *cands):
    low = {str(c).lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for cand in cands:
        for col in cols:
            if cand.lower() in str(col).lower():
                return col
    return None

# ================= Loaders =================
@st.cache_data(show_spinner=False)
def load_wb():
    try:
        df = pd.read_csv(_raw(FILES["wb"]))
    except (URLError, HTTPError, FileNotFoundError) as e:
        st.error(f"Could not load World Bank file: {e}")
        return pd.DataFrame(columns=["country","year","continent","score","grade"])

    country = _find_col(df.columns, "country", "country_name", "Country Name")
    year    = _find_col(df.columns, "year", "Year")
    cont    = _find_col(df.columns, "continent", "region")
    score   = _find_col(df.columns, "score", "viability_score", "composite_score")
    grade   = _find_col(df.columns, "grade", "letter_grade")

    need = [("country",country), ("year",year), ("continent",cont)]
    miss = [n for n,c in need if c is None]
    if miss:
        st.error(f"World Bank CSV missing columns: {', '.join(miss)}")
        return pd.DataFrame(columns=["country","year","continent","score","grade"])

    keep = {country:"country", year:"year", cont:"continent"}
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

@st.cache_data(show_spinner=False)
def load_capex():
    try:
        df = pd.read_csv(_raw(FILES["cap"]))
    except (URLError, HTTPError, FileNotFoundError) as e:
        st.error(f"Could not load CAPEX file: {e}")
        return pd.DataFrame(columns=["country","year","capex","grade"])

    src = _find_col(df.columns, "Source Country", "Country", "Source Co")
    grade_col = _find_col(df.columns, "Grade")
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    if not src or not year_cols:
        st.error("CAPEX CSV must have a source-country column and 4-digit year columns.")
        return pd.DataFrame(columns=["country","year","capex","grade"])

    id_vars = [src] + ([grade_col] if grade_col else [])
    m = df.melt(id_vars=id_vars, value_vars=year_cols,
                var_name="year", value_name="capex").rename(columns={src:"country"})
    if grade_col: m = m.rename(columns={grade_col:"grade"})
    else: m["grade"] = np.nan

    def _numify(x):
        if pd.isna(x): return np.nan
        s = str(x).replace(",", "")
        s = re.sub(r"[^\d\.\-]", "", s)
        try: return float(s)
        except Exception: return np.nan

    m["year"]   = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
    m["capex"]  = m["capex"].map(_numify)
    m["country"] = m["country"].astype(str).str.strip()

    order = ["A+", "A", "B", "C", "D"]
    m["grade"] = m["grade"].astype(str).str.strip()
    m.loc[~m["grade"].isin(order), "grade"] = np.nan
    m["grade"] = pd.Categorical(m["grade"], categories=order, ordered=True)
    return m[["country","year","capex","grade"]]

@st.cache_data(show_spinner=False)
def load_sectors():
    try:
        df = pd.read_csv(_raw(FILES["sectors"]))
    except Exception:
        return pd.DataFrame(columns=["country","sector","companies","jobs_created","capex","projects"])

    col_country = _find_col(df.columns, "country")
    col_sector  = _find_col(df.columns, "sector")
    col_comp    = _find_col(df.columns, "companies", "# companies", "number of companies")
    col_jobs    = _find_col(df.columns, "jobs created", "jobs", "job")
    col_capex   = _find_col(df.columns, "capex", "capital expenditure", "capex (in million usd)")
    col_proj    = _find_col(df.columns, "projects")

    need = [("country",col_country),("sector",col_sector),("companies",col_comp),
            ("jobs",col_jobs),("capex",col_capex),("projects",col_proj)]
    if any(c is None for _,c in need):
        return pd.DataFrame(columns=["country","sector","companies","jobs_created","capex","projects"])

    df = df.rename(columns={
        col_country:"country", col_sector:"sector",
        col_comp:"companies", col_jobs:"jobs_created",
        col_capex:"capex", col_proj:"projects"
    }).copy()

    for c in ["companies","jobs_created","capex","projects"]:
        df[c] = pd.to_numeric(pd.Series(df[c]).astype(str).str.replace(",", "", regex=False)
                              .str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
    df["country"] = df["country"].astype(str).str.strip()
    df["sector"]  = df["sector"].astype(str).str.strip()
    df = (df.groupby(["country","sector"], as_index=False)[["companies","jobs_created","capex","projects"]]
            .sum(min_count=1))
    return df

@st.cache_data(show_spinner=False)
def load_destinations():
    try:
        df = pd.read_csv(_raw(FILES["destinations"]))
    except Exception:
        return pd.DataFrame(columns=["source_country","destination_country","companies","jobs_created","capex","projects"])

    col_source = _find_col(df.columns, "source country", "source_country", "source")
    col_dest   = _find_col(df.columns, "destination country", "destination_country", "destination", "dest")
    col_comp    = _find_col(df.columns, "companies", "# companies", "number of companies")
    col_jobs    = _find_col(df.columns, "jobs created", "jobs", "job")
    col_capex   = _find_col(df.columns, "capex", "capital expenditure", "capex (in million usd)")
    col_proj    = _find_col(df.columns, "projects")

    need = [("source",col_source),("destination",col_dest),("companies",col_comp),
            ("jobs",col_jobs),("capex",col_capex),("projects",col_proj)]
    if any(c is None for _,c in need):
        return pd.DataFrame(columns=["source_country","destination_country","companies","jobs_created","capex","projects"])

    df = df.rename(columns={
        col_source:"source_raw", col_dest:"dest_raw",
        col_comp:"companies", col_jobs:"jobs_created",
        col_capex:"capex", col_proj:"projects"
    })
    for c in ["companies","jobs_created","capex","projects"]:
        df[c] = pd.to_numeric(pd.Series(df[c]).astype(str).str.replace(",", "", regex=False)
                              .str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
    df["source_country"]      = df["source_raw"].astype(str).str.strip()
    df["destination_country"] = df["dest_raw"].astype(str).str.strip()
    bad = {"total","all destinations","all","overall"}
    df = df[~df["destination_country"].str.lower().isin(bad)]
    df = (df.groupby(["source_country","destination_country"], as_index=False)
            [["companies","jobs_created","capex","projects"]].sum(min_count=1))
    return df

# ================= UI helpers =================
CSS = """
<style>
  .kpi { text-align:center; padding:16px 0; }
  .kpi .t { font-weight:700; }
  .kpi .v { font-size:42px; line-height:1; font-weight:800; margin:8px 0 4px; }
  .kpi .s { opacity:.75; }
  .pill { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; }
  .pill.good { background:#e7f3ff; color:#144a7c; }
  .pill.neutral { background:#f4f4f4; color:#333; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def _kpi(title, value, unit=""):
    disp = "-" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{float(value):,.1f}"
    st.markdown(f"""
      <div class="kpi">
        <div class="t">{title}</div>
        <div class="v">{disp}</div>
        <div class="s">{unit}</div>
      </div>
    """, unsafe_allow_html=True)

def _grade_pill(g: str) -> str:
    if not g or g == "-" or g is np.nan:
        return '<span class="pill neutral">–</span>'
    return f'<span class="pill good">{g}</span>'

# ================= Public entrypoint =================
def render_compare_tab():
    wb  = load_wb()
    cap = load_capex()
    sec = load_sectors()
    dst = load_destinations()

    if wb.empty:
        st.info("World Bank data required.")
        st.stop()

    # -------- Controls: Year & Countries first (metric options depend on them) --------
    wb_years  = sorted(wb["year"].dropna().astype(int).unique().tolist())
    cap_years = sorted(cap["year"].dropna().astype(int).unique().tolist())
    years_all = sorted(set(wb_years).union(cap_years))

    c1, c2, c3 = st.columns([1, 1, 1.5], gap="small")
    with c1:
        year_opts = ["All"] + years_all
        year_any = st.selectbox("Year", year_opts, index=0, key="cmp_year")
    latest_wb_year = max(wb_years) if wb_years else None
    countries = sorted(wb.loc[wb["year"] == latest_wb_year, "country"].dropna().unique().tolist()) if latest_wb_year else []
    if len(countries) < 2:
        st.info("Need at least two countries in data.")
        st.stop()
    with c2:
        a = st.selectbox("Country A", countries, index=0, key="cmp_a")
    with c3:
        b = st.selectbox("Country B", countries, index=1, key="cmp_b")

    if a == b:
        st.warning("Please choose two different countries.")
        st.stop()

    # Build metric options after we know A & B (to enforce allowed list for Sectors/Destinations)
    allowed_pair = (a in SECT_DEST_ALLOWED) and (b in SECT_DEST_ALLOWED)
    metric_options = ["Score & Grade", "CAPEX"]
    if allowed_pair:
        metric_options += ["Sectors", "Destinations"]

    mcol = st.container()
    with mcol:
        metric = st.selectbox("Compare by", metric_options, index=0, key="cmp_metric")

    st.markdown("---")

    # ---------------- Metric: Score & Grade (combined) ----------------
    if metric == "Score & Grade":
        def _score_grade(country):
            s = wb[wb["country"] == country]
            cont = s["continent"].dropna().iloc[0] if not s.empty and s["continent"].notna().any() else "-"
            if year_any == "All":
                score = float(s["score"].mean()) if not s.empty else np.nan
                grade = None  # no specific year, so we don't show a grade
            else:
                row = s[s["year"] == int(year_any)]
                score = float(row["score"].mean()) if not row.empty else np.nan
                grade = row["grade"].astype(str).dropna().iloc[0] if not row.empty and row["grade"].notna().any() else "-"
            return score, grade, cont

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader(a)
            scA, gA, contA = _score_grade(a)
            c1, c2 = st.columns([1, 1])
            with c1: _kpi("Score", scA)
            with c2:
                if gA is None:
                    st.caption("Select a specific year to see Grade.")
                else:
                    st.markdown(_grade_pill(gA), unsafe_allow_html=True)
            st.markdown(f"**Continent:** {contA}")

            if year_any == "All":
                s_tr = wb[wb["country"] == a].groupby("year", as_index=False)["score"].mean()
                if not s_tr.empty:
                    s_tr["ys"] = s_tr["year"].astype(int).astype(str)
                    fig = px.line(s_tr, x="ys", y="score", markers=True,
                                  labels={"ys":"","score":"Mean score"},
                                  title=f"{a} • Viability Score Trend")
                    fig.update_xaxes(type="category", showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader(b)
            scB, gB, contB = _score_grade(b)
            c1, c2 = st.columns([1, 1])
            with c1: _kpi("Score", scB)
            with c2:
                if gB is None:
                    st.caption("Select a specific year to see Grade.")
                else:
                    st.markdown(_grade_pill(gB), unsafe_allow_html=True)
            st.markdown(f"**Continent:** {contB}")

            if year_any == "All":
                s_tr = wb[wb["country"] == b].groupby("year", as_index=False)["score"].mean()
                if not s_tr.empty:
                    s_tr["ys"] = s_tr["year"].astype(int).astype(str)
                    fig = px.line(s_tr, x="ys", y="score", markers=True,
                                  labels={"ys":"","score":"Mean score"},
                                  title=f"{b} • Viability Score Trend")
                    fig.update_xaxes(type="category", showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Metric: CAPEX ----------------
    elif metric == "CAPEX":
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader(a)
            scope = cap[cap["country"] == a]
            if year_any == "All":
                tr = scope.groupby("year", as_index=False)["capex"].sum()
                if not tr.empty:
                    tr["ys"] = tr["year"].astype(int).astype(str)
                    fig = px.line(tr, x="ys", y="capex", markers=True,
                                  labels={"ys":"","capex":"CAPEX ($B)"},
                                  title=f"{a} • CAPEX Trend")
                    fig.update_xaxes(type="category", showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                _kpi(f"{a} CAPEX — {year_any}", scope.loc[scope["year"] == int(year_any), "capex"].sum(), "$B")
        with right:
            st.subheader(b)
            scope = cap[cap["country"] == b]
            if year_any == "All":
                tr = scope.groupby("year", as_index=False)["capex"].sum()
                if not tr.empty:
                    tr["ys"] = tr["year"].astype(int).astype(str)
                    fig = px.line(tr, x="ys", y="capex", markers=True,
                                  labels={"ys":"","capex":"CAPEX ($B)"},
                                  title=f"{b} • CAPEX Trend")
                    fig.update_xaxes(type="category", showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                _kpi(f"{b} CAPEX — {year_any}", scope.loc[scope["year"] == int(year_any), "capex"].sum(), "$B")

    # ---------------- Metric: Sectors (only if both A & B in whitelist) ----------------
    elif metric == "Sectors" and allowed_pair:
        if sec.empty:
            st.info("No sectors data available.")
            st.stop()
        sectors_list = sorted(sec["sector"].dropna().unique().tolist())
        c1, c2 = st.columns([1.4, 1], gap="small")
        with c1:
            sector_opt = st.selectbox("Sector", sectors_list, index=0, key="cmp_sector")
        with c2:
            sector_metric = st.selectbox("Metric", ["Companies","Jobs Created","Capex","Projects"], index=0, key="cmp_sector_metric")

        metric_map = {"Companies":"companies","Jobs Created":"jobs_created","Capex":"capex","Projects":"projects"}
        col = metric_map[sector_metric]
        valA = float(sec.loc[(sec["country"] == a) & (sec["sector"] == sector_opt), col].sum())
        valB = float(sec.loc[(sec["country"] == b) & (sec["sector"] == sector_opt), col].sum())

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader(a)
            _kpi(f"{sector_opt} • {sector_metric}", valA, "USD m" if col == "capex" else "")
        with right:
            st.subheader(b)
            _kpi(f"{sector_opt} • {sector_metric}", valB, "USD m" if col == "capex" else "")

    # ---------------- Metric: Destinations (only if both A & B in whitelist) ----------------
    elif metric == "Destinations" and allowed_pair:
        if dst.empty:
            st.info("No destinations data available.")
            st.stop()
        destsA = sorted(dst.loc[dst["source_country"] == a, "destination_country"].dropna().unique().tolist())
        destsB = sorted(dst.loc[dst["source_country"] == b, "destination_country"].dropna().unique().tolist())
        union_dests = sorted(set(destsA).union(destsB))

        c1, c2 = st.columns([1.4, 1], gap="small")
        with c1:
            dest_country = st.selectbox("Destination country", union_dests, index=0 if union_dests else None, key="cmp_dest_country")
        with c2:
            dest_metric = st.selectbox("Metric", ["Companies","Jobs Created","Capex","Projects"], index=0, key="cmp_dest_metric")

        metric_map = {"Companies":"companies","Jobs Created":"jobs_created","Capex":"capex","Projects":"projects"}
        col = metric_map[dest_metric]
        valA = float(dst.loc[(dst["source_country"] == a) & (dst["destination_country"] == dest_country), col].sum()) if union_dests else np.nan
        valB = float(dst.loc[(dst["source_country"] == b) & (dst["destination_country"] == dest_country), col].sum()) if union_dests else np.nan

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader(a)
            _kpi(f"{a} → {dest_country} • {dest_metric}", valA, "USD m" if col == "capex" else "")
        with right:
            st.subheader(b)
            _kpi(f"{b} → {dest_country} • {dest_metric}", valB, "USD m" if col == "capex" else "")

    # If user somehow lands on a hidden metric
    else:
        st.info("Select a metric to compare.")
