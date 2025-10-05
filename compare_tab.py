# compare_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import re
import math
from overview import info_button, emit_auto_jump_script

# ================= Config =================
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":       "world_bank_data_with_scores_and_continent.csv",
    "wb_avg":   "world_bank_data_average_scores_and_grades.csv",  # used for KPI panel
    "cap":      "capex_EDA_cleaned_filled.csv",
    "sectors":  "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
}

# Countries we have complete Sectors/Destinations drilldowns for
SECT_DEST_ALLOWED = {
    "Canada","China","France","Germany","Japan","Netherlands",
    "South Korea","United Arab Emirates","United Kingdom","United States"
}

TOP_COUNTRIES = [
    "United States","United Kingdom","Germany","France","China",
    "Japan","South Korea","Canada","Netherlands","United Arab Emirates",
    "Bahrain","Kuwait","Qatar","Oman","Saudi Arabia"
]

# Multiselect section headers (non-selectable)
HDR_TOP = "— Top countries —"
HDR_CONT = "— Continents —"
HDR_OTHER = "— Other countries —"
HEADERS = {HDR_TOP, HDR_CONT, HDR_OTHER}

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

def _style_compare_line(fig, unit: str | None = None):
    """Apply consistent styling/hover to compare tab line charts."""
    if unit == "$B":
        htmpl = "Series: %{fullData.name}<br>Year: %{x}<br>Value: %{y:.2f} $B<extra></extra>"
    else:
        htmpl = "Series: %{fullData.name}<br>Year: %{x}<br>Score: %{y:.3f}<extra></extra>"
    fig.update_traces(mode="lines+markers", hovertemplate=htmpl, texttemplate=None, textposition=None)
    fig.update_xaxes(type="category", showgrid=False, title=None)
    fig.update_yaxes(showgrid=False, title=None)
    fig.update_layout(hovermode="closest")
    return fig

# === Canonicalize country names ===
def _canon_country(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    swap = {
        "usa": "United States",
        "us": "United States",
        "u.s.": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "south korea": "South Korea",
        "republic of korea": "South Korea",
        "korea, rep.": "South Korea",
        "uae": "United Arab Emirates",
    }
    low = s.lower()
    if low in swap:
        return swap[low]
    t = " ".join(w.capitalize() for w in low.split())
    t = t.replace("Of", "of")
    return t

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
    df["country"]   = df["country"].astype(str).str.strip().map(_canon_country)
    df["continent"] = df["continent"].astype(str).str.strip()

    order = ["A+", "A", "B", "C", "D"]
    df["grade"] = df["grade"].astype(str).str.strip()
    df.loc[~df["grade"].isin(order), "grade"] = np.nan
    df["grade"] = pd.Categorical(df["grade"], categories=order, ordered=True)
    return df

@st.cache_data(show_spinner=False)
def load_wb_avg():
    """Averages file used for the KPI panel."""
    try:
        df = pd.read_csv(_raw(FILES["wb_avg"]))
    except (URLError, HTTPError, FileNotFoundError) as e:
        st.error(f"Could not load averages file: {e}")
        return pd.DataFrame(columns=["country","avg_score","grade"])

    col_country = _find_col(df.columns, "country name", "country", "Country Name")
    col_avg     = _find_col(df.columns, "average_score", "avg_score", "Average_Score", "score")
    col_grade   = _find_col(df.columns, "final_grade", "grade", "Final_Grade", "letter_grade")

    if col_country is None or col_avg is None:
        st.error("Averages CSV missing required columns.")
        return pd.DataFrame(columns=["country","avg_score","grade"])

    out = df.rename(columns={
        col_country: "country",
        col_avg: "avg_score",
        col_grade if col_grade else "": "grade",
    }).copy()

    if "grade" not in out.columns:
        out["grade"] = np.nan

    out["country"]   = out["country"].astype(str).str.strip().map(_canon_country)
    out["avg_score"] = pd.to_numeric(out["avg_score"], errors="coerce")

    order = ["A+", "A", "B", "C", "D"]
    out["grade"] = out["grade"].astype(str).str.strip()
    out.loc[~out["grade"].isin(order), "grade"] = np.nan
    out["grade"] = pd.Categorical(out["grade"], categories=order, ordered=True)

    return out[["country","avg_score","grade"]]

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
    # Convert CAPEX from millions to billions (absolute scale only)
    m["capex"] = m["capex"] / 1000.0
    m["country"] = m["country"].astype(str).str.strip().map(_canon_country)

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
        df[c] = pd.to_numeric(pd.Series(df[c]).astype(str)
                              .str.replace(",", "", regex=False)
                              .str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
    df["country"] = df["country"].astype(str).str.strip().map(_canon_country)
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
        df[c] = pd.to_numeric(
            pd.Series(df[c]).astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce"
        )

    df["source_country"]      = df["source_raw"].astype(str).str.strip().map(_canon_country)
    df["destination_country"] = df["dest_raw"].astype(str).str.strip().map(_canon_country)

    bad = {"total","all destinations","all","overall"}
    df = df[~df["destination_country"].astype(str).str.lower().isin(bad)]

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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        disp = "-"
    else:
        disp = f"{float(value):,.3f}"
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

# ---------- Entity helpers (country vs continent) ----------
def _build_selection_lists(all_countries, continents):
    top = [c for c in TOP_COUNTRIES if c in all_countries]
    others = [c for c in all_countries if c not in top]
    options = [HDR_TOP] + top + [HDR_CONT] + continents + [HDR_OTHER] + others
    m = {}
    for c in top:
        m[c] = ("country", c, c)
    for ct in continents:
        # Display just the continent name; do not append "(aggregate)"
        m[ct] = ("continent", ct, ct)
    for c in others:
        m[c] = ("country", c, c)
    return options, m

def _expand_score_series(wb, kind, name, disp):
    if kind == "country":
        s = wb[wb["country"] == name].groupby(["year"], as_index=False)["score"].mean()
    else:
        s = (wb[wb["continent"] == name].groupby(["year"], as_index=False)["score"].mean())
    if s.empty:
        return pd.DataFrame(columns=["entity","year","score","ys"])
    s["entity"] = disp
    s["ys"] = s["year"].astype(int).astype(str)
    return s[["entity","year","ys","score"]]

def _expand_capex_series(cap, wb, kind, name, disp):
    if kind == "country":
        c = cap[cap["country"] == name].groupby(["year"], as_index=False)["capex"].sum()
    else:
        countries_in = wb[wb["continent"] == name]["country"].unique().tolist()
        c = cap[cap["country"].isin(countries_in)].groupby(["year"], as_index=False)["capex"].sum()
    if c.empty:
        return pd.DataFrame(columns=["entity","year","capex","ys"])
    c["entity"] = disp
    c["ys"] = c["year"].astype(int).astype(str)
    return c[["entity","year","ys","capex"]]

def _responsive_columns(n, max_per_row=4):
    """Yield Streamlit columns in rows of up to max_per_row."""
    for i in range(0, n, max_per_row):
        cols = st.columns(min(max_per_row, n - i), gap="large")
        yield cols, i

# ================= Public entrypoint =================
def render_compare_tab():
    wb      = load_wb()
    wb_avg  = load_wb_avg()  # used for KPI panel
    cap     = load_capex()
    sec     = load_sectors()
    dst     = load_destinations()

    if wb.empty:
        st.info("World Bank data required.")
        st.stop()

    # --- Top bar (compact) ---
    top_l, top_r = st.columns([30, 1], gap="small")
    with top_l:
        st.caption("Benchmarking — multi-country / continent comparison")
    with top_r:
        info_button("compare")
    emit_auto_jump_script()

    # -------- Entities (multiselect up to 6) --------
    wb_years  = sorted(wb["year"].dropna().astype(int).unique().tolist())
    latest_wb_year = max(wb_years) if wb_years else None
    all_countries = sorted(wb.loc[wb["year"] == latest_wb_year, "country"].dropna().unique().tolist()) if latest_wb_year else []
    continents = sorted(wb["continent"].dropna().unique().tolist())
    options_labels, label_map = _build_selection_lists(all_countries, continents)

    sel_labels = st.multiselect(
        "Select countries/continents (up to 6)",
        options=options_labels,
        default=[c for c in ["United States", "France"] if c in label_map][:2] or [c for c in options_labels if c not in HEADERS][:2],
        max_selections=6
    )

    if len(sel_labels) == 0:
        st.info("Pick at least one country or continent.")
        st.stop()

    # Parse selection into structured list, skipping headers
    sel_entities = [label_map[lbl] for lbl in sel_labels if lbl in label_map]  # (kind, name, display)

    # Country-only list for Sectors/Destinations (and also filtered to allowed)
    sel_countries_allowed = [disp for (kind, name, disp) in sel_entities if kind == "country" and disp in SECT_DEST_ALLOWED]

    st.markdown("---")

    # ======================= SCORE — LINE (2023 labels on the RIGHT) + KPI panel =======================

    score_parts = [_expand_score_series(wb, kind, name, disp) for (kind,name,disp) in sel_entities]
    score_df = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame(columns=["entity","year","ys","score"])

    if not score_df.empty:
        # Label only for year 2023, positioned on the right
        score_df["label_2023"] = np.where(
            score_df["year"].eq(2023),
            score_df["score"].map(lambda v: f"{v:.3f}"),
            ""
        )

        st.markdown(
            """
            <div style="font-size:28px; font-weight:800; line-height:1.2; margin:0;">
              Comparative Viability Score Trends (2021–2023)
            </div>
            <div style="color:#6b7280; margin:.35rem 0 1rem;">
              Tracks year-over-year FDI viability scores for selected countries/continents.
            </div>
            """,
            unsafe_allow_html=True
        )

        fig_score = px.line(
            score_df, x="ys", y="score", color="entity",
            color_discrete_sequence=px.colors.qualitative.Safe,
    
        )
        # add markers + right-side labels for 2023
        fig_score.update_traces(
            mode="lines+markers+text",
            text=score_df["label_2023"],
            textposition="middle right",
            cliponaxis=False
        )

        if len(sel_entities) >= 6:
            dash_seq = ["solid","dot","dash","longdash","dashdot","longdashdot"]
            dash_map = {e[2]: dash_seq[i % len(dash_seq)] for i, e in enumerate(sel_entities)}
            for disp, d in dash_map.items():
                fig_score.for_each_trace(lambda tr: tr.update(line=dict(dash=d)) if tr.name == disp else ())

        yvals = score_df["score"].astype(float)
        pad = max((yvals.max() - yvals.min()) * 0.12, 0.002)
        fig_score.update_xaxes(type="category", showgrid=False, title=None)
        fig_score.update_yaxes(visible=False, range=[float(yvals.min()-pad), float(yvals.max()+pad)])
        _style_compare_line(fig_score, unit=None)

        # Title left-align, add room for right labels and KPI
        fig_score.update_layout(
            title_text="",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(automargin=True),
            xaxis=dict(automargin=True),
        )
        
        # Plot + KPI panel (Avg Score, Grade, Continent) — up to 2 cols, 3 rows each
        plot_col, kpi_col = st.columns([5, 1.8], gap="large")
        with plot_col:
            st.plotly_chart(fig_score, use_container_width=True)
        with kpi_col:
            rows_per_col = 3
            num_cols = max(1, min(2, math.ceil(len(sel_entities) / rows_per_col)))
            cols = st.columns(num_cols, gap="large")

            for i, (kind, name, disp) in enumerate(sel_entities):
                target = cols[min(i // rows_per_col, num_cols - 1)]
                with target:
                    if kind == "country":
                        avg_row = wb_avg[wb_avg["country"] == name]
                        sc = float(avg_row["avg_score"].mean()) if not avg_row.empty else np.nan
                        gr = (avg_row["grade"].dropna().astype(str).iloc[0]
                              if (not avg_row.empty and avg_row["grade"].notna().any()) else "-")
                        cont_series = wb.loc[wb["country"] == name, "continent"].dropna()
                        cont = cont_series.iloc[-1] if not cont_series.empty else "-"
                        st.markdown(f"**{disp}**")
                        st.markdown(f"**Avg Score:** {sc:.3f} &nbsp;&nbsp; **Grade:** {gr}" if pd.notna(sc) else "**Avg Score:** –")
                        st.markdown(f"**Continent:** {cont}")
                    else:
                        # Continent selected: show ONLY Avg Score
                        sc = float(wb.loc[wb["continent"] == name, "score"].mean())
                        st.markdown(f"**{disp}**")
                        st.markdown(f"**Avg Score:** {sc:.3f}" if pd.notna(sc) else "**Avg Score:** –")

                    within_col_idx = i % rows_per_col
                    is_last_item = (i == len(sel_entities) - 1)
                    if (within_col_idx < rows_per_col - 1) and not is_last_item:
                        st.markdown("<hr style='margin:8px 0; opacity:.25'>", unsafe_allow_html=True)
    else:
        st.info("No score data for selection.")

    st.markdown("---")

    # ======================= CAPEX — LINE (absolute only) =======================

    cap_parts = [_expand_capex_series(cap, wb, kind, name, disp) for (kind,name,disp) in sel_entities]
    cap_df = pd.concat(cap_parts, ignore_index=True) if cap_parts else pd.DataFrame(columns=["entity","year","ys","capex"])

    if not cap_df.empty:
        st.markdown(
            """
            <div style="font-size:28px; font-weight:800; line-height:1.2; margin:0;">
              Comparative Capex Trends (2021–2024)
            </div>
            <div style="color:#6b7280; margin:.35rem 0 1rem;">
              Tracks year-over-year CAPEX for selected countries.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        fig_cap = px.line(
            cap_df, x="ys", y="capex", color="entity", markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )

        if len(sel_entities) >= 6:
            dash_seq = ["solid","dot","dash","longdash","dashdot","longdashdot"]
            dash_map = {e[2]: dash_seq[i % len(dash_seq)] for i, e in enumerate(sel_entities)}
            for disp, d in dash_map.items():
                fig_cap.for_each_trace(lambda tr: tr.update(line=dict(dash=d)) if tr.name == disp else ())

        _style_compare_line(fig_cap, unit="$B")
        fig_cap.update_layout(
            title_text="",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(automargin=True),
            xaxis=dict(automargin=True),
        )
        st.plotly_chart(fig_cap, use_container_width=True)
    else:
        st.info("No CAPEX data for selection.")

    # ======================= Sectors — ALL eligible selected countries =======================
    st.markdown("---")
    
    if not sel_countries_allowed:
        st.caption("Select one or more **top**/allowed countries to see sector KPIs.")
    elif sec.empty:
        st.caption("No sectors data available.")
    else:
        # 1) Reserve a spot ABOVE the filters for the title/subtitle
        sectors_title_ph = st.empty()
    
        # 2) Filters (these render BELOW the placeholder)
        sectors_list = sorted(sec["sector"].dropna().unique().tolist())
        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            sector_opt = st.selectbox("Sector", sectors_list, index=0, key="cmp_sector")
        with c2:
            sector_metric = st.selectbox("Metric", ["Companies","Jobs Created","Capex","Projects"], index=0, key="cmp_sector_metric")
    
        # 3) Build dynamic subtitle parts
        metric_label_map = {
            "Companies": "number of companies",
            "Jobs Created": "number of jobs created",
            "Capex": "capex (USD m)",
            "Projects": "number of projects",
        }
        if len(sel_countries_allowed) == 0:
            countries_text = "the selected countries"
        elif len(sel_countries_allowed) == 1:
            countries_text = sel_countries_allowed[0]
        elif len(sel_countries_allowed) == 2:
            countries_text = f"{sel_countries_allowed[0]} and {sel_countries_allowed[1]}"
        else:
            countries_text = ", ".join(sel_countries_allowed[:-1]) + f", and {sel_countries_allowed[-1]}"
    
        # 4) Fill the placeholder ABOVE the filters (note: f-string so braces evaluate)
        sectors_title_ph.markdown(
            f"""
            <h3 style="margin:0; font-weight:800 !important; line-height:1.2; font-size:28px;">
              Sectoral Benchmarking
            </h3>
            <div style="color:#6b7280; margin:.35rem 0 1rem;">
              Compares the {metric_label_map.get(sector_metric, sector_metric.lower())} in a chosen sector across {countries_text}.
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # 5) KPIs
        metric_map = {"Companies":"companies","Jobs Created":"jobs_created","Capex":"capex","Projects":"projects"}
        col = metric_map[sector_metric]
    
        n = len(sel_countries_allowed)
        for cols, i in _responsive_columns(n, max_per_row=4):
            for j, country in enumerate(sel_countries_allowed[i:i+len(cols)]):
                with cols[j]:
                    val = float(sec.loc[(sec["country"] == country) & (sec["sector"] == sector_opt), col].sum())
                    st.markdown(f"**{country}**")
                    _kpi(f"{sector_opt} • {sector_metric}", val, "USD m" if col == "capex" else "")

    # ======================= Destinations — ALL eligible selected countries =======================
    st.markdown("---")

    if not sel_countries_allowed:
        st.caption("Select one or more **top**/allowed countries to see destination KPIs.")
    elif dst.empty:
        st.caption("No destinations data available.")
    else:
        if sel_countries_allowed:
            if len(sel_countries_allowed) == 1:
                sources_text = sel_countries_allowed[0]
            elif len(sel_countries_allowed) == 2:
                sources_text = f"{sel_countries_allowed[0]} and {sel_countries_allowed[1]}"
            else:
                sources_text = ", ".join(sel_countries_allowed[:-1]) + f", and {sel_countries_allowed[-1]}"
        else:
            sources_text = "the selected countries"

        st.markdown(
            f"""
            <h3 style="margin:0; font-weight:800 !important; line-height:1.2; font-size:28px;">
              Outbound FDI Destinations
            </h3>
            <div style="color:#6b7280; margin:.35rem 0 1rem;">
              Benchmarks top outward investment destinations from {sources_text}.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Union of destinations for the selected allowed sources
        all_dests = set()
        for src_cty in sel_countries_allowed:
            all_dests.update(dst.loc[dst["source_country"] == src_cty, "destination_country"].dropna().unique().tolist())
        union_dests = sorted(all_dests)

        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            dest_country = st.selectbox("Destination country", union_dests, index=0 if union_dests else None, key="cmp_dest_country")
        with c2:
            dest_metric = st.selectbox("Metric", ["Companies","Jobs Created","Capex","Projects"], index=0, key="cmp_dest_metric")

        metric_map = {"Companies":"companies","Jobs Created":"jobs_created","Capex":"capex","Projects":"projects"}
        col = metric_map[dest_metric]

        if not union_dests:
            st.caption("No destination options for the current selection.")
        else:
            n = len(sel_countries_allowed)
            for cols, i in _responsive_columns(n, max_per_row=4):
                for j, country in enumerate(sel_countries_allowed[i:i+len(cols)]):
                    with cols[j]:
                        val = float(dst.loc[
                            (dst["source_country"] == country) &
                            (dst["destination_country"] == dest_country), col
                        ].sum())
                        st.markdown(f"**{country}**")
                        _kpi(f"{country} → {dest_country} • {dest_metric}", val, "USD m" if col == "capex" else "")
