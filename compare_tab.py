# compare_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import re
from overview import info_button, emit_auto_jump_script

# ================= Config =================
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":       "world_bank_data_with_scores_and_continent.csv",
    "wb_avg":   "world_bank_data_average_scores_and_grades.csv",  # used when Year = All
    "cap":      "capex_EDA_cleaned_filled.csv",
    "sectors":  "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
}

SECT_DEST_ALLOWED = {
    "Canada","China","France","Germany","Japan","Netherlands",
    "South Korea","United Arab Emirates","United Kingdom","United States"
}

TOP_COUNTRIES = [
    "United States","United Kingdom","Germany","France","China",
    "Japan","South Korea","Canada","Netherlands","United Arab Emirates",
    "Bahrain","Kuwait","Qatar","Oman","Saudi Arabia"
]

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
    """
    Apply consistent styling/hover to compare tab line charts.
    unit=None -> Score; unit="$B" -> CAPEX
    """
    if unit == "$B":
        htmpl = "Year: %{x}<br>FDI: %{y:.4f} $B<extra></extra>"
    else:
        htmpl = "Year: %{x}<br>Score: %{y:.3f}<extra></extra>"

    fig.update_traces(
        mode="lines+markers",
        hovertemplate=htmpl,
        text=None, texttemplate=None, textposition=None
    )
    fig.update_xaxes(type="category", showgrid=False, title=None)
    fig.update_yaxes(showgrid=False, title=None)
    fig.update_layout(hovermode="closest")  # removes dotted guideline
    return fig

# === Canonicalize country names (same mapping you use elsewhere) ===
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
    """Averages file used when Year = All."""
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
    # Convert CAPEX from millions to billions
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

    # canonicalize both source and destination country names
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
    # compose labels, ordered: top countries -> continents -> other countries
    top = [c for c in TOP_COUNTRIES if c in all_countries]
    others = [c for c in all_countries if c not in top]
    lbl_top = [f"⭐ {c}" for c in top]
    lbl_cont = [f"🌍 {ct}" for ct in continents]
    lbl_others = others  # plain labels

    # mapping label -> (kind, name, display)
    m = {}
    for c in top:
        m[f"⭐ {c}"] = ("country", c, c)
    for ct in continents:
        m[f"🌍 {ct}"] = ("continent", ct, f"{ct} (avg/sum)")
    for c in others:
        m[c] = ("country", c, c)

    options = lbl_top + lbl_cont + lbl_others
    return options, m

def _expand_score_series(wb, kind, name, disp):
    if kind == "country":
        s = wb[wb["country"] == name].groupby(["year"], as_index=False)["score"].mean()
    else:
        # continent aggregate = mean score across countries in that continent
        s = (wb[wb["continent"] == name]
             .groupby(["year"], as_index=False)["score"].mean())
    if s.empty:
        return pd.DataFrame(columns=["entity","year","score","ys"])
    s["entity"] = disp
    s["ys"] = s["year"].astype(int).astype(str)
    return s[["entity","year","ys","score"]]

def _expand_capex_series(cap, wb, kind, name, disp):
    if kind == "country":
        c = cap[cap["country"] == name].groupby(["year"], as_index=False)["capex"].sum()
    else:
        # for continent: sum CAPEX of all countries in that continent
        countries_in = wb[wb["continent"] == name]["country"].unique().tolist()
        c = cap[cap["country"].isin(countries_in)].groupby(["year"], as_index=False)["capex"].sum()
    if c.empty:
        return pd.DataFrame(columns=["entity","year","capex","ys"])
    c["entity"] = disp
    c["ys"] = c["year"].astype(int).astype(str)
    return c[["entity","year","ys","capex"]]

# ================= Public entrypoint =================
def render_compare_tab():
    wb      = load_wb()
    wb_avg  = load_wb_avg()
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

    # -------- Controls: Year & Entities (multiselect up to 6) --------
    wb_years  = sorted(wb["year"].dropna().astype(int).unique().tolist())
    cap_years = sorted(cap["year"].dropna().astype(int).unique().tolist())
    years_all = sorted(set(wb_years).union(cap_years))

    c1, c2 = st.columns([1, 3], gap="small")
    with c1:
        year_opts = ["All"] + years_all
        year_any = st.selectbox("Year", year_opts, index=0, key="cmp_year")

    latest_wb_year = max(wb_years) if wb_years else None
    all_countries = sorted(wb.loc[wb["year"] == latest_wb_year, "country"].dropna().unique().tolist()) if latest_wb_year else []
    continents = sorted(wb["continent"].dropna().unique().tolist())
    options_labels, label_map = _build_selection_lists(all_countries, continents)

    with c2:
        default_labels = [lbl for lbl in options_labels if lbl.startswith("⭐ ") and lbl[2:] in {"United States","France"}][:2]
        if not default_labels:
            # fallback to first two options if stars missing
            default_labels = options_labels[:2]
        sel_labels = st.multiselect("Select countries/continents (up to 6)",
                                    options=options_labels,
                                    default=default_labels,
                                    max_selections=6)

    if len(sel_labels) == 0:
        st.info("Pick at least one country or continent.")
        st.stop()

    # Parse selection into structured list
    sel_entities = [label_map[lbl] for lbl in sel_labels if lbl in label_map]  # list of tuples (kind,name,display)

    # For sectors/destinations we only consider first two selected countries (not continents)
    only_countries = [disp for kind,name,disp in sel_entities if kind == "country"]
    if len(only_countries) >= 1:
        a = only_countries[0]
    else:
        a = None
    if len(only_countries) >= 2:
        b = only_countries[1]
    else:
        b = None
    allowed_pair = bool(a and b and (a in SECT_DEST_ALLOWED) and (b in SECT_DEST_ALLOWED))

    st.markdown("---")

    # ======================= SCORE =======================
    st.subheader("Score & Grade")

    # View options
    col_vm, col_lb = st.columns([1,1])
    with col_vm:
        view_mode = st.radio("View", options=["Overlay", "Heatmap table"], index=0, horizontal=True)
    with col_lb:
        show_all_labels = st.checkbox("Show labels on every point", value=(len(sel_entities) <= 3))

    # Build combined score df
    score_parts = [ _expand_score_series(wb, kind, name, disp) for (kind,name,disp) in sel_entities ]
    score_df = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame(columns=["entity","year","ys","score"])

    if not score_df.empty:
        if view_mode == "Overlay":
            last_year = score_df["year"].max()
            score_df["label"] = np.where(score_df["year"].eq(last_year),
                                         score_df["score"].map(lambda v: f"{v:.3f}"), "")

            fig_score = px.line(
                score_df, x="ys", y="score", color="entity", markers=True,
                text=("score" if show_all_labels else "label"),
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Viability Score Trend"
            )

            if len(sel_entities) >= 6:
                dash_seq = ["solid","dot","dash","longdash","dashdot","longdashdot"]
                dash_map = {e[2]: dash_seq[i % len(dash_seq)] for i, e in enumerate(sel_entities)}
                for disp, d in dash_map.items():
                    fig_score.for_each_trace(lambda tr: tr.update(line=dict(dash=d)) if tr.name == disp else ())

            fig_score.update_xaxes(type="category", showgrid=False, title=None)
            yvals = score_df["score"].astype(float)
            pad = max((yvals.max() - yvals.min()) * 0.12, 0.002)
            fig_score.update_yaxes(visible=False, range=[float(yvals.min()-pad), float(yvals.max()+pad)])
            fig_score.update_traces(
                textposition=("top center" if show_all_labels else "middle right"),
                cliponaxis=False,
                hovertemplate="Series: %{fullData.name}<br>Year: %{x}<br>Score: %{y:.3f}<extra></extra>"
            )
            fig_score.update_layout(
                legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
                margin=dict(l=10, r=200, t=60, b=30),
                height=380,
                legend_title_text=None
            )

            # KPI stack at right (latest available year per series)
            plot_col, kpi_col = st.columns([5, 1.8], gap="large")
            with plot_col:
                st.plotly_chart(fig_score, use_container_width=True)
            with kpi_col:
                kyear = last_year
                for i, (kind, name, disp) in enumerate(sel_entities):
                    if kind == "country":
                        row = wb[(wb["country"] == name) & (wb["year"] == kyear)]
                        sc = row["score"].mean() if not row.empty else np.nan
                        gr = (row["grade"].dropna().astype(str).iloc[0]
                              if (not row.empty and row["grade"].notna().any()) else "-")
                        cont = (row["continent"].dropna().iloc[0]
                                if (not row.empty and row["continent"].notna().any()) else "-")
                    else:
                        sc = wb.loc[(wb["continent"] == name) & (wb["year"] == kyear), "score"].mean()
                        gr = "-"  # undefined for continent aggregate
                        cont = name
                    st.markdown(f"**{disp}**")
                    st.markdown(f"**Score:** {sc:.3f} &nbsp;&nbsp; **Grade:** {gr}" if pd.notna(sc) else "**Score:** –")
                    st.markdown(f"**Continent:** {cont}")
                    if i < len(sel_entities)-1:
                        st.markdown("<hr style='margin:8px 0; opacity:.25'>", unsafe_allow_html=True)

        else:
            # ---------- Score & Grade: Heatmap table ----------
            s_list = []
            for kind, name, disp in sel_entities:
                if kind == "country":
                    s = (wb[wb["country"] == name]
                         .groupby(["year"], as_index=False)
                         .agg(score=("score","mean"),
                              grade=("grade", lambda x: x.dropna().astype(str).iloc[0] if x.notna().any() else "-")))
                else:
                    s = (wb[wb["continent"] == name]
                         .groupby(["year"], as_index=False)
                         .agg(score=("score","mean")))
                    s["grade"] = "-"  # no grade for continents
                s["entity"] = disp
                s_list.append(s)

            s = pd.concat(s_list, ignore_index=True) if s_list else pd.DataFrame(columns=["entity","year","score","grade"])
            if s.empty:
                st.info("No score data for the selection.")
            else:
                s["year"] = s["year"].astype(int)
                years = sorted(s["year"].unique().tolist())

                p_score = s.pivot(index="entity", columns="year", values="score").reindex(index=[e[2] for e in sel_entities])
                text_map = (s
                            .assign(txt=lambda d: d["score"].map(lambda v: f"{v:.3f}") + " (" + d["grade"].astype(str) + ")")
                            .pivot(index="entity", columns="year", values="txt")
                            .reindex(index=[e[2] for e in sel_entities]))

                fig_hm = px.imshow(
                    p_score.to_numpy(),
                    x=years,
                    y=p_score.index.tolist(),
                    color_continuous_scale="Blues",
                    aspect="auto",
                    origin="lower"
                )
                fig_hm.update_traces(
                    text=text_map.to_numpy(),
                    texttemplate="%{text}",
                    hovertemplate="Series: %{y}<br>Year: %{x}<br>Score: %{z:.3f}<extra></extra>"
                )
                fig_hm.update_layout(
                    title="Score & Grade (Heatmap)",
                    margin=dict(l=10, r=10, t=60, b=10),
                    height=140 + 40 * len(sel_entities),
                    xaxis=dict(side="top", showgrid=False, title=None),
                    yaxis=dict(showgrid=False, title=None),
                    coloraxis_colorbar=dict(title="")
                )
                fig_hm.update_xaxes(showgrid=False, type="category")
                fig_hm.update_yaxes(showgrid=False)
                st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No score data for selection.")

    st.markdown("---")

    # ======================= CAPEX =======================
    st.subheader("CAPEX")

    cap_parts = [ _expand_capex_series(cap, wb, kind, name, disp) for (kind,name,disp) in sel_entities ]
    cap_df = pd.concat(cap_parts, ignore_index=True) if cap_parts else pd.DataFrame(columns=["entity","year","ys","capex"])

    if not cap_df.empty:
        scale = st.radio("Scale", options=["Absolute ($B)", "Index (base=100)", "Log"],
                         index=0, horizontal=True)

        plot_df = cap_df.copy()
        if scale == "Index (base=100)":
            plot_df["capex"] = (plot_df.sort_values("year")
                                .groupby("entity")["capex"]
                                .transform(lambda s: (s / s.iloc[0])*100 if s.iloc[0] else np.nan))
            title_suffix = "Index=100"
        elif scale == "Log":
            plot_df["capex"] = np.log10(plot_df["capex"].clip(lower=1e-6))
            title_suffix = "log10"
        else:
            title_suffix = "$B"

        if view_mode == "Overlay":
            fig_cap = px.line(
                plot_df, x="ys", y="capex", color="entity", markers=True,
                color_discrete_sequence=px.colors.qualitative.Safe,
                title=f"CAPEX Trend ({title_suffix})"
            )
            if len(sel_entities) >= 6:
                dash_seq = ["solid","dot","dash","longdash","dashdot","longdashdot"]
                dash_map = {e[2]: dash_seq[i % len(dash_seq)] for i, e in enumerate(sel_entities)}
                for disp, d in dash_map.items():
                    fig_cap.for_each_trace(lambda tr: tr.update(line=dict(dash=d)) if tr.name == disp else ())

            fig_cap.update_xaxes(type="category", showgrid=False, title=None)
            fig_cap.update_yaxes(showgrid=False, title=None)
            fig_cap.update_traces(hovertemplate="Series: %{fullData.name}<br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>")
            fig_cap.update_layout(
                legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
                margin=dict(l=10, r=200, t=60, b=30),
                height=380,
                legend_title_text=None
            )
            st.plotly_chart(fig_cap, use_container_width=True)

        else:
            c = plot_df.copy()
            if c.empty:
                st.info("No CAPEX data for the selection.")
            else:
                c["year"] = c["year"].astype(int)
                years = sorted(c["year"].unique().tolist())

                p_cap = c.pivot(index="entity", columns="year", values="capex").reindex(index=[e[2] for e in sel_entities])

                def fmt_val(v):
                    if pd.isna(v): return "–"
                    if scale == "Absolute ($B)":
                        return f"{v:,.2f}"
                    if scale.startswith("Index"):
                        return f"{v:,.0f}"
                    return f"{v:.2f}"  # log

                text_cap = p_cap.applymap(fmt_val)

                fig_cap_hm = px.imshow(
                    p_cap.to_numpy(),
                    x=years,
                    y=p_cap.index.tolist(),
                    color_continuous_scale="Blues",
                    aspect="auto",
                    origin="lower"
                )
                fig_cap_hm.update_traces(
                    text=text_cap.to_numpy(),
                    texttemplate="%{text}",
                    hovertemplate="Series: %{y}<br>Year: %{x}<br>Value: %{z:.2f}<extra></extra>"
                )
                fig_cap_hm.update_layout(
                    title=f"CAPEX Heatmap ({'log10' if scale=='Log' else ('Index=100' if scale.startswith('Index') else '$B')})",
                    xaxis_title="Year",
                    yaxis_title=None,
                    margin=dict(l=10, r=10, t=60, b=10),
                    height=140 + 40 * len(sel_entities)
                )
                fig_cap_hm.update_xaxes(showgrid=False, type="category")
                fig_cap_hm.update_yaxes(showgrid=False)
                st.plotly_chart(fig_cap_hm, use_container_width=True)
    else:
        st.info("No CAPEX data for selection.")

    # ======================= Sectors (first two selections that are countries only) =======================
    if a and b and allowed_pair:
        st.markdown("---")
        st.subheader("Sectors")
        if sec.empty:
            st.caption("No sectors data available.")
        else:
            sectors_list = sorted(sec["sector"].dropna().unique().tolist())
            c1, c2 = st.columns([1, 1], gap="small")
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
                st.markdown(f"**{a}**")
                _kpi(f"{sector_opt} • {sector_metric}", valA, "USD m" if col == "capex" else "")
            with right:
                st.markdown(f"**{b}**")
                _kpi(f"{sector_opt} • {sector_metric}", valB, "USD m" if col == "capex" else "")

    # ======================= Destinations (first two selections that are countries only) =======================
    if a and b and allowed_pair:
        st.markdown("---")
        st.subheader("Destinations")
        if dst.empty:
            st.caption("No destinations data available.")
        else:
            destsA = sorted(dst.loc[dst["source_country"] == a, "destination_country"].dropna().unique().tolist())
            destsB = sorted(dst.loc[dst["source_country"] == b, "destination_country"].dropna().unique().tolist())
            union_dests = sorted(set(destsA).union(destsB))

            c1, c2 = st.columns([1, 1], gap="small")
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
                st.markdown(f"**{a}**")
                _kpi(f"{a} → {dest_country} • {dest_metric}", valA, "USD m" if col == "capex" else "")
            with right:
                st.markdown(f"**{b}**")
                _kpi(f"{b} → {dest_country} • {dest_metric}", valB, "USD m" if col == "capex" else "")
