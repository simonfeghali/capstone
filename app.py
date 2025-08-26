# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import quote
from urllib.error import URLError, HTTPError

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FDI Analytics • Scoring", layout="wide")
st.title("FDI Analytics Dashboard")
st.caption("Scoring • (World Bank–based)")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .metric-value { font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# DATA (GitHub raw)
# ──────────────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
WB_FILE  = "world_bank_data_with_scores_and_continent (1).csv"

def gh_raw_url(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    for c in cands:
        for col in cols:
            if c.lower() in col.lower(): return col
    return None

@st.cache_data(show_spinner=True)
def load_scoring() -> pd.DataFrame:
    url = gh_raw_url(WB_FILE)
    try:
        df = pd.read_csv(url)
    except (URLError, HTTPError, FileNotFoundError) as e:
        raise RuntimeError(f"Could not fetch {WB_FILE} from GitHub: {e}")

    country = find_col(df.columns, "country", "country_name", "Country Name")
    year    = find_col(df.columns, "year")
    cont    = find_col(df.columns, "continent", "region")
    score   = find_col(df.columns, "score", "viability_score", "composite_score")
    grade   = find_col(df.columns, "grade", "letter_grade")

    req = {"country": country, "year": year, "continent": cont}
    missing = [k for k,v in req.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns in {WB_FILE}: {missing} (found {list(df.columns)})")

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

    order = pd.CategoricalDtype(categories=["A+", "A", "B", "C", "D"], ordered=True)
    df["grade"] = df["grade"].astype(str).str.strip()
    df["grade"] = df["grade"].where(df["grade"].isin(order.categories), other=np.nan).astype(order)

    return df

wb = load_scoring()

# ──────────────────────────────────────────────────────────────────────────────
# FILTERS (Year • Continent • Country)
# ──────────────────────────────────────────────────────────────────────────────
years = sorted(wb["year"].dropna().astype(int).unique().tolist())
c1, c2, c3 = st.columns([1, 1, 2], gap="small")

with c1:
    sel_year = st.selectbox("Year", years, index=0)

with c2:
    conts = ["All"] + sorted(wb.loc[wb["year"] == sel_year, "continent"].dropna().unique().tolist())
    sel_cont = st.selectbox("Continent", conts, index=0)

with c3:
    wb_scope = wb[wb["year"] == sel_year].copy()
    if sel_cont != "All":
        wb_scope = wb_scope[wb_scope["continent"] == sel_cont]
    countries = ["All"] + sorted(wb_scope["country"].unique().tolist())
    sel_country = st.selectbox("Country", countries, index=0)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["year"] == sel_year].copy()
    if sel_cont != "All":
        out = out[out["continent"] == sel_cont]
    if sel_country != "All":
        out = out[out["country"] == sel_country]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# SCORING (single tab)
# ──────────────────────────────────────────────────────────────────────────────
tab_scoring, = st.tabs(["Scoring"])

with tab_scoring:
    where_title = sel_country if sel_country != "All" else (sel_cont if sel_cont != "All" else "Worldwide")
    st.subheader(where_title)

    # ── TOP ROW: (left) YoY Viability line • (right) Global map
    t1, t2 = st.columns([1, 2], gap="large")

    # Year-over-Year Viability Score
    with t1:
        if sel_country != "All":
            yoy_df = (wb[wb["country"] == sel_country]
                      .groupby("year", as_index=False)["score"].mean()
                      .sort_values("year"))
            title = f"Year-over-Year Viability Score — {sel_country}"
        elif sel_cont != "All":
            yoy_df = (wb[wb["continent"] == sel_cont]
                      .groupby("year", as_index=False)["score"].mean()
                      .sort_values("year"))
            title = f"Year-over-Year Viability Score — {sel_cont}"
        else:
            yoy_df = wb.groupby("year", as_index=False)["score"].mean().sort_values("year")
            title = "Year-over-Year Viability Score — Global"

        fig_line = px.line(
            yoy_df, x="year", y="score", markers=True,
            labels={"year": "", "score": "Mean score"},
            title=title
        )
        fig_line.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=340)
        st.plotly_chart(fig_line, use_container_width=True)

    # Global Performance Map (selected year + filters)
    with t2:
        map_df = apply_filters(wb)[["country", "score"]].copy()
        if map_df.empty:
            st.info("No data for this selection.")
        else:
            fig_map = px.choropleth(
                map_df,
                locations="country",
                locationmode="country names",
                color="score",
                color_continuous_scale="Blues",
                title="Global Performance Map",
            )
            fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410)
            st.plotly_chart(fig_map, use_container_width=True)

    # ── BOTTOM: charts by default; KPIs if a country is selected
    if sel_country == "All":
        b1, b2, b3 = st.columns([1.2, 1, 1.2], gap="large")

        # Top 10 Performing Countries
        with b1:
            top_base = apply_filters(wb)[["country", "score"]].dropna()
            top10 = top_base.sort_values("score", ascending=False).head(10)
            if top10.empty:
                st.info("No countries available for Top 10 with this filter.")
            else:
                fig_top = px.bar(
                    top10.sort_values("score"),
                    x="score", y="country", orientation="h",
                    labels={"score": "", "country": ""},
                    title="Top 10 Performing Countries",
                    text=top10["score"].round(3),
                )
                fig_top.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                fig_top.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_top, use_container_width=True)

        # Grade Distribution (donut)
        with b2:
            donut_base = apply_filters(wb)
            if donut_base.empty or donut_base["grade"].isna().all():
                st.info("No grade data for this selection.")
            else:
                grades = ["A+", "A", "B", "C", "D"]
                donut = (
                    donut_base.assign(grade=donut_base["grade"].astype(str))
                              .loc[lambda d: d["grade"].isin(grades)]
                              .groupby("grade", as_index=False)["country"].nunique()
                              .rename(columns={"country": "count"})
                ).set_index("grade").reindex(grades, fill_value=0).reset_index()

                fig_donut = px.pie(
                    donut, names="grade", values="count", hole=0.55, title="Grade Distribution"
                )
                fig_donut.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420, showlegend=True)
                st.plotly_chart(fig_donut, use_container_width=True)

        # Continent Viability Score (bar)
        with b3:
            cont_base = wb[wb["year"] == sel_year].copy()
            if sel_cont != "All":
                cont_base = cont_base[cont_base["continent"] == sel_cont]
            cont_bar = (cont_base.groupby("continent", as_index=False)["score"].mean()
                                  .sort_values("score", ascending=True))
            if cont_bar.empty:
                st.info("No continent data for this selection.")
            else:
                fig_cont = px.bar(
                    cont_bar, x="score", y="continent", orientation="h",
                    labels={"score": "", "continent": ""},
                    title="Continent Viability Score",
                    text=cont_bar["score"].round(3),
                )
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
                fig_cont.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_cont, use_container_width=True)

    else:
        # ── KPI mode when a *country* is selected
        # Country’s score & grade for the selected year
        ctry_rows = wb[(wb["year"] == sel_year) & (wb["country"] == sel_country)]
        country_score = float(ctry_rows["score"].mean()) if not ctry_rows.empty else np.nan
        country_grade = ctry_rows["grade"].astype(str).dropna().iloc[0] if not ctry_rows.empty and ctry_rows["grade"].notna().any() else "-"

        # Determine that country's continent, then continent average score (same year)
        if not ctry_rows.empty and ctry_rows["continent"].notna().any():
            ctry_cont = ctry_rows["continent"].dropna().iloc[0]
        else:
            ctry_cont = sel_cont if sel_cont != "All" else None

        if ctry_cont:
            cont_avg = float(wb[(wb["year"] == sel_year) & (wb["continent"] == ctry_cont)]["score"].mean())
        else:
            cont_avg = np.nan

        k1, k2, k3 = st.columns(3, gap="large")
        with k1:
            st.metric("Country Score", "-" if np.isnan(country_score) else f"{country_score:,.3f}")
        with k2:
            st.metric("Grade", country_grade)
        with k3:
            label = f"{ctry_cont} Avg Score" if ctry_cont else "Continent Avg Score"
            st.metric(label, "-" if np.isnan(cont_avg) else f"{cont_avg:,.3f}")
