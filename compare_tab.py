# compare_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.parse import quote

RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "wb":  "world_bank_data_with_scores_and_continent.csv",
    "cap": "capex_EDA_cleaned_filled.csv",
}

def _raw(fname: str) -> str: return f"{RAW_BASE}/{quote(fname)}"

@st.cache_data
def _load_wb():
    df = pd.read_csv(_raw(FILES["wb"]))
    df = df.rename(columns={
        "Country Name":"country","country_name":"country","Country":"country",
        "year":"year","continent":"continent","region":"continent",
        "score":"score","viability_score":"score","composite_score":"score",
        "grade":"grade","letter_grade":"grade"
    })
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df[["country","year","continent","score","grade"]]

@st.cache_data
def _load_capex():
    df = pd.read_csv(_raw(FILES["cap"]))
    # reshape wide→long (year columns)
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c))==4]
    src = next((c for c in ("Source Country","Country","Source Co") if c in df.columns), None)
    grade = "Grade" if "Grade" in df.columns else None
    m = df.melt(id_vars=[c for c in [src, grade] if c],
                value_vars=year_cols, var_name="year", value_name="capex")
    m = m.rename(columns={src:"country", grade:"grade"})
    m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
    # numeric capex
    m["capex"] = (m["capex"].astype(str).str.replace(",", "", regex=False)
                              .str.replace(r"[^\d\.\-]", "", regex=True)
                              .astype(float))
    return m[["country","year","capex","grade"]]

def render_compare_tab():
    wb = _load_wb()
    cap = _load_capex()

    years = sorted(set(wb["year"].dropna().astype(int)) | set(cap["year"].dropna().astype(int)))
    year_opts = ["All"] + years

    c1, c2, c3 = st.columns([1,1,2], gap="small")
    with c1:
        year_any = st.selectbox("Year", year_opts, index=0, key="cmp_year")
    countries = sorted(wb.loc[wb["year"]==max(years), "country"].dropna().unique())
    with c2:
        a = st.selectbox("Country A", countries, index=0, key="cmp_a")
    with c3:
        b = st.selectbox("Country B", countries, index=min(1, len(countries)-1), key="cmp_b")

    if a == b:
        st.warning("Choose two different countries.")
        return

    def _score_meta(country):
        s = wb[wb["country"]==country]
        cont = s["continent"].dropna().iloc[0] if not s.empty else "-"
        if year_any == "All":
            score = float(s["score"].mean()) if not s.empty else np.nan
            grade = "-"
        else:
            row = s[s["year"]==int(year_any)]
            score = float(row["score"].mean()) if not row.empty else np.nan
            grade = row["grade"].astype(str).dropna().iloc[0] if not row.empty and row["grade"].notna().any() else "-"
        return score, grade, cont

    def _kpi(title, val, sub=""):
        st.markdown(f"""
        <div style="text-align:center;padding:14px 0">
          <div style="font-weight:700">{title}</div>
          <div style="font-size:48px;font-weight:800;line-height:1;margin:8px 0 2px">{'-' if pd.isna(val) else f'{val:,.1f}'}</div>
          <div style="opacity:.75">{sub}</div>
        </div>""", unsafe_allow_html=True)

    colA, colB = st.columns(2, gap="large")
    for country, col in [(a, colA), (b, colB)]:
        with col:
            st.subheader(country)
            sc, gr, cont = _score_meta(country)
            k1,k2,k3 = st.columns(3)
            with k1: _kpi("Score", sc)
            with k2: st.markdown(f"**Grade:** {gr}")
            with k3: st.markdown(f"**Continent:** {cont}")

            if year_any == "All":
                s_trend = wb[wb["country"]==country].groupby("year", as_index=False)["score"].mean()
                if not s_trend.empty:
                    s_trend["ys"] = s_trend["year"].astype(int).astype(str)
                    fig = px.line(s_trend, x="ys", y="score", markers=True,
                                  labels={"ys":"","score":"Mean score"},
                                  title=f"{country} • Viability Score Trend")
                    fig.update_xaxes(type="category", showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=300)
                    st.plotly_chart(fig, use_container_width=True)
            # CAPEX
            cap_scope = cap[cap["country"]==country]
            if year_any != "All":
                cap_scope = cap_scope[cap_scope["year"]==int(year_any)]
                _kpi(f"{country} CAPEX — {year_any}", cap_scope["capex"].sum(), "$B")
            else:
                t = cap_scope.groupby("year", as_index=False)["capex"].sum()
                if not t.empty:
                    t["ys"] = t["year"].astype(int).astype(str)
                    fig = px.line(t, x="ys", y="capex", markers=True,
                                  labels={"ys":"","capex":"CAPEX ($B)"},
                                  title=f"{country} • CAPEX Trend")
                    fig.update_xaxes(type="category", showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=300)
                    st.plotly_chart(fig, use_container_width=True)

            # CAPEX by Grade
            if year_any != "All":
                gb = (cap[(cap["country"]==country) & (cap["year"]==int(year_any))]
                      .assign(grade=lambda d: d["grade"].astype(str))
                      .groupby("grade", as_index=False)["capex"].sum())
                grades = ["A+","A","B","C","D"]
                gb = gb.set_index("grade").reindex(grades, fill_value=0).reset_index()
                fig = px.bar(gb, x="capex", y="grade", orientation="h",
                             title=f"{country} • CAPEX by Grade — {year_any}",
                             labels={"capex":"","grade":""},
                             color="capex", color_continuous_scale="Blues")
                fig.update_coloraxes(showscale=False)
                fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=320)
                st.plotly_chart(fig, use_container_width=True)

    # quick head-to-head
    st.markdown("### Head-to-Head")
    scA, grA, _ = _score_meta(a)
    scB, grB, _ = _score_meta(b)
    capA = cap[(cap["country"]==a) & ((cap["year"]==int(year_any)) if year_any!='All' else True)]["capex"].sum()
    capB = cap[(cap["country"]==b) & ((cap["year"]==int(year_any)) if year_any!='All' else True)]["capex"].sum()
    d1,d2,d3 = st.columns(3)
    with d1: _kpi("Score Δ (A−B)", (scA if not pd.isna(scA) else 0) - (scB if not pd.isna(scB) else 0))
    with d2: st.markdown(f"**Grade A/B:** `{grA}` / `{grB}`")
    with d3: _kpi("CAPEX Δ (A−B)", capA - capB, "$B")
