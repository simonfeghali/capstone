# overview.py
# -----------------------------------------------------------------------------
# Overview tab for the FDI & Viability dashboard.
# - Business-first explanations for every tab / plot
# - Deep technical notes (weights, methodology, forecasting details)
# - In-page anchors so other tabs (or external links) can jump here:
#     e.g. add '?jump=score_trend' to the URL to auto-scroll to that section.
# -----------------------------------------------------------------------------

from __future__ import annotations
import streamlit as st
import pandas as pd
from streamlit.components.v1 import html as st_html

# Map short keys -> (section title, anchor id)
SECTIONS = {
    "score_trend": ("Country Viability Composite Score",            "ov-score-trend"),
    "grade_map":   ("Grades & Percentile Buckets",        "ov-grade-map"),
    "capex_trend": ("CAPEX: Definition & Trends",         "ov-capex-trend"),
    "sectors_bar": ("Investment Profile: Top Industries ",       "ov-sectors"),
    "destinations_bar": ("Investment Profile: Top Destinations", "ov-destinations"),
    "compare":     ("Benchmarking (Country vs. Country)", "ov-compare"),
    "forecast":    ("FDI Forecasts (2025–2028)",          "ov-forecast"),
    # alias keys used by prior samples
    "compare_scores": ("Benchmarking (Country vs. Country)", "ov-compare"),
    "forecast_line":  ("FDI Forecasts (2025–2028)",          "ov-forecast"),
}

# Indicator weights table (exact values provided by you)
_WEIGHTS = [
    ("GDP growth (annual %)", 10),
    ("GDP per capita, PPP (current international $)", 8),
    ("Current account balance (% of GDP)", 6),
    ("Foreign direct investment, net outflows (% of GDP)", 6),
    ("Inflation, consumer prices (annual %)", 5),
    ("Exports of goods and services (% of GDP)", 5),
    ("Imports of goods and services (% of GDP)", 5),
    ("Political Stability and Absence of Violence/Terrorism: Estimate", 12),
    ("Government Effectiveness: Estimate", 10),
    ("Control of Corruption: Estimate", 8),
    ("Access to electricity (% of population)", 9),
    ("Individuals using the Internet (% of population)", 8),
    ("Total reserves in months of imports", 8),
]

# Category explanations you provided (business language)
_CATEGORIES = {
    "Economic Performance": [
        "GDP growth: Measures the pace of economic expansion year over year.",
        "Unemployment: Indicates labor market health and economic inclusiveness.",
        "GNI per capita: Reflects average income and development level.",
        "GDP per capita: Economic output per person, adjusting for purchasing power.",
        "Current account balance: Measures external competitiveness and trade sustainability.",
        "FDI inflows: Captures foreign investor confidence and global integration.",
        "Exports of goods and services: Sign of productive capacity and trade links.",
        "Imports of goods and services: Reflects consumption and openness to trade.",
        "Tax revenue: Indicates fiscal capacity and public service potential.",
        "Inflation (CPI): Price stability indicator from the consumer perspective.",
        "Inflation (GDP deflator): Broad measure of inflation across the economy.",
    ],
    "Governance & Political Stability": [
        "Political Stability: Measures likelihood of unrest/violence that could disrupt operations.",
        "Government Effectiveness: Assesses public service quality and policy execution.",
        "Control of Corruption: Reflects the integrity of political and economic institutions.",
        "Rule of Law: Gauges confidence in and adherence to legal frameworks.",
        "Regulatory Quality: Evaluates ease and fairness of private-sector regulation.",
    ],
    "Financial/Infrastructure Development": [
        "Access to electricity: A proxy for infrastructure readiness.",
        "Access to clean fuels: Indicates energy quality and public health impacts.",
        "Internet usage: Reflects digital infrastructure and connectivity of consumers/firms.",
        "Broad money: Represents overall money supply and financial sector scale.",
        "Domestic credit to private sector: Availability of finance for private entities.",
        "External debt (% of GNI): Highlights financial exposure and debt sustainability.",
        "Market capitalization (% of GDP): Shows financial market depth.",
        "Total reserves (months of imports): Represents capacity to withstand economic shocks.",
    ],
}

# Grade methodology text (exact business framing you requested)
_GRADES = [
    "A+: 90th percentile and above — leading destinations with consistently high attractiveness scores.",
    "A: 75th–90th percentile — strong performers with favorable fundamentals but minor constraints.",
    "B: 50th–75th percentile — moderate performers; viable under specific conditions or with reforms.",
    "C: 25th–50th percentile — countries with significant challenges; requiring high-risk tolerance.",
    "D: Bottom 25% — least attractive destinations for near-term investment.",
    "Grading is performed independently for each year,meaning that a country's grade in a year is based on its performance relative to other countries **in that same year**. This ensures fairness and prevents any single year with unusually high or low global performance from skewing the grades across the entire dataset.",
]

def _toc():
    st.markdown("### Quick Navigation")
    seen = set()
    cols = st.columns(2)
    items = []
    for key, (title, anchor) in SECTIONS.items():
        if anchor not in seen:  # avoid duplicates
            items.append((title, anchor))
            seen.add(anchor)

    left = items[:len(items)//2]
    right = items[len(items)//2:]
    with cols[0]:
        for title, anchor in left:
            st.markdown(f"- [{title}](#{anchor})")
    with cols[1]:
        for title, anchor in right:
            st.markdown(f"- [{title}](#{anchor})")


def _anchor(title: str, anchor_id: str):
    # reserve scroll margin so headers aren't hidden under Streamlit's chrome
    st.markdown(
        f"""<div id="{anchor_id}" style="scroll-margin-top: 80px;"></div>""",
        unsafe_allow_html=True,
    )
    st.subheader(title)

def _weights_table():
    # Build rows
    rows = "\n".join(
        f"<tr><td class='ind'>{ind}</td><td class='num'>{w}</td></tr>"
        for ind, w in _WEIGHTS
    )

    # Dynamic height so no scrollbars
    height = min(700, 120 + 36 * len(_WEIGHTS))

    st_html(
        f"""
        <style>
          .weights-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-family: inherit;
            font-size: inherit;
          }}
          .weights-table th, .weights-table td {{
            border: 1px solid #e6e6e6;
            padding: 10px 12px;
            vertical-align: middle;
          }}
          .weights-table thead th {{
            background: #f8f9fa;
            font-weight: 600;
            text-align: center;
          }}
          .weights-table td.ind {{ width: 75%; }}
          .weights-table td.num {{ width: 25%; text-align: center; }}
          .weights-wrap {{ border-radius: 10px; overflow: hidden; border: 1px solid #eee; }}
        </style>

        <div class="weights-wrap">
          <table class="weights-table">
            <thead>
              <tr><th>Indicator</th><th>Weight (%)</th></tr>
            </thead>
            <tbody>
              {rows}
            </tbody>
          </table>
        </div>
        """,
        height=height,
        scrolling=False,
    )


def _categories():
    for cat, bullets in _CATEGORIES.items():
        with st.expander(cat, expanded=False):
            for b in bullets:
                st.markdown(f"- {b}")

def _grades_section():
    for line in _GRADES:
        st.markdown(f"- {line}")

def _business_and_technical_pairs(pairs: list[tuple[str, list[str], list[str]]]):
    """
    Render sections with two sub-blocks:
    - Business Use (what to interpret / how to use)
    - Technical Notes (data/units/method)
    Each pair: (title, business_bullets[], technical_bullets[])
    """
    for title, biz, tech in pairs:
        st.markdown(f"**{title}**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("_Business use_")
            for b in biz: st.markdown(f"- {b}")
        with c2:
            st.markdown("_Technical notes_")
            for t in tech: st.markdown(f"- {t}")
        st.markdown("---")

def _score_trend_section():
    st.markdown("### Interpreting the score trend")

    st.markdown("**Why it matters:**")
    st.markdown("""
    - **Identify trajectory** — rising scores signal improving fundamentals, while declines may indicate emerging risks.
    - **Benchmark effectively** — compare your target market’s trend against peers to spot relative over- or under-performance.
    - **Time decisions** — use trends to gauge whether now is the right moment for expansion, consolidation, or exit. 
    """)

def _auto_jump():
    """
    Support URL param ?jump=<key> or session_state['overview_focus'] to auto-scroll.
    External files can set st.session_state['overview_focus'] and rerun to trigger.
    """
    params = st.query_params
    jump_key = None
    if "jump" in params:
        jump_key = params.get("jump")
    elif "overview_focus" in st.session_state:
        jump_key = st.session_state.get("overview_focus")

    key = str(jump_key) if jump_key is not None else None
    if key and key in SECTIONS:
        _, anchor = SECTIONS[key]
        from streamlit.components.v1 import html
        html(
            f"""
            <script>
              setTimeout(function() {{
                var el = window.parent.document.getElementById("{anchor}");
                if (el) {{
                  el.scrollIntoView({{ behavior: "smooth", block: "start" }});
                }}
              }}, 300);
            </script>
            """,
            height=0,
        )

def render_overview_tab():

    # Table of contents / quick links
    _toc()
    st.markdown("---")

    # 1) Viability Score & Trend
    _anchor(*SECTIONS["score_trend"])
    st.markdown(
    """- **What it is:** A normalized 0–1 composite index (higher = more attractive) of macro, governance, and infrastructure indicators for each country-year observation.  
    **Weighted mix:** `Score = 0.45 Econ + 0.30 Gov + 0.25 Infra`"""
    )

    st.markdown("**Indicators Weights** as a share of the composite score:")
    _weights_table()
    st.markdown("**Indicators by Category**")
    _categories()

    _score_trend_section()

    # 2) Grades & Percentile Buckets
    _anchor(*SECTIONS["grade_map"])
    st.markdown("**Grading Scale (peer-relative by year)**")
    _grades_section()
    _business_and_technical_pairs([
        (
            "Interpreting the score trend",
            [
                "Track direction of change across years to spot improving or deteriorating fundamentals.",
                "Use alongside grades to understand both *absolute level* (score) and *peer-relative position* (grade).",
            ],
            [
                "Scores are averaged per country-year from weighted indicators; normalization ensures cross-indicator comparability.",
                "If viewing a single year in the app, the ‘latest’ trend point will match the selection filters in the Scoring tab.",
            ]
        ),
    ])

    # 3) CAPEX — Definition & Trend
    _anchor(*SECTIONS["capex_trend"])
    _business_and_technical_pairs([
        (
            "Using grades in decisions",
            [
                "Grades simplify communication with executives and non-technical stakeholders.",
                "Combine grades with sector context (e.g., a ‘B’ country might still be optimal for specific industries).",
            ],
            [
                "Grades are computed by percentile **within each year** to avoid cross-year distortions.",
                "Countries near threshold cut-offs can shift grades year-to-year despite small score changes.",
            ],
        ),
    ])

    # 4) CAPEX — Geographic View
    _anchor(*SECTIONS["capex_map"])
    _business_and_technical_pairs([
        (
            "Reading the CAPEX map",
            [
                "Identify geographic concentration vs. diversification opportunities.",
                "Layer sector and grade insights to prioritize where to expand or defend.",
            ],
            [
                "Choropleth is aggregated by country (and by selected year if you filter).",
                "Mind population/size effects; use continent/country filters to reduce bias.",
            ],
        ),
    ])

    # 5) Industry Landscape (Sectors)
    _anchor(*SECTIONS["sectors_bar"])
    _business_and_technical_pairs([
        (
            "Sector bars (Companies / Jobs / Projects / CAPEX)",
            [
                "Expose the composition of activity: e.g., high CAPEX with few projects ⇒ larger average deal size.",
                "Use Jobs/Companies to assess labor intensity and ecosystem depth.",
            ],
            [
                "Data is grouped by (country, sector) and summed over 2021–2024 in your current build.",
                "Sector canonicalization harmonizes labels (e.g., ‘IT’, ‘Software’) for coherent comparisons.",
            ],
        ),
    ])

    # 6) Target Countries (Destinations)
    _anchor(*SECTIONS["destinations_bar"])
    _business_and_technical_pairs([
        (
            "Destinations ranking (for a source country)",
            [
                "Rank markets by Companies, Jobs, Projects, or CAPEX to gauge outbound focus and white space.",
                "Evaluate concentration risk (few destinations dominate) vs. diversification.",
            ],
            [
                "Aggregations exclude ‘Total/All’ rollups; flows are summed by (source → destination).",
                "Units: CAPEX in $B; other metrics are counts (companies/projects) or people (jobs).",
            ],
        ),
    ])

    # 7) Benchmarking (Compare)
    _anchor(*SECTIONS["compare"])
    _business_and_technical_pairs([
        (
            "Side-by-side shortlisting",
            [
                "Contrast two markets on average viability score and total CAPEX to build an initial shortlist.",
                "Use as an executive-ready snapshot; follow up with sector/destination detail where gaps emerge.",
            ],
            [
                "Score = mean of the country’s available yearly scores in scope; CAPEX = summed in $B.",
                "Selections in the Compare tab use canonicalized country names to align with the core datasets.",
            ],
        ),
    ])

    # 8) FDI Forecasts (2025–2028)
    _anchor(*SECTIONS["forecast"])
    st.markdown(
        "The **Forecast** tab projects CAPEX for **2025–2028**. Treat forecasts as *directional scenarios*, not point guarantees."
    )
    # Technical details aligned to forecasting.py
    st.markdown("**Model selection & training (matches `forecasting.py`)**")
    st.markdown(
        """
- Split: last **15%** of the series (bounded to **2–4 years**) used as a test window for RMSE selection.  
- Candidates: **ARIMA**, **ARIMAX**, **SARIMA**, **SARIMAX** with small order grids.  
- Exogenous variables (when present): standardized with **StandardScaler fit on TRAIN only** (no leakage).  
- Selection: model with lowest **RMSE** on the held-out test years.  
- Refit: best model is refit on the **full** log-CAPEX series; future exog is the **last scaled row repeated**.  
- Horizon: **exactly 2025–2028** (only years beyond the last observed).  
- Plot: shows **Actual CAPEX** (history) and **Forecast** (2025–2028 dashed).  
        """
    )
    st.markdown("**Business guidance**")
    st.markdown(
        """
- Use forecasts to compare *relative* momentum across countries; validate with pipeline intelligence.  
- Stress-test with alternative exogenous sets and scenario bounds; treat large residual volatility with caution.  
        """
    )

    # Auto-jump if query param or session flag is present
    _auto_jump()

# ---- Aliases so one ℹ️ per tab can jump to the right section ----------------
SECTIONS.update({
    # one-button-per-tab keys → map to your existing anchors
    "scoring_tab":       SECTIONS["score_trend"],
    "capex_tab":         SECTIONS["capex_trend"],
    "sectors_tab":       SECTIONS["sectors_bar"],
    "destinations_tab":  SECTIONS["destinations_bar"],
    "compare_tab":       SECTIONS["compare"],
    "forecast_tab":      SECTIONS["forecast"],
})

# ---- Public helper: ℹ️ button you can use from any tab ----------------------
def info_button(section_key: str, help_text: str = "What is this?"):
    """
    Renders a small ℹ️ button. When clicked, sets session state so the app
    switches to the Overview tab and scrolls to the correct section.
    """
    if st.button("ℹ️", key=f"info_{section_key}", help=help_text):
        st.session_state["overview_focus"] = section_key
        st.session_state["_force_overview"] = True

# ---- Public helper: JS to switch to Overview and smooth-scroll ---------------
def emit_auto_jump_script():
    """
    If `_force_overview` is set, click the Overview tab and scroll to the
    anchor for `overview_focus`.
    """
    if not st.session_state.get("_force_overview"):
        return

    key = st.session_state.get("overview_focus")
    if not key or key not in SECTIONS:
        st.session_state["_force_overview"] = False
        return

    _, anchor_id = SECTIONS[key]

    from streamlit.components.v1 import html
    html(f"""
    <script>
    (function() {{
      function jump() {{
        try {{
          const root = window.parent.document;
          // Click the Overview tab (match by label text)
          const tabs = root.querySelectorAll('[role="tab"], [data-baseweb="tab"]');
          let over = null;
          tabs.forEach(t => {{
            const txt = (t.innerText || '').trim().toLowerCase();
            if (txt.includes('overview')) over = t;
          }});
          if (over) over.click();

          // Smooth-scroll to the target anchor
          setTimeout(() => {{
            const el = root.getElementById("{anchor_id}");
            if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          }}, 300);
        }} catch (e) {{
          // no-op
        }}
      }}
      setTimeout(jump, 60);
    }})();
    </script>
    """, height=0)

    # reset the trigger so we don't keep jumping
    st.session_state["_force_overview"] = False

