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
    "score_trend": ("Country Viability Composite Score", "ov-score-trend"),
    "grade_map":   ("Grades & Percentile Buckets",       "ov-grade-map"),
    "capex_trend": ("CAPEX: Definition & Trends",        "ov-capex-trend"),
    "investment_profile": ("Investment Profile: Top Industries & Destinations", "ov-sectors"),
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

# Category explanations
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

# Grade methodology text
_GRADES = [
    "A+: 90th percentile and above — leading destinations with consistently high attractiveness scores.",
    "A: 75th–90th percentile — strong performers with favorable fundamentals but minor constraints.",
    "B: 50th–75th percentile — moderate performers; viable under specific conditions or with reforms.",
    "C: 25th–50th percentile — countries with significant challenges; requiring high-risk tolerance.",
    "D: Bottom 25% — least attractive destinations for near-term investment."
]

# ------------------------------
# UI helpers
# ------------------------------

def _toc():
    """Two-column Quick Navigation with chip-style buttons (no blue links)."""
    seen = set()
    items: list[tuple[str, str]] = []
    for _, (title, anchor) in SECTIONS.items():
        if anchor not in seen:
            items.append((title, anchor))
            seen.add(anchor)

    left = items[:len(items)//2]
    right = items[len(items)//2:]

    st.markdown("### Quick Navigation")
    c1, c2 = st.columns(2)

    def _render_col(col_items):
        chips = "\n".join(
            f"""
            <div class="toc-chip" onclick="
              (function(){{
                const root = window.parent?.document || document;
                const el = root.getElementById('{anchor}');
                if (el) el.scrollIntoView({{behavior:'smooth', block:'start'}});
              }})();
            ">
              <span class="toc-label">{title}</span>
            </div>
            """
            for (title, anchor) in col_items
        )
        st_html(f"""
        <style>
          .toc-wrap {{ display: flex; flex-direction: column; gap: 10px; }}
          .toc-chip {{
            background: #f8f9fa;
            border: 1px solid #e6e6e6;
            border-radius: 10px;
            padding: 12px 14px;
            cursor: pointer;
            transition: background 120ms ease, border-color 120ms ease, transform 60ms ease;
            user-select: none;
          }}
          .toc-chip:hover {{ background: #f3f4f6; border-color: #dfe3e8; }}
          .toc-chip:active {{ transform: translateY(1px); }}
          .toc-label {{ color: #111827; font-weight: 500; }}
        </style>
        <div class="toc-wrap">{chips}</div>
        """, height=52*max(1, len(col_items)), scrolling=False)

    with c1: _render_col(left)
    with c2: _render_col(right)


def _anchor(title: str, anchor_id: str):
    st.markdown(f"""<div id="{anchor_id}" style="scroll-margin-top: 80px;"></div>""",
                unsafe_allow_html=True)
    st.subheader(title)


def _weights_table():
    rows = "\n".join(
        f"<tr><td class='ind'>{ind}</td><td class='num'>{w}</td></tr>"
        for ind, w in _WEIGHTS
    )
    height = min(700, 120 + 36 * len(_WEIGHTS))
    st_html(
        f"""
        <style>
          .weights-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
          .weights-table th, .weights-table td {{
            border: 1px solid #e6e6e6; padding: 10px 12px; vertical-align: middle;
          }}
          .weights-table thead th {{ background: #f8f9fa; font-weight: 600; }}
          .weights-table td.ind {{ width: 75%; }}
          .weights-table td.num {{ width: 25%; text-align: center; }}
          .weights-wrap {{ border-radius: 10px; border: 1px solid #eee; }}
        </style>
        <div class="weights-wrap">
          <table class="weights-table">
            <thead><tr><th>Indicator</th><th>Weight (%)</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """, height=height, scrolling=False)


def _categories():
    for cat, bullets in _CATEGORIES.items():
        with st.expander(cat, expanded=False):
            for b in bullets: st.markdown(f"- {b}")


def _grades_section():
    grade_lines = _GRADES[:5]
    methodology = (
        "Grading is performed independently for each year, meaning that a country's grade in a year "
        "is based on its performance relative to other countries in that same year."
    )
    items = []
    for line in grade_lines:
        label, desc = line.split(":", 1)
        items.append((label.strip(), desc.strip()))
    st_html(f"""
    <style>
      .grade-grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
      .grade-item {{
        display: grid; grid-template-columns: 84px 1fr; gap: 12px; align-items: center;
        border: 1px solid #e6e6e6; border-radius: 10px; padding: 12px 14px;
      }}
      .grade-badge {{
        text-align: center; font-weight: 700; border-radius: 10px;
        border: 1px solid #dfe3e8; background: #f8f9fa; padding: 10px 0; width: 70px;
      }}
      .callout {{ border: 1px solid #e6e6e6; border-left: 4px solid #2563eb;
        background: #f8fafc; border-radius: 8px; padding: 12px 14px; }}
    </style>
    <div class="grade-grid">
      {"".join([f"<div class='grade-item'><div class='grade-badge'>{l}</div><div>{d}</div></div>" for l,d in items])}
    </div>
    <div class="callout"><strong>Methodology:</strong> {methodology}</div>
    """, height=len(items)*86 + 96, scrolling=False)


def _benchmarking_explainer_block(what: list[str], why: list[str], how: list[str]):
    """Stacked, boxed explainer with HTML-safe bullets."""
    def _box(title: str, bullets: list[str]):
        st.markdown(f"#### {title}")
        parts = []
        for b in bullets:
            if isinstance(b, str) and b.lstrip().startswith("<"):
                parts.append(b)
            else:
                parts.append(f"<p>• {b}</p>")
        st.markdown(
            "<div style='padding:10px; border:1px solid #e6e6e6; border-radius:6px; background:#fafafa;'>"
            + "".join(parts) + "</div>",
            unsafe_allow_html=True,
        )
    _box("What it is", what)
    _box("Why it matters", why)
    _box("How to navigate", how)
    st.markdown("---")


# ------------------------------
# Main render
# ------------------------------

def render_overview_tab():
    _toc()
    st.markdown("---")

    # Score
    _anchor(*SECTIONS["score_trend"])
    st.markdown(
        "**What it is:** A normalized 0–1 composite index (higher = more attractive) "
        "of macro, governance, and infrastructure indicators for each country-year.  "
        "**Weighted mix:** `Score = 0.45 Econ + 0.30 Gov + 0.25 Infra`"
    )
    st.markdown("**Indicator Weights**")
    _weights_table()
    st.markdown("**Indicators by Category**")
    _categories()
    st.markdown("---")

    # Grades
    _anchor(*SECTIONS["grade_map"])
    st.markdown("**Grades Distribution**")
    _grades_section()
    st.markdown("---")

    # CAPEX
    _anchor(*SECTIONS["capex_trend"])
    _benchmarking_explainer_block(
        what=[
            "Capital expenditure (CAPEX) represents funds allocated by governments or firms to build, acquire, or upgrade long-lived assets and infrastructure.",
        ],
        why=[
            "Tracking CAPEX trends highlights momentum in cross-border investment flows and helps distinguish sustained growth from episodic spikes.",
            "Consistent CAPEX growth indicates durable investor confidence, while volatility may reflect exposure to external shocks or policy uncertainty.",
        ],
        how=[
            "Views provided: global trend, CAPEX by grade, top source countries by absolute value and by growth.",
            "All CAPEX values are displayed in **billions of USD ($B)** for consistency.",
            "Use the filters (year, continent, country, grade) at the top of the CAPEX tab to adjust scope.",
            "View the **Global CAPEX Trend** line chart for overall momentum; switch to a single year to see KPIs.",
            "Inspect the **CAPEX Map** to identify geographic concentration versus diversification.",
            "Check **Top Countries** and **Growth Ranking** to spot rising sources.",
            "Use **CAPEX by Grade** to compare investment flows across grades.",
        ],
    )

    # Investment Profile
    _anchor(*SECTIONS["investment_profile"])
    _benchmarking_explainer_block(
        what=["Sector-level and destination-level views of FDI flows (Companies, Jobs, Projects, CAPEX)."],
        why=["Identify industry strengths vs. diversification gaps; assess concentration risk and white space opportunities."],
        how=[
            "In **Industries tab**: choose source country and metric to see sector bars or KPIs.",
            "In **Destinations tab**: choose source country and metric to see top destinations and map.",
        ],
    )

    # Benchmarking
    _anchor(*SECTIONS["compare"])
    _benchmarking_explainer_block(
        what=[
            "A head-to-head comparison of two countries on overall attractiveness and realized investment flows.",
            "Provides a high-level snapshot of how markets stack up against each other."
        ],
        why=[
            "Highlight trade-offs between viability fundamentals and actual investor activity.",
            "Support early-stage decision-making by showing which market deserves deeper investigation.",
        ],
        how=[
            "Select two countries in the Compare tab.",
            "Adjust filters (year, continent, grade) to control the scope.",
            "Review the KPIs: Average Viability Score vs. Total CAPEX ($B).",
        ],
    )

    # Forecasting
    _anchor(*SECTIONS["forecast"])
    what_forecast = [
        "Forward-looking projections of CAPEX for 2025–2028, based on country-level time-series models.",
        "Extends the dashboard beyond static snapshots by incorporating predictive analytics."
    ]
    why_forecast = [
        "Allows agencies to anticipate shifts in FDI flows rather than reacting to them.",
        "Supports strategic prioritization by highlighting countries expected to accelerate in attractiveness."
    ]
    how_forecast = [
        (
            "<p>Each forecast is generated using ARIMA-family models:</p>"
            "<ul><li><strong>ARIMA</strong>: based only on past CAPEX values.</li>"
            "<li><strong>ARIMAX</strong>: ARIMA with exogenous economic/governance indicators.</li>"
            "<li><strong>SARIMA</strong>: adds seasonal or cyclical patterns.</li>"
            "<li><strong>SARIMAX</strong>: combines seasonality and exogenous indicators.</li></ul>"
        ),
        (
            "<p>The <em>order</em> (p,d,q) shown under the chart explains:</p>"
            "<ul><li>p = lag depth (autoregression)</li>"
            "<li>d = differencing to remove trends</li>"
            "<li>q = moving average of past shocks</li></ul>"
        ),
        (
            "<p><strong>RMSE</strong> (Root Mean Squared Error) measures forecast accuracy on the test window — lower is better.</p>"
            "<p>Dashed lines = forecasts for 2025–2028; solid lines = historical CAPEX ($B).</p>"
        )
    ]
    _benchmarking_explainer_block(what_forecast, why_forecast, how_forecast)


# ------------------------------
# Info button + auto jump
# ------------------------------

def info_button(section_key: str,
                help_text: str = "What is this?",
                key_suffix: str | None = None):
    if not key_suffix:
        key_suffix = f"default_{section_key}"
    unique_key = f"info_{key_suffix}"
    if st.button("ℹ️", key=unique_key, help=help_text):
        st.session_state["overview_focus"] = section_key
        st.session_state["_force_overview"] = True


def emit_auto_jump_script():
    if not st.session_state.get("_force_overview"):
        return
    key = st.session_state.get("overview_focus")
    if not key or key not in SECTIONS:
        st.session_state["_force_overview"] = False
        return
    _, anchor_id = SECTIONS[key]
    html(f"""
    <script>
    (function(){{
      function jump(){{
        try {{
          const root = window.parent.document;
          const tabs = root.querySelectorAll('[role="tab"], [data-baseweb="tab"]');
          let over = null;
          tabs.forEach(t => {{
            const txt = (t.innerText||'').trim().toLowerCase();
            if (txt.includes('overview')) over = t;
          }});
          if (over) over.click();
          setTimeout(() => {{
            const el = root.getElementById("{anchor_id}");
            if (el) el.scrollIntoView({{behavior:'smooth', block:'start'}});
          }}, 300);
        }} catch(e){{}}
      }}
      setTimeout(jump,60);
    }})();
    </script>
    """, height=0)
    st.session_state["_force_overview"] = False
