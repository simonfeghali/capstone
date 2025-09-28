# overview.py
# -----------------------------------------------------------------------------
# Overview tab for the FDI & Viability dashboard.
# - Business-first explanations for every tab / plot
# - Deep technical notes (weights, methodology, forecasting details)
# - In-page anchors so other tabs (or external links) can jump here
# -----------------------------------------------------------------------------

from __future__ import annotations
import streamlit as st
from streamlit.components.v1 import html as st_html

FONT_CSS = """
<style>
  /* Load Streamlit's default font inside component iframes */
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
  :root, html, body, #root, .root, .container, * {
    font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, 'Helvetica Neue', Arial, sans-serif !important;
  }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Section map: title + anchor id
# ─────────────────────────────────────────────────────────────────────────────
SECTIONS = {
    "score_trend":       ("Country Viability Composite Score", "ov-score-trend"),
    "grade_map":         ("Grades & Percentile Buckets",       "ov-grade-map"),
    "capex_trend":       ("CAPEX: Definition & Trends",        "ov-capex-trend"),
    "investment_profile":("Investment Profile: Top Industries & Destinations", "ov-sectors"),
    "compare":           ("Benchmarking (Country vs. Country)", "ov-compare"),
    "forecast":          ("FDI Forecasts (2025–2028)",          "ov-forecast"),
    # legacy alias keys used elsewhere
    "compare_scores":    ("Benchmarking (Country vs. Country)", "ov-compare"),
    "forecast_line":     ("FDI Forecasts (2025–2028)",          "ov-forecast"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Indicator weights (provided)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Category explanations
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Grades copy
# ─────────────────────────────────────────────────────────────────────────────
_GRADES = [
    "A+: 90th percentile and above — leading destinations with consistently high attractiveness scores.",
    "A: 75th–90th percentile — strong performers with favorable fundamentals but minor constraints.",
    "B: 50th–75th percentile — moderate performers; viable under specific conditions or with reforms.",
    "C: 25th–50th percentile — countries with significant challenges; requiring high-risk tolerance.",
    "D: Bottom 25% — least attractive destinations for near-term investment."
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _toc():
    """Two-column Quick Navigation rendered as neutral 'chips' (no blue links)."""
    seen = set()
    items: list[tuple[str, str]] = []
    for _, (title, anchor) in SECTIONS.items():
        if anchor not in seen:
            items.append((title, anchor))
            seen.add(anchor)

    left = items[: len(items) // 2]
    right = items[len(items) // 2 :]

    st.markdown("### Quick Navigation")
    c1, c2 = st.columns(2)

    def _render(col_items):
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
        st_html(
            f"""
            {FONT_CSS}
            <style>
              .toc-wrap {{ 
                display: flex; 
                flex-direction: column; 
                gap: 10px; 
                padding-bottom: 10px;   /* NEW: avoids clipping */
              }}
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
            <div class="toc-wrap">
              {chips}
            </div>
            """,
            height=60 * max(1, len(col_items)) + 12,  # NEW: a bit taller than the content
            scrolling=False,
        )

    with c1:
        _render(left)
    with c2:
        _render(right)


def _anchor(title: str, anchor_id: str):
    st.markdown(f"""<div id="{anchor_id}" style="scroll-margin-top: 80px;"></div>""", unsafe_allow_html=True)
    st.subheader(title)


def _weights_table():
    rows = "\n".join(
        f"<tr><td class='ind'>{ind}</td><td class='num'>{w}</td></tr>" for ind, w in _WEIGHTS
    )
    height = min(700, 120 + 36 * len(_WEIGHTS))

    st_html(
        f"""
        {FONT_CSS}
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
            <thead><tr><th>Indicator</th><th>Weight (%)</th></tr></thead>
            <tbody>{rows}</tbody>
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
    """Single-column grade scale with badges + methodology callout."""
    items = []
    for line in _GRADES:
        label, desc = line.split(":", 1)
        items.append((label.strip(), desc.strip()))

    st_html(
        f"""
        {FONT_CSS}
        <style>
          .grade-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
            margin: 6px 0 14px 0;
          }}
          .grade-item {{
            display: grid;
            grid-template-columns: 84px 1fr;
            gap: 12px;
            align-items: center;
            background: #ffffff;
            border: 1px solid #e6e6e6;
            border-radius: 10px;
            padding: 12px 14px;
          }}
          .grade-badge {{
            text-align: center;
            font-weight: 700;
            border-radius: 10px;
            border: 1px solid #dfe3e8;
            background: #f8f9fa;
            padding: 10px 0;
            width: 70px;
          }}
          .grade-desc {{ color: #111827; }}
          .callout {{
            border: 1px solid #e6e6e6;
            border-left: 4px solid #2563eb;
            background: #f8fafc;
            border-radius: 8px;
            padding: 12px 14px;
            margin-top: 10px;
          }}
        </style>
        <div class="grade-grid">
          {"".join([
            f"<div class='grade-item'><div class='grade-badge'>{label}</div><div class='grade-desc'>{desc}</div></div>"
            for (label, desc) in items
          ])}
        </div>
        <div class="callout">
          <strong>Methodology:</strong> Grading is performed independently for each year, based on a country's
          performance relative to other countries in that same year.
        </div>
        """,
        height=len(items) * 86 + 96,
        scrolling=False,
    )


def _capex_explainer_block(what: list[str], why: list[str], how: list[str]):
    def _box(title: str, bullets: list[str]):
        st.markdown(f"#### {title}")
        st.markdown(
            "<div style='padding:10px; border:1px solid #e6e6e6; border-radius:6px; background-color:#fafafa;'>"
            + "".join([f"<p>• {b}</p>" for b in bullets])
            + "</div>",
            unsafe_allow_html=True,
        )
    _box("What it is", what)
    _box("Why it matters", why)
    _box("How to navigate", how)
    st.markdown("---")


def _what_why_how_block(title: str, what: list[str], why: list[str], how: list[str]):
    st.markdown(f"**{title}**")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("_What it is_")
        for b in what:
            st.markdown(f"- {b}")
    with cols[1]:
        st.markdown("_Why it matters_")
        for b in why:
            st.markdown(f"- {b}")
    with cols[2]:
        st.markdown("_How to navigate_")
        for b in how:
            st.markdown(f"- {b}")
    st.markdown("---")


def _benchmarking_explainer_block(what: list[str], why: list[str], how: list[str]):
    def _box(title: str, bullets: list[str]):
        st.markdown(f"#### {title}")
        st.markdown(
            "<div style='padding:10px; border:1px solid #e6e6e6; border-radius:6px; background-color:#fafafa;'>"
            + "".join([f"<p>• {b}</p>" for b in bullets])
            + "</div>",
            unsafe_allow_html=True,
        )
    _box("What it is", what)
    _box("Why it matters", why)
    _box("How to navigate", how)
    st.markdown("---")


def _forecasting_explainer_block(what: list[str], why: list[str], how: list[object]):
    """
    Stacked cards for Forecasting, with nested list support (so we can
    show ARIMA/ARIMAX/SARIMA/SARIMAX under one bullet without blank dots).
    """
    def _box(title: str, bullets: list[object]):
        def render_items(items: list[object]) -> str:
            html = "<ul class='fx-list'>"
            for it in items:
                if isinstance(it, list):
                    html += "<ul class='fx-sublist'>"
                    for sub in it:
                        html += f"<li>{sub}</li>"
                    html += "</ul>"
                else:
                    html += f"<li>{it}</li>"
            html += "</ul>"
            return html

        st.markdown(f"#### {title}")
        st.markdown(
            """
            <style>
              .fx-card     { padding:10px; border:1px solid #e6e6e6; border-radius:6px; background:#fafafa; }
              .fx-list     { margin:0; padding-left:1.1rem; }
              .fx-list>li  { margin:4px 0; }
              .fx-sublist  { margin:4px 0 6px 1.1rem; padding-left:1.1rem; }
              .fx-sublist>li { margin:2px 0; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='fx-card'>{render_items(bullets)}</div>", unsafe_allow_html=True)

    _box("What it is", what)
    _box("Why it matters", why)
    _box("How the forecasts are generated", how)
    st.markdown("---")


def _auto_jump():
    """Support URL param ?jump=<key> or session_state['overview_focus'] to auto-scroll."""
    params = st.query_params
    jump_key = None
    if "jump" in params:
        jump_key = params.get("jump")
    elif "overview_focus" in st.session_state:
        jump_key = st.session_state.get("overview_focus")

    key = str(jump_key) if jump_key is not None else None
    if key and key in SECTIONS:
        _, anchor = SECTIONS[key]
        st_html(
            f"""
            <script>
              setTimeout(function() {{
                var el = window.parent.document.getElementById("{anchor}");
                if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});
              }}, 300);
            </script>
            """,
            height=0,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Main renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_overview_tab():

    st_html(
        """
        <style>
          .toc-wrap, .toc-chip, .toc-label,
          .weights-table, .weights-table td, .weights-table th,
          .grade-grid, .grade-item, .grade-badge, .grade-desc,
          .callout, .fx-card, .fx-list, .fx-sublist {
            font-family: inherit !important;
          }
        </style>
        """,
        height=0,
    )
    
    _toc()
    st.markdown("---")

    # 1) Score & indicators
    _anchor(*SECTIONS["score_trend"])
    st.markdown(
        """**What it is:** A normalized 0–1 composite index (higher = more attractive) of macro, governance,
        and infrastructure indicators for each country-year observation.  
        **Weighted mix:** `Score = 0.45 Econ + 0.30 Gov + 0.25 Infra`"""
    )
    st.markdown("**Indicator Weights** (as a share of the composite):")
    _weights_table()
    st.markdown("**Indicators by Category**")
    _categories()

    st.markdown("<hr style='margin: 1.5em 0; border: none; border-top: 1px solid #e6e6e6;'>", unsafe_allow_html=True)

    # 2) Grades
    _anchor(*SECTIONS["grade_map"])
    st.markdown("**Grades Distribution**")
    _grades_section()

    st.markdown("<hr style='margin: 1.5em 0; border: none; border-top: 1px solid #e6e6e6;'>", unsafe_allow_html=True)

    # 3) CAPEX
    _anchor(*SECTIONS["capex_trend"])
    _capex_explainer_block(
        [
            "Capital expenditure (CAPEX) represents funds allocated by governments or firms to build, acquire, or upgrade long-lived assets and infrastructure that support economic growth and public well-being.",
        ],
        [
            "Tracking CAPEX trends highlights momentum in cross-border investment flows and helps distinguish sustained growth from episodic spikes.",
            "Consistent CAPEX growth indicates durable investor confidence, while volatility may reflect exposure to external shocks or policy uncertainty.",
        ],
        [
            "Views provided: global trend, CAPEX by grade, top source countries by absolute value and by growth.",
            "All CAPEX values are displayed in billions of USD ($B) for consistency.",
            "Use the year / continent / country / grade filters at the top of the CAPEX tab to adjust scope.",
            "View the Global CAPEX Trend line chart for overall momentum; switch to a single year to see KPIs.",
            "Inspect the CAPEX Map (choropleth) to identify geographic concentration versus diversification.",
            "Check Top Countries (bars or KPIs) for absolute levels and the Growth Ranking chart to spot rising sources.",
            "Use the CAPEX by Grade view to compare investment flows across attractiveness grades.",
        ],
    )

    # 4) Investment Profile (Combined)
    _anchor(*SECTIONS["investment_profile"])

    _what_why_how_block(
        "Top industries",
        [
            "Sector-level view of outbound FDI, for the top 10 investing countries, by **Companies, Jobs, Projects, and CAPEX**.",
            "Covers 16 harmonized sectors for comparability across countries.",
        ],
        [
            "Reveals industry strengths vs. gaps; balance capital intensity (CAPEX) against job intensity.",
            "Helps prioritize clusters and ecosystem development where a country is genuinely competitive.",
        ],
        [
            "Choose a **Source Country** and **Metric**; keep **Sector = All** to see a ranked bar chart.",
            "Pick a **specific sector** to switch the view to a single KPI for that sector/country.",
            "Use the **download** button to export the standardized sector table for the selected country.",
        ],
    )

    _what_why_how_block(
        "Top destinations",
        [
            "Destination ranking of the top 10 source countries’ outbound FDI across **Companies, Jobs, Projects, and CAPEX**.",
            "Shows the **Top 15** destinations and a companion map (or a route view for a selected pair).",
        ],
        [
            "Quantifies concentration risk (a few destinations dominate) versus diversification.",
            "Highlights white space: large investors with limited presence in certain regions.",
        ],
        [
            "Select a **Source Country** and **Metric**; keep **Destination = All** to see **Top 15** + map.",
            "Choose a **specific destination** to see a KPI and a **route map** for that pair.",
            "Use the **download** button to export all destination rows for the selected source country.",
        ],
    )

    # 5) Benchmarking (Compare)
    _anchor(*SECTIONS["compare"])
    what_bench = [
        "A head-to-head comparison of two countries on overall attractiveness (Viability Score) and realized investment flows (CAPEX).",
        "Provides a high-level snapshot of how markets stack up within the same timeframe and filters.",
    ]
    why_bench = [
        "Quickly highlight trade-offs between strong viability fundamentals and actual investor activity.",
        "Support early-stage decision-making by identifying which market deserves deeper investigation.",
        "Give executives an easy-to-digest summary that balances quantitative performance with real investment flows.",
    ]
    how_bench = [
        "Select two countries in the Compare tab.",
        "Adjust filters (year, continent, grade) to control the scope of comparison.",
        "Review the headline KPIs: Average Viability Score vs. Total CAPEX ($B)for each country.",
        "Use insights here as a starting point; if a country is a top investor, dive into Industry Landscape and Target Countries for detail.",
    ]
    _benchmarking_explainer_block(what_bench, why_bench, how_bench)

    # 6) Forecasts
    _anchor(*SECTIONS["forecast"])
    what_forecast = [
        "Forward-looking projection of country-level FDI CAPEX for 2025–2028.",
        "Built using ARIMA-family time-series models trained on historical CAPEX data.",
    ]
    why_forecast = [
        "Shows whether countries are likely to gain or lose momentum in attracting investment.",
        "Adds a predictive layer to complement the composite score and past CAPEX analysis.",
    ]
    how_forecast = [
        "Each forecast is generated using ARIMA-family models:",
        [
            "ARIMA: based only on past CAPEX values.",
            "ARIMAX: ARIMA extended with exogenous (economic/governance) indicators.",
            "SARIMA: adds seasonal/cyclical patterns.",
            "SARIMAX: seasonality + exogenous (economic/governance) indicators.",
        ],
        "The order (<em>p,d,q</em>) shown under the chart means the model:",
        [
            "looks back at <em>p</em> past values,",
            "differences the series <em>d</em> times to remove trends,",
            "and models <em>q</em> past shocks/noise.",
        ],
        "RMSE (Root Mean Squared Error) measures forecast accuracy on the test window — lower means better fit.",
        "Dashed lines are forecasts for 2025–2028; solid lines are historical CAPEX ($B).",
    ]
    _forecasting_explainer_block(what_forecast, why_forecast, how_forecast)

    _auto_jump()

# ─────────────────────────────────────────────────────────────────────────────
# Public helpers used by other tabs
# ─────────────────────────────────────────────────────────────────────────────
SECTIONS.update({
    "scoring_tab":      SECTIONS["score_trend"],
    "capex_tab":        SECTIONS["capex_trend"],
    "sectors_tab":      SECTIONS["investment_profile"],
    "destinations_tab": SECTIONS["investment_profile"],
    "compare_tab":      SECTIONS["compare"],
    "forecast_tab":     SECTIONS["forecast"],
})

def info_button(section_key: str, help_text: str = "What is this?", key_suffix: str | None = None):
    """
    Render a small ℹ️ button that jumps to `section_key` on the Overview tab.
    Pass a UNIQUE `key_suffix` per call (e.g., 'sectors_hdr', 'dest_hdr') to avoid key collisions.
    """
    if not key_suffix:
        key_suffix = f"default_{section_key}"
    unique_key = f"info_{key_suffix}"
    if st.button("ℹ️", key=unique_key, help=help_text):
        st.session_state["overview_focus"] = section_key
        st.session_state["_force_overview"] = True


def emit_auto_jump_script():
    """If `_force_overview` is set, switch to Overview and smooth-scroll to the target anchor."""
    if not st.session_state.get("_force_overview"):
        return

    key = st.session_state.get("overview_focus")
    if not key or key not in SECTIONS:
        st.session_state["_force_overview"] = False
        return

    _, anchor_id = SECTIONS[key]
    st_html(
        f"""
        <script>
        (function() {{
          function jump() {{
            try {{
              const root = window.parent.document;
              const tabs = root.querySelectorAll('[role="tab"], [data-baseweb="tab"]');
              let over = null;
              tabs.forEach(t => {{
                const txt = (t.innerText || '').trim().toLowerCase();
                if (txt.includes('overview')) over = t;
              }});
              if (over) over.click();
              setTimeout(() => {{
                const el = root.getElementById("{anchor_id}");
                if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
              }}, 300);
            }} catch (e) {{}}
          }}
          setTimeout(jump, 60);
        }})();
        </script>
        """,
        height=0,
    )
    st.session_state["_force_overview"] = False
