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
    "D: Bottom 25% — least attractive destinations for near-term investment."
]

def _toc():
    """Two-column Quick Navigation with chip-style buttons (no blue links)."""
    # Collect unique (title, anchor) pairs (since multiple keys can share the same anchor)
    seen = set()
    items: list[tuple[str, str]] = []
    for _, (title, anchor) in SECTIONS.items():
        if anchor not in seen:
            items.append((title, anchor))
            seen.add(anchor)

    # Split into two columns (left/right)
    left = items[:len(items)//2]
    right = items[len(items)//2:]

    st.markdown("### Quick Navigation")
    c1, c2 = st.columns(2)

    def _render_col(col, col_items):
        from streamlit.components.v1 import html as st_html
        # Build the chips as buttons; clicking scrolls smoothly to the anchor
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
            box-shadow: 0 0 0 rgba(0,0,0,0);
          }}
          .toc-chip:hover {{
            background: #f3f4f6;
            border-color: #dfe3e8;
          }}
          .toc-chip:active {{
            transform: translateY(1px);
          }}
          .toc-label {{
            color: #111827;           /* neutral text, not link blue */
            font-weight: 500;
          }}
        </style>
        <div class="toc-wrap">
          {chips}
        </div>
        """, height=52*max(1, len(col_items)), scrolling=False)

    with c1: _render_col(c1, left)
    with c2: _render_col(c2, right)



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
    """Single-column grade scale with badges + methodology callout."""
    grade_lines = _GRADES[:5]
    methodology = _GRADES[5] if len(_GRADES) > 5 else (
        "Grading is performed independently for each year, meaning that a country's grade in a year "
        "is based on its performance relative to other countries in that same year."
    )

    items = []
    for line in grade_lines:
        label, desc = line.split(":", 1)
        items.append((label.strip(), desc.strip()))

    st_html(f"""
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
        display: inline-block;
        text-align: center;
        font-weight: 700;
        border-radius: 10px;
        border: 1px solid #dfe3e8;
        background: #f8f9fa;
        padding: 10px 0;
        width: 70px;
      }}
      .grade-desc {{
        font-weight: 400;
        color: #111827;
      }}
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
        f"""
        <div class='grade-item'>
          <div class='grade-badge'>{label}</div>
          <div class='grade-desc'>{desc}</div>
        </div>
        """ for (label, desc) in items
      ])}
    </div>

    <div class="callout">
      <strong>Methodology:</strong> {methodology}
    </div>
    """, height=len(items)*86 + 96, scrolling=False)

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
            for b in biz:
                st.markdown(f"- {b}")
        with c2:
            st.markdown("_Technical notes_")
            for t in tech:
                st.markdown(f"- {t}")
        st.markdown("---")

        
def _what_why_how_block(title: str, what: list[str], why: list[str], how: list[str]):
    st.markdown(f"**{title}**")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("_What it is_")
        for b in what: st.markdown(f"- {b}")
    with cols[1]:
        st.markdown("_Why it matters_")
        for b in why: st.markdown(f"- {b}")
    with cols[2]:
        st.markdown("_How to navigate_")
        for b in how: st.markdown(f"- {b}")
    st.markdown("---")

def _capex_explainer_block(what: list[str], why: list[str], how: list[str]):
    # What it is
    st.markdown("#### What it is")
    st.markdown(
        "<div style='padding:10px; border:1px solid #e6e6e6; border-radius:6px; background-color:#fafafa;'>"
        + "".join([f"<p>• {b}</p>" for b in what])
        + "</div>",
        unsafe_allow_html=True,
    )

    # Why it matters
    st.markdown("#### Why it matters")
    st.markdown(
        "<div style='padding:10px; border:1px solid #e6e6e6; border-radius:6px; background-color:#fafafa;'>"
        + "".join([f"<p>• {b}</p>" for b in why])
        + "</div>",
        unsafe_allow_html=True,
    )

    # How to navigate
    st.markdown("#### How to navigate")
    st.markdown(
        "<div style='padding:10px; border:1px solid #e6e6e6; border-radius:6px; background-color:#fafafa;'>"
        + "".join([f"<p>• {b}</p>" for b in how])
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

def _benchmarking_explainer_block(what: list[str], why: list[str], how: list[str]):
    """Stacked, boxed explainer for Overview sections."""
    def _box(title: str, bullets: list[str]):
        st.markdown(f"#### {title}")

        # Build content allowing raw HTML blocks (e.g., <ul> lists)
        parts = []
        for b in bullets:
            if isinstance(b, str) and b.lstrip().startswith("<"):
                parts.append(b)  # render as-is
            else:
                parts.append(f"<p>• {b}</p>")

        st.markdown(
            "<div style='padding:10px; border:1px solid #e6e6e6; border-radius:6px; background-color:#fafafa;'>"
            + "".join(parts) +
            "</div>",
            unsafe_allow_html=True,
        )

    _box("What it is", what)
    _box("Why it matters", why)
    _box("How to navigate", how)
    st.markdown("---")



def _score_trend_section():
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
        """**What it is:** A normalized 0–1 composite index (higher = more attractive) of macro, governance, and infrastructure indicators for each country-year observation.  
        **Weighted mix:** `Score = 0.45 Econ + 0.30 Gov + 0.25 Infra`"""
    )
    st.markdown("**Indicator Weights** (as a share of the composite):")
    _weights_table()
    st.markdown("**Indicators by Category**")
    _categories()
    _score_trend_section()

    # --- separator between Score and Grades ---
    st.markdown("<hr style='margin: 1.5em 0; border: none; border-top: 1px solid #e6e6e6;'>", unsafe_allow_html=True)


    # 2) Grades & Percentile Buckets
    _anchor(*SECTIONS["grade_map"])
    st.markdown("**Grades Distribution**")
    _grades_section()

    # --- separator between Grades and Capex ---
    st.markdown("<hr style='margin: 1.5em 0; border: none; border-top: 1px solid #e6e6e6;'>", unsafe_allow_html=True)


    # 3) CAPEX — Definition & Trends
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
            "All CAPEX values are displayed in **billions of USD ($B)** for consistency.",
            "Use the **year/continent/country/grade filters** at the top of the CAPEX tab to adjust scope.",
            "View the **Global CAPEX Trend** line chart for overall momentum; switch to a single year to see KPIs.",
            "Inspect the **CAPEX Map** (choropleth) to identify geographic concentration versus diversification.",
            "Check **Top Countries** (bars or KPIs) for absolute levels and the **Growth Ranking** chart to spot rising sources.",
            "Use the **CAPEX by Grade** view to compare investment flows across attractiveness grades.",
        ],
    )


    # Investment Profile — combined section
    _anchor(*SECTIONS["investment_profile"])  # both old keys resolve to the same anchor
    
    # Top industries
    _what_why_how_block(
        "Top industries",
        # WHAT
        [
            "Sector-level view of outbound FDI, for top 10 investing countries identified, by **Companies, Jobs, Projects, and CAPEX**.",
            "Covers 16 canonized sectors for comparability across the top source countries."
        ],
        # WHY
        [
            "Reveals industry strengths vs. gaps; balance capital intensity (CAPEX) against job intensity.",
            "Helps prioritize clusters and ecosystem development where a country is genuinely competitive."
        ],
        # HOW (matches the Sectors tab UI/behavior)
        [
            "Choose a **Source Country** and **Metric**; keep **Sector = All** to see a ranked bar chart.",
            "Pick a **specific sector** to switch the view to a single KPI for that sector/country.",
            "Use the **download** button to export the standardized sector table for the selected country."
        ],
    )
    
    # Top destinations
    _what_why_how_block(
        "Top destinations",
        # WHAT
        [
            "Destination-country ranking of the top 10 source countries' outbound FDI across **Companies, Jobs, Projects, and CAPEX**.",
            "Shows the **Top 15** destinations and a companion map (or a route view for a selected pair)."
        ],
        # WHY
        [
            "Quantifies concentration risk (a few destinations dominate) versus diversification.",
            "Highlights white space: large investors with limited presence in certain regions."
        ],
        # HOW (matches the Destinations tab UI/behavior)
        [
            "Select a **Source Country** and **Metric**; keep **Destination = All** to see **Top 15** + map.",
            "Choose a **specific destination** to see a KPI and a **route map** for that pair.",
            "Use the **download** button to export all destination rows for the selected source country."
        ],
    )


    # 6) Benchmarking (Compare)
    _anchor(*SECTIONS["compare"])
    what_bench = [
        "A head-to-head comparison of two countries on overall attractiveness and realized investment flows.",
        "Provides a high-level snapshot of how markets stack up against each other within the same timeframe and filters."
    ]

    why_bench = [
        "Quickly highlight trade-offs between strong viability fundamentals and actual investor activity.",
        "Support early-stage decision-making by identifying which market deserves deeper investigation.",
        "Give executives an easy-to-digest summary that balances quantitative performance with real investment flows."
    ]

    how_bench = [
        "Select two countries in the Compare tab.",
        "Adjust filters (year, continent, grade) to control the scope of comparison.",
        "Review the headline KPIs: Average Viability Score vs. Total CAPEX ($B) for each country.",
        "Use insights here as a starting point, then if among the top 10 investing countries, explore the Industry Landscape and Target Countries tabs for deeper context."
    ]

    _benchmarking_explainer_block(
        what=what_bench,
        why=why_bench,
        how=how_bench,
    )

   # 7) FDI Forecasts (2025–2028)
    _anchor(*SECTIONS["forecast"])
    
    what_forecast = [
        "Forward-looking projection of country-level FDI CAPEX for 2025–2028.",
        "Built using ARIMA-family time-series models trained on historical CAPEX data."
    ]
    
    why_forecast = [
    "Provides insight into whether countries are likely to gain or lose momentum in attracting investment.",
    "Supports prioritization by comparing future trajectories across peer countries, not just current levels.",
    "Adds a predictive layer to complement the composite score and past CAPEX analysis."
    ]

    how_forecast = [
    (
        "<p>Each forecast is generated using ARIMA-family models:</p>"
        "<ul style='margin-top:6px'>"
        "<li><strong>ARIMA</strong>: based only on past CAPEX values.</li>"
        "<li><strong>ARIMAX</strong>: ARIMA extended with exogenous (economic/governance) indicators.</li>"
        "<li><strong>SARIMA</strong>: adds seasonal/cyclical patterns.</li>"
        "<li><strong>SARIMAX</strong>: seasonality + exogenous indicators.</li>"
        "</ul>"
    ),
    (
        "<p>The <em>order</em> <code>(p,d,q)</code> shown under the chart means the model:</p>"
        "<ul style='margin-top:6px'>"
        "<li>looks back at <em>p</em> past values (autoregressive part),</li>"
        "<li>differences the series <em>d</em> times to remove trends,</li>"
        "<li>and models <em>q</em> past shocks/noise (moving-average part).</li>"
        "</ul>"
    ),
    (
        "<p><strong>RMSE</strong> (Root Mean Squared Error) is the test-window error; "
        "lower is better.</p>"
        "<p>Dashed lines = forecasts for 2025–2028; solid lines = historical CAPEX ($B).</p>"
    ),
    ]



    _benchmarking_explainer_block(
        what=what_forecast,
        why=why_forecast,
        how=how_forecast,
    )

# ---- Aliases so one ℹ️ per tab can jump to the right section ----------------
SECTIONS.update({
    # one-button-per-tab keys → map to your existing anchors
    "scoring_tab":       SECTIONS["score_trend"],
    "capex_tab":         SECTIONS["capex_trend"],
    "sectors_tab":       SECTIONS["investment_profile"],
    "destinations_tab":  SECTIONS["investment_profile"],
    "compare_tab":       SECTIONS["compare"],
    "forecast_tab":      SECTIONS["forecast"],
})

# ---- Public helper: ℹ️ button you can use from any tab ----------------------
def info_button(section_key: str,
                help_text: str = "What is this?",
                key_suffix: str | None = None):
    """
    Render a small ℹ️ button that jumps to `section_key` in Overview.
    Pass a UNIQUE `key_suffix` per call (e.g., 'sectors_hdr', 'dest_hdr').
    """
    # Use the caller-provided suffix verbatim to avoid accidental collisions.
    if not key_suffix:
        # Absolute fallback to keep the app running, but you SHOULD pass a suffix.
        key_suffix = f"default_{section_key}"

    unique_key = f"info_{key_suffix}"
    if st.button("ℹ️", key=unique_key, help=help_text):
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
