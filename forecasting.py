# forecasting.py
# ─────────────────────────────────────────────────────────────────────────────
# Unified chart:
# - History (…–2023) = light grey
# - 2024 is FORECASTED but drawn in light grey (with a single "CAPEX:" hover)
# - 2025–2028 forecasts = dark blue
# - Clean axis (every year tick), no gridlines, continuous line
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import quote
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from overview import info_button, emit_auto_jump_script


# --- Slider helpers: left handle free (2004..2025), right handle fixed at last year ---
def _init_hist_slider_state(fixed_end: int, default_start: int = 2015):
    """Initialize session_state for the slider widget value and the 'locked' range used for plotting."""
    if "hist_slider_value" not in st.session_state:
        st.session_state.hist_slider_value = (default_start, fixed_end)
    if "hist_range_locked" not in st.session_state:
        st.session_state.hist_range_locked = (default_start, fixed_end)

def _on_hist_slider_change(fixed_end: int):
    """
    Range slider callback:
    - Clamp LEFT handle between 2004 and 2025
    - Force RIGHT handle to fixed_end
    - Mirror to hist_range_locked (used by the plot)
    """
    left, _right = st.session_state.hist_slider_value
    left = int(max(2004, min(left, 2025)))
    st.session_state.hist_slider_value = (left, fixed_end)   # snap UI
    st.session_state.hist_range_locked = (left, fixed_end)   # value for plotting
    

RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    "final_clean": "final_capex_and_indicators_cleaned.csv",
    "forecasting_final_clean": "forecasting_final_capex_and_indicators_cleaned.csv",
    "combined": "combined_capex_and_indicators_filtered.csv",
    "capex_long": "capex2003-2025.csv",
    "indicators": "indicators.csv",
}

EXOG_DEFAULT = [
    "Exports of goods and services (% of GDP)",
    "GDP per capita (current US$)",
    "Imports of goods and services (% of GDP)",
    "Political Stability and Absence of Violence/Terrorism: Estimate",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _adaptive_test_horizon(n_total: int) -> int:
    # 15% of the series, capped to [2, 4]
    return max(2, min(4, int(np.ceil(0.15 * n_total))))

def _raw(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def _numify(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = str(x).replace(",", "").strip()
    s = re.sub(r"[^\d\.\-]", "", s)
    try: return float(s)
    except Exception: return np.nan

def _find_col(cols, *cands):
    low = {str(c).lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    for cand in cands:
        for col in cols:
            if cand.lower() in str(col).lower(): return col
    return None

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # evaluate on billions (consistent with display)
    y_true = np.asarray(y_true, dtype=float) / 1000.0
    y_pred = np.asarray(y_pred, dtype=float) / 1000.0
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

# ── data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=True)
def _load_notebook_style_panel() -> pd.DataFrame:
    df = None
    for key in ("final_clean", "forecasting_final_clean", "combined"):
        try:
            tmp = pd.read_csv(_raw(FILES[key]))
            ctry = _find_col(tmp.columns, "Country", "country")
            year = _find_col(tmp.columns, "Year", "year")
            cap  = _find_col(tmp.columns, "CAPEX", "capex")
            if not (ctry and year and cap):
                raise ValueError("Missing Country/Year/CAPEX columns.")
            tmp = tmp.rename(columns={ctry: "Country", year: "Year", cap: "CAPEX"}).copy()
            tmp["CAPEX"] = tmp["CAPEX"].map(_numify)
            df = tmp
            break
        except Exception:
            df = None

    if df is None:
        cap = pd.read_csv(_raw(FILES["capex_long"]))
        ctry = _find_col(cap.columns, "Source Country", "Country", "source_country", "Source Co")
        if not ctry:
            for c in cap.columns:
                if "country" in str(c).lower():
                    ctry = c; break
        if not ctry:
            raise RuntimeError("No country column in capex file.")
        year_cols = [c for c in cap.columns if re.fullmatch(r"\d{4}", str(c))]
        if not year_cols:
            raise RuntimeError("No 4-digit year columns in capex file.")
        m = cap.melt(id_vars=[ctry], value_vars=year_cols, var_name="Year", value_name="CAPEX")
        m = m.rename(columns={ctry: "Country"})
        m["Year"] = pd.to_numeric(m["Year"], errors="coerce").astype("Int64")
        m["CAPEX"] = m["CAPEX"].map(_numify)
        m["Country"] = m["Country"].astype(str).str.strip()
        cap_long = m.dropna(subset=["Year"]).copy()

        try:
            ind = pd.read_csv(_raw(FILES["indicators"]))
            ctry_i = _find_col(ind.columns, "Country", "Country Name")
            year_i = _find_col(ind.columns, "Year")
            if not (ctry_i and year_i):
                raise ValueError("Indicators missing Country/Year.")
            ind = ind.rename(columns={ctry_i: "Country", year_i: "Year"})
            ind["Country"] = ind["Country"].astype(str).str.strip()
            ind["Year"] = pd.to_numeric(ind["Year"], errors="coerce").astype("Int64")
        except Exception:
            df = cap_long
        else:
            df = cap_long.merge(ind, on=["Country", "Year"], how="left")

    # basic cleanups
    df = df[df["Year"] != 2003].copy()
    miss = df[df["CAPEX"].isna()]
    if not miss.empty:
        bad_countries = miss["Country"].unique().tolist()
        df = df[~df["Country"].isin(bad_countries)].copy()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Country"] = df["Country"].astype(str).str.strip()
    return df

# ── modeling ─────────────────────────────────────────────────────────────────

def _prep_country_notebook(df_all: pd.DataFrame, country: str):
    d = df_all[df_all["Country"] == country].copy().sort_values("Year")
    d["CAPEX"] = d["CAPEX"].map(_numify)
    d = d.dropna(subset=["CAPEX"])
    d = d[d["CAPEX"] > 0]
    if d.shape[0] < 6:
        raise ValueError("Not enough datapoints after cleaning (need ≥ 6).")

    years = d["Year"].astype(int).values
    endog_log = pd.Series(np.log(d["CAPEX"].values), index=years, name="log_CAPEX")

    # exogenous (optional)
    exog_cols = []
    for col in EXOG_DEFAULT:
        hit = _find_col(d.columns, col)
        if hit: exog_cols.append(hit)

    exog_raw = None
    if exog_cols:
        xr = d[exog_cols].apply(pd.to_numeric, errors="coerce")
        xr = xr.interpolate(limit_direction="both")
        exog_raw = pd.DataFrame(xr.values, index=years, columns=exog_cols)

    # train/test split for model selection
    n = len(endog_log)
    split_years = _adaptive_test_horizon(n)
    train_idx = years[: n - split_years]
    test_idx  = years[n - split_years :]

    train_y = endog_log.loc[train_idx]
    test_y  = endog_log.loc[test_idx]

    if exog_raw is not None:
        scaler = StandardScaler()
        train_x_raw = exog_raw.loc[train_idx]
        test_x_raw  = exog_raw.loc[test_idx]

        train_x = pd.DataFrame(scaler.fit_transform(train_x_raw),
                               index=train_x_raw.index, columns=train_x_raw.columns)
        test_x  = pd.DataFrame(scaler.transform(test_x_raw),
                               index=test_x_raw.index, columns=test_x_raw.columns)
        exog_full = pd.DataFrame(scaler.transform(exog_raw),
                                 index=exog_raw.index, columns=exog_raw.columns)
        last_row_scaled = exog_full.loc[[exog_full.index.max()]]
    else:
        train_x = test_x = exog_full = None
        last_row_scaled = None

    last_year = int(max(years))

    # Forecast horizon: 2024–2028 (only years > last observed)
    requested_years = [2024, 2025, 2026, 2027, 2028]
    future_years = [y for y in requested_years if y > last_year]
    future_index = pd.Index(future_years, name="Year")

    if last_row_scaled is not None and len(future_years) > 0:
        future_exog = pd.DataFrame(
            np.repeat(last_row_scaled.values, repeats=len(future_years), axis=0),
            columns=last_row_scaled.columns, index=future_index
        )
    else:
        future_exog = None

    return {
        "capex_actual": pd.Series(d["CAPEX"].values / 1000.0, index=years, name="CAPEX"),
        "endog_log": endog_log,
        "train_y": train_y, "test_y": test_y,
        "train_x": train_x, "test_x": test_x,
        "exog_full": exog_full,
        "future_index": future_index, "future_exog": future_exog,
    }

def _fit_eval_arima(train_y, test_y):
    best = {"name": "ARIMA", "order": None, "rmse": np.inf, "fit": None}
    for p in range(0, 3):
        for d in range(0, 3):
            for q in range(0, 3):
                try:
                    res = ARIMA(train_y, order=(p, d, q)).fit()
                    pred_log = res.forecast(steps=len(test_y))
                    rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                    if rmse < best["rmse"]:
                        best.update({"order": (p, d, q), "rmse": rmse, "fit": res})
                except Exception:
                    continue
    return best

def _fit_eval_arimax(train_y, test_y, train_x, test_x):
    if train_x is None or test_x is None:
        return {"name": "ARIMAX", "order": None, "rmse": np.inf, "fit": None}
    best = {"name": "ARIMAX", "order": None, "rmse": np.inf, "fit": None}
    for p in range(0, 4):
        for d in range(0, 4):
            for q in range(0, 4):
                try:
                    res = ARIMA(train_y, exog=train_x, order=(p, d, q)).fit()
                    pred_log = res.forecast(steps=len(test_y), exog=test_x)
                    rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                    if rmse < best["rmse"]:
                        best.update({"order": (p, d, q), "rmse": rmse, "fit": res})
                except Exception:
                    continue
    return best

def _fit_eval_sarima(train_y, test_y):
    best = {"name": "SARIMA", "order": None, "seasonal": None, "rmse": np.inf, "fit": None}
    for p in range(0, 2):
        for d in range(0, 2):
            for q in range(0, 2):
                for P in range(0, 2):
                    for D in range(0, 2):
                        for Q in range(0, 2):
                            try:
                                res = SARIMAX(train_y, order=(p, d, q), seasonal_order=(P, D, Q, 1)).fit(disp=False)
                                pred_log = res.forecast(steps=len(test_y))
                                rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                                if rmse < best["rmse"]:
                                    best.update({"order": (p, d, q), "seasonal": (P, D, Q, 1), "rmse": rmse, "fit": res})
                            except Exception:
                                continue
    return best

def _fit_eval_sarimax(train_y, test_y, train_x, test_x):
    if train_x is None or test_x is None:
        return {"name": "SARIMAX", "order": None, "seasonal": None, "rmse": np.inf, "fit": None}
    best = {"name": "SARIMAX", "order": None, "seasonal": None, "rmse": np.inf, "fit": None}
    for p in range(0, 2):
        for d in range(0, 2):
            for q in range(0, 2):
                for P in range(0, 2):
                    for D in range(0, 2):
                        for Q in range(0, 2):
                            try:
                                res = SARIMAX(train_y, exog=train_x, order=(p, d, q),
                                              seasonal_order=(P, D, Q, 1)).fit(disp=False)
                                pred_log = res.forecast(steps=len(test_y), exog=test_x)
                                rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                                if rmse < best["rmse"]:
                                    best.update({"order": (p, d, q), "seasonal": (P, D, Q, 1), "rmse": rmse, "fit": res})
                            except Exception:
                                continue
    return best

def _refit_and_forecast_full(best_model: dict, endog_log: pd.Series,
                             exog_full: pd.DataFrame, future_index: pd.Index,
                             future_exog: pd.DataFrame):
    steps = len(future_index)
    if steps == 0:
        return pd.Series([], dtype=float, name="forecast")

    name = best_model["name"]
    if name == "ARIMA":
        final = ARIMA(endog_log, order=best_model["order"]).fit()
        future_log = final.forecast(steps=steps)
    elif name == "ARIMAX":
        final = ARIMA(endog_log, exog=exog_full.loc[endog_log.index], order=best_model["order"]).fit()
        future_log = final.forecast(steps=steps, exog=future_exog)
    elif name == "SARIMA":
        final = SARIMAX(endog_log, order=best_model["order"], seasonal_order=best_model["seasonal"]).fit(disp=False)
        future_log = final.forecast(steps=steps)
    else:  # SARIMAX
        final = SARIMAX(endog_log, exog=exog_full.loc[endog_log.index],
                        order=best_model["order"], seasonal_order=best_model["seasonal"]).fit(disp=False)
        future_log = final.forecast(steps=steps, exog=future_exog)

    future = pd.Series(np.exp(future_log).values / 1000.0, index=future_index, name="forecast")
    return future

# ── unified plotting ─────────────────────────────────────────────────────────

def _plot_forecast_unified(country: str,
                           actual: pd.Series,
                           future_idx: pd.Index,
                           future_pred: pd.Series,
                           best_name: str,
                           rmse: float,
                           start_year: int = 2015):
    """
    Single-axis design:
      • Light grey for history (start_year–2023)
      • 2024 forecast drawn in light grey (single tooltip: "CAPEX: …")
      • Dark blue for 2025–2028 forecasts
      • Every year tick (dtick=1), no gridlines
    """
    # Prepare data
    hist_years = [int(y) for y in actual.index if start_year <= int(y) <= 2023]
    hist_vals  = [float(actual.loc[y]) for y in hist_years]

    f_years_all = list(map(int, future_idx.values)) if len(future_idx) > 0 else []
    f_pred = future_pred  # Series indexed by future years

    fig = make_subplots(rows=1, cols=1)

    # History — light grey up to 2023
    if hist_years:
        fig.add_trace(
            go.Scatter(
                x=hist_years, y=hist_vals,
                mode="lines",
                line=dict(color="rgba(120,120,120,0.75)", width=2.0, shape="linear"),
                name=f"Actual ({start_year}–2023)",
                hovertemplate="Year: %{x}<br>CAPEX: %{y:.4f} $B<extra></extra>",
                showlegend=False,
            )
        )

    # 2024 forecast shown in GREY with a single "CAPEX:" tooltip
    has_f_2024 = (2024 in f_years_all)
    if has_f_2024:
        y2024f = float(f_pred.loc[2024])
        # connector 2023->2024 with NO hover (avoids duplicate tooltip at 2023)
        if 2023 in actual.index:
            fig.add_trace(
                go.Scatter(
                    x=[2023, 2024],
                    y=[float(actual.loc[2023]), y2024f],
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.75)", width=2.0, shape="linear"),
                    showlegend=False,
                    hoverinfo="skip"  # suppress connector hover
                )
            )
        # a single grey marker at 2024 that carries the tooltip "CAPEX: …"
        fig.add_trace(
            go.Scatter(
                x=[2024],
                y=[y2024f],
                mode="markers",
                marker=dict(size=6, color="rgba(120,120,120,0.9)"),
                showlegend=False,
                hovertemplate="Year: %{x}<br>CAPEX: %{y:.4f} $B<extra></extra>",
            )
        )

    # Forecast — dark blue from 2025 onward
    f_years_2528 = [y for y in f_years_all if y >= 2025]
    if f_years_2528:
        f_vals_2528 = [float(f_pred.loc[y]) for y in f_years_2528]

        # Draw a connector from 2024 (if forecasted) or 2023 (if no 2024 forecast) to 2025 with NO hover
        if has_f_2024:
            anchor_year, anchor_val = 2024, float(f_pred.loc[2024])
        elif 2023 in actual.index:
            anchor_year, anchor_val = 2023, float(actual.loc[2023])
        else:
            anchor_year = None

        if anchor_year is not None:
            fig.add_trace(
                go.Scatter(
                    x=[anchor_year, 2025],
                    y=[anchor_val, f_vals_2528[0]],
                    mode="lines",
                    line=dict(color="#2E8EF7", width=2.4, shape="linear"),
                    showlegend=False,
                    hoverinfo="skip"  # avoid a second tooltip at 2024 or 2025
                )
            )

        # Main blue forecast line 2025–2028 (with forecast tooltip)
        fig.add_trace(
            go.Scatter(
                x=f_years_2528, y=f_vals_2528,
                mode="lines",
                line=dict(color="#2E8EF7", width=2.4, shape="linear"),
                name="Forecast (2025–2028)",
                hovertemplate="Year: %{x}<br>CAPEX (forecast): %{y:.4f} $B<extra></extra>",
                showlegend=False,
            )
        )

    # X span — show every year tick
    xmax_candidates = []
    if hist_years: xmax_candidates.append(max(hist_years))
    if has_f_2024: xmax_candidates.append(2024)
    if f_years_2528: xmax_candidates.append(max(f_years_2528))
    xmax = max(xmax_candidates) if xmax_candidates else 2028

    fig.update_xaxes(
        tickmode="linear",
        tick0=start_year, dtick=1,
        tickangle=0,
        range=[start_year - 0.5, xmax + 0.5],
        showgrid=False,
        title_text=""
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title_text=""
    )

    fig.update_layout(
        title=f"{best_name} Forecast for {country} | RMSE: {rmse:.2f} $B",
        hovermode="x",
        hoverlabel=dict(bgcolor="white", font_size=12, font_color="black"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig

# ── public entrypoint ────────────────────────────────────────────────────────

def render_forecasting_tab():
    _f_left, _f_right = st.columns([20, 1], gap="small")
    with _f_left:
        st.caption("Forecasts — 2024–2028")
    with _f_right:
        info_button("forecast")

    emit_auto_jump_script()

    panel = _load_notebook_style_panel()
    if panel.empty:
        st.info("No forecasting data available.")
        return

    countries = sorted(panel["Country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", countries, index=0, key="forecast_country_unified")

    try:
        prep = _prep_country_notebook(panel, sel_country)
    except Exception as e:
        st.error(f"Could not prepare data: {e}")
        return

    train_y, test_y = prep["train_y"], prep["test_y"]
    train_x, test_x = prep["train_x"], prep["test_x"]

    # model candidates
    cand = [
        _fit_eval_arima(train_y, test_y),
        _fit_eval_arimax(train_y, test_y, train_x, test_x),
        _fit_eval_sarima(train_y, test_y),
        _fit_eval_sarimax(train_y, test_y, train_x, test_x),
    ]
    best = min(cand, key=lambda d: d["rmse"])
    best_name = best["name"]

    future_pred = _refit_and_forecast_full(
        best, prep["endog_log"], prep["exog_full"], prep["future_index"], prep["future_exog"]
    )

    
    # Fixed last year (right end) = last forecast year if available, else 2028
    fixed_end = int(prep["future_index"][-1]) if len(prep["future_index"]) else 2028
    
    Toggle = getattr(st, "toggle", st.checkbox)
    show_more_hist = Toggle(
        "Show earlier history",
        value=False,
        help="Drag the LEFT handle to include earlier years. The right handle is locked."
    )
    
    if show_more_hist:
        _init_hist_slider_state(fixed_end, default_start=2015)
    
        st.slider(
            "Range of Years shown",
            min_value=2004,
            max_value=fixed_end,
            value=st.session_state.hist_slider_value,  # widget state (separate key)
            step=1,
            key="hist_slider_value",
            on_change=_on_hist_slider_change,
            args=(fixed_end,),
            help="Left handle moves (down to 2004, up to 2025). Right handle is fixed."
        )
    
        start_year = int(st.session_state.hist_range_locked[0])
    
    
    else:
        start_year = 2015

    fig = _plot_forecast_unified(
        sel_country,
        prep["capex_actual"],
        prep["future_index"],
        future_pred,
        best_name,
        best["rmse"],
        start_year=start_year
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Best model:** `{best_name}`")
        if best.get("order") is not None:
            st.markdown(f"**Order:** `{best['order']}`")
        if best.get("seasonal") is not None:
            st.markdown(f"**Seasonal:** `{best['seasonal']}`")
    with right:
        st.markdown(f"**RMSE (test):** `{best['rmse']:.2f} $B`")
        if len(prep["future_index"]) > 0:
            st.markdown(f"**Forecast horizon:** `{int(prep['future_index'][0])}–{int(prep['future_index'][-1])}`")
        else:
            st.markdown("**Forecast horizon:** `No future years beyond last observed data`")
