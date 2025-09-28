# forecasting.py
# ─────────────────────────────────────────────────────────────────────────────
# Forecasting tab aligned to notebook behavior, with simplified plot:
# - Data prep: prefer final cleaned CSVs; drop 2003; remove countries with any missing CAPEX
# - Exog scaling: StandardScaler fit on FULL series (pre-split) to match notebook
# - Split: last 4 years (for RMSE selection only)
# - Forecast horizon: EXACTLY 2025–2028
# - Plot: ONLY Actual CAPEX + Forecast (no fitted/test lines)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import quote
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from overview import info_button, emit_auto_jump_script



RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    # Notebook-preferred files
    "final_clean": "final_capex_and_indicators_cleaned.csv",
    "forecasting_final_clean": "forecasting_final_capex_and_indicators_cleaned.csv",
    # Older app sources as fallbacks
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
    # 15% bounded to [2, 4] — matches the new notebook
    return max(2, min(4, int(np.ceil(0.15 * n_total))))

def _raw(fname: str) -> str:
    return f"{RAW_BASE}/{quote(fname)}"

def _numify(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", "").strip()
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

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

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float) / 1000.0
    y_pred = np.asarray(y_pred, dtype=float) / 1000.0
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


# ── data loading (notebook-aligned) ──────────────────────────────────────────

@st.cache_data(show_spinner=True)
def _load_notebook_style_panel() -> pd.DataFrame:
    """
    Load the same data the notebook trains on when available, else fall back.
    Normalize to: Country, Year, CAPEX, plus indicators/exog columns.
    Apply notebook cleaning (drop 2003; drop countries with any missing CAPEX).
    """
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
        # Fallback: merge capex_long + indicators (as in original app)
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

    # Notebook cleaning rules
    # - Drop year 2003
    df = df[df["Year"] != 2003].copy()

    # - Remove any country that still has missing CAPEX
    miss = df[df["CAPEX"].isna()]
    if not miss.empty:
        bad_countries = miss["Country"].unique().tolist()
        df = df[~df["Country"].isin(bad_countries)].copy()

    # Coerce year to int and tidy
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Country"] = df["Country"].astype(str).str.strip()
    return df

# ── modeling (notebook grids & choices) ──────────────────────────────────────

def _prep_country_notebook(df_all: pd.DataFrame, country: str):
    """
    Matches new notebook prep (no leakage + adaptive split):
      - endog = log(CAPEX) with CAPEX>0
      - exog = StandardScaler fit on TRAIN ONLY
      - split = adaptive last 15% (bounded 2–4 years)
      - future years = EXACT 2025–2028 (repeat last TRAIN-based scaled row)
    """
    d = df_all[df_all["Country"] == country].copy().sort_values("Year")
    d["CAPEX"] = d["CAPEX"].map(_numify)
    d = d.dropna(subset=["CAPEX"])
    d = d[d["CAPEX"] > 0]
    if d.shape[0] < 6:
        raise ValueError("Not enough datapoints after cleaning (need ≥ 6).")

    # endog on year index
    years = d["Year"].astype(int).values
    endog_log = pd.Series(np.log(d["CAPEX"].values), index=years, name="log_CAPEX")

    # exog columns (if present)
    exog_cols = []
    for col in EXOG_DEFAULT:
        hit = _find_col(d.columns, col)
        if hit:
            exog_cols.append(hit)

    exog_raw = None
    if exog_cols:
        xr = d[exog_cols].apply(pd.to_numeric, errors="coerce")
        xr = xr.interpolate(limit_direction="both")
        exog_raw = pd.DataFrame(xr.values, index=years, columns=exog_cols)

    # --- adaptive split (like notebook) ---
    n = len(endog_log)
    split_years = _adaptive_test_horizon(n)
    train_idx = years[: n - split_years]
    test_idx  = years[n - split_years :]

    train_y = endog_log.loc[train_idx]
    test_y  = endog_log.loc[test_idx]

    # --- TRAIN-only scaling (no leakage) ---
    if exog_raw is not None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_x_raw = exog_raw.loc[train_idx]
        test_x_raw  = exog_raw.loc[test_idx]

        train_x = pd.DataFrame(scaler.fit_transform(train_x_raw),
                               index=train_x_raw.index, columns=train_x_raw.columns)
        test_x  = pd.DataFrame(scaler.transform(test_x_raw),
                               index=test_x_raw.index, columns=test_x_raw.columns)
        exog_full = pd.DataFrame(scaler.transform(exog_raw),
                                 index=exog_raw.index, columns=exog_raw.columns)

        # future exog: repeat last observed (already scaled with TRAIN stats)
        last_row_scaled = exog_full.loc[[exog_full.index.max()]]
    else:
        train_x = test_x = exog_full = None
        last_row_scaled = None

    # --- future horizon: exactly 2025–2028 (only > last observed) ---
    last_year = int(max(years))
    requested_years = [2025, 2026, 2027, 2028]
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
        "capex_actual": pd.Series(d["CAPEX"].values / 1000.0, index=years, name="CAPEX"),  # <-- scaled to $B
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
    """
    Refit best model on full series; forecast future_index length.
    Returns: future (Series indexed by Year). No fitted values are returned/used.
    """
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

    future = pd.Series(np.exp(future_log).values / 1000.0, index=future_index, name="forecast")  # <-- scaled to $B
    return future

def _plot_forecast_only(country: str,
                        actual: pd.Series,
                        future_idx: pd.Index,
                        future_pred: pd.Series,
                        best_name: str,
                        rmse: float):
    """
    Two-panel layout:
      - Row 1: full-history sparkline (faded, thin) for context
      - Row 2: focus on recent years (2018 -> last forecast), thick line + dashed forecast,
               band over forecast horizon (2025–2028).
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ---- settings you can tweak ----
    focus_start = 2018
    band_start  = 2025
    band_end    = 2028

    color_hist_faded = "rgba(60,60,60,0.35)"
    color_hist_main  = "rgba(35,35,35,0.95)"
    color_forecast   = "rgba(33,150,243,0.95)"
    band_color       = "rgba(33,150,243,0.10)"

    # convert X to datetimes (nicer axis control)
    x_hist = pd.to_datetime(pd.Series(actual.index.astype(int), name="year"), format="%Y")
    y_hist = actual.values

    x_fc = pd.to_datetime(pd.Series(future_idx.astype(int)), format="%Y") if len(future_idx) else pd.Series([], dtype="datetime64[ns]")
    y_fc = future_pred.values if len(future_idx) else []

    # boundaries
    min_year = int(x_hist.dt.year.min())
    max_year = int((pd.concat([x_hist, x_fc]) if len(x_fc) else x_hist).dt.year.max())

    # subplots: small top sparkline + large focus plot
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, shared_yaxes=True,
        row_heights=[0.26, 0.74], vertical_spacing=0.06
    )

    # ---------- Row 1: full-history sparkline (faded) ----------
    fig.add_trace(go.Scatter(
        x=x_hist, y=y_hist, mode="lines",
        line=dict(color=color_hist_faded, width=2),
        hovertemplate="Year: %{x|%Y}<br>FDI: %{y:.4f} $B<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    # sparse ticks (every ~3–5 years depending on span)
    span = max_year - min_year
    step = 5 if span > 20 else 3
    ticks_top = pd.to_datetime([f"{y}-01-01" for y in range(min_year, max_year + 1, step)])

    fig.update_xaxes(
        row=1, col=1,
        tickmode="array",
        tickvals=ticks_top,
        tickformat="%Y",
        showgrid=False,
        title=None
    )
    fig.update_yaxes(row=1, col=1, title=None, showgrid=False)

    # ---------- Row 2: focus panel ----------
    # recent actuals (>= focus_start)
    mask_focus = x_hist.dt.year >= focus_start
    if mask_focus.any():
        fig.add_trace(go.Scatter(
            x=x_hist[mask_focus], y=y_hist[mask_focus], mode="lines",
            line=dict(color=color_hist_main, width=3),
            hovertemplate="Year: %{x|%Y}<br>FDI: %{y:.4f} $B<extra></extra>",
            showlegend=False
        ), row=2, col=1)

    # forecast (dashed)
    if len(x_fc) > 0:
        fig.add_trace(go.Scatter(
            x=x_fc, y=y_fc, mode="lines",
            line=dict(color=color_forecast, width=3, dash="dash"),
            hovertemplate="Year: %{x|%Y}<br>FDI (forecast): %{y:.4f} $B<extra></extra>",
            showlegend=False
        ), row=2, col=1)

        # band over 2025–2028
        fig.add_vrect(
            x0=f"{band_start}-01-01", x1=f"{band_end}-12-31",
            fillcolor=band_color, line_width=0, layer="below",
            row=2, col=1
        )

    # bottom axis: yearly ticks from max(focus_start,min_year) to max_year
    left = max(focus_start, min_year)
    ticks_bottom = pd.to_datetime([f"{y}-01-01" for y in range(left, max_year + 1)])

    fig.update_xaxes(
        row=2, col=1,
        tickmode="array", tickvals=ticks_bottom, tickformat="%Y",
        showgrid=False, title=None,
        range=[pd.to_datetime(f"{left}-01-01"), pd.to_datetime(f"{max_year}-12-31")]
    )
    fig.update_yaxes(row=2, col=1, title=None, showgrid=True)

    # ---------- overall layout ----------
    fig.update_layout(
        title=f"{best_name} Forecast for {country} | RMSE: {rmse:.2f} $B",
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        height=560
    )

    return fig


# ── public entrypoint ────────────────────────────────────────────────────────

def render_forecasting_tab():
    # Top bar: caption + info button (opens Overview → FDI Forecasts)
    _f_left, _f_right = st.columns([20, 1], gap="small")
    with _f_left:
        st.caption("Forecasts — 2025–2028")
    with _f_right:
        info_button("forecast")  # scrolls to 'FDI Forecasts (2025–2028)' in Overview

    # Ensure the auto-jump script is emitted (safe to call more than once)
    emit_auto_jump_script()


    panel = _load_notebook_style_panel()
    if panel.empty:
        st.info("No forecasting data available.")
        return

    countries = sorted(panel["Country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", countries, index=0, key="forecast_country_forecastonly")

    # Prep data (notebook style)
    try:
        prep = _prep_country_notebook(panel, sel_country)
    except Exception as e:
        st.error(f"Could not prepare data: {e}")
        return

    # Model selection on the last-4-years test (no plotting of fitted/test lines)
    train_y, test_y = prep["train_y"], prep["test_y"]
    train_x, test_x = prep["train_x"], prep["test_x"]

    cand = []
    cand.append(_fit_eval_arima(train_y, test_y))
    cand.append(_fit_eval_arimax(train_y, test_y, train_x, test_x))
    cand.append(_fit_eval_sarima(train_y, test_y))
    cand.append(_fit_eval_sarimax(train_y, test_y, train_x, test_x))
    best = min(cand, key=lambda d: d["rmse"])
    best_name = best["name"]

    # Refit on full & forecast exactly 2025–2028
    future_pred = _refit_and_forecast_full(
        best, prep["endog_log"], prep["exog_full"], prep["future_index"], prep["future_exog"]
    )

    # Plot ONLY Actual + Forecast
    fig = _plot_forecast_only(
        sel_country, prep["capex_actual"], prep["future_index"], future_pred, best_name, best["rmse"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Small summary (kept for clarity)
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
