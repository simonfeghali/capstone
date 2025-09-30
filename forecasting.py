# forecasting.py
# ─────────────────────────────────────────────────────────────────────────────
# Split view:
# - Left (smaller): 2004–2023 actuals, thin/light
#   * X ticks every 3 years, rotated 90°
# - Right (larger): 2025–2028 forecast, bold with markers
# - No shaded area, no connection between panels
# - Panels are close together
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

# ── plotting (split, gap, custom ticks) ──────────────────────────────────────

def _plot_forecast_split_gap(country: str,
                             actual: pd.Series,
                             future_idx: pd.Index,
                             future_pred: pd.Series,
                             best_name: str,
                             rmse: float):
    """
    Two subplots share Y:
      • Left (smaller): actuals 2004–2023, thin/light, ticks every 3 years rotated 90°
      • Right (larger): forecast 2025–2028, clean slim line (no markers), tighter year spacing
      • No connecting line; subtle dotted guide at last actual value across the gap
    """
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        horizontal_spacing=0.004,         # panels very close
        column_widths=[0.36, 0.64]        # squeezed history, wider forecast
    )

    # ── LEFT: actual history (2004–2023), squeezed ───────────────────────────
    if len(actual) > 0:
        left_x = [y for y in actual.index.astype(int) if 2004 <= y <= 2023]
        left_y = [actual.loc[y] for y in left_x]
        if left_x:
            fig.add_trace(
                go.Scatter(
                    x=left_x, y=left_y,
                    mode="lines",
                    line=dict(color="rgba(90,90,90,0.70)", width=1.6),
                    name="Actual (2004–2023)",
                    hovertemplate="Year: %{x}<br>FDI: %{y:.4f} $B<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.update_xaxes(
                tickmode="linear", tick0=2004, dtick=3, tickangle=90,
                range=[min(left_x) - 0.5, max(left_x) + 0.5],
                showgrid=False, title_text="", row=1, col=1
            )
        else:
            fig.update_xaxes(tickmode="linear", tick0=2004, dtick=3, tickangle=90,
                             showgrid=False, title_text="", row=1, col=1)

    # ── RIGHT: forecast only (2025–2028), dominant but clean ─────────────────
    if len(future_idx) > 0:
        right_x = list(map(int, future_idx.values))
        right_y = list(map(float, future_pred.values))

        # single slim line (no markers, no glow)
        fig.add_trace(
            go.Scatter(
                x=right_x, y=right_y,
                mode="lines",
                line=dict(color="#0D2A52", width=2.2, shape="linear"),
                name="Forecast (2025–2028)",
                hovertemplate="Year: %{x}<br>FDI (forecast): %{y:.4f} $B<extra></extra>",
                showlegend=False
            ),
            row=1, col=2
        )

        # tighter year spacing and smaller side margins on the x-range
        fig.update_xaxes(
            tickmode="linear", tick0=right_x[0], dtick=1,
            range=[min(right_x) - 0.15, max(right_x) + 0.15],
            tickangle=0, showgrid=False, title_text="", row=1, col=2
        )

        # subtle continuity guide at the last historical value (not a connector)
        if len(actual) > 0:
            last_hist_years = [y for y in actual.index.astype(int) if y <= 2023]
            if last_hist_years:
                last_hist_val = float(actual.loc[last_hist_years[-1]])
                d1 = fig.layout.xaxis.domain
                d2 = fig.layout.xaxis2.domain
                fig.add_shape(
                    type="line",
                    xref="paper", yref="y",
                    x0=d1[1], x1=d2[0],
                    y0=last_hist_val, y1=last_hist_val,
                    line=dict(color="rgba(90,90,90,0.30)", width=1, dash="dot")
                )
    else:
        fig.update_xaxes(tickmode="linear", dtick=1, showgrid=False, title_text="", row=1, col=2)

    # ── Shared styling ────────────────────────────────────────────────────────
    fig.update_yaxes(showgrid=False, title_text="")

    fig.update_layout(
        title=f"{best_name} Forecast for {country} | RMSE: {rmse:.2f} $B",
        hovermode="x",
        hoverlabel=dict(bgcolor="white", font_size=12, font_color="black"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
        xaxis=dict(tickfont=dict(size=10)),   # left ticks smaller
        xaxis2=dict(tickfont=dict(size=12))   # right ticks slightly larger
    )
    return fig
    
# ── public entrypoint ────────────────────────────────────────────────────────

def render_forecasting_tab():
    _f_left, _f_right = st.columns([20, 1], gap="small")
    with _f_left:
        st.caption("Forecasts — 2025–2028")
    with _f_right:
        info_button("forecast")

    emit_auto_jump_script()

    panel = _load_notebook_style_panel()
    if panel.empty:
        st.info("No forecasting data available.")
        return

    countries = sorted(panel["Country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", countries, index=0, key="forecast_country_split")

    try:
        prep = _prep_country_notebook(panel, sel_country)
    except Exception as e:
        st.error(f"Could not prepare data: {e}")
        return

    train_y, test_y = prep["train_y"], prep["test_y"]
    train_x, test_x = prep["train_x"], prep["test_x"]

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

    fig = _plot_forecast_split_gap(
        sel_country, prep["capex_actual"], prep["future_index"], future_pred, best_name, best["rmse"]
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
