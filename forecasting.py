# forecasting.py
# ─────────────────────────────────────────────────────────────────────────────
# Forecasting tab: ARIMA, ARIMAX, SARIMA, SARIMAX with compact grid search.
# Picks the model with the lowest RMSE (on original CAPEX scale) for each country.
# Uses Plotly for charts; no sklearn/matplotlib needed.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import quote
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    # Preferred combined panel (capex + indicators)
    "combined": "combined_capex_and_indicators_filtered.csv",
    # Fallbacks
    "capex_long": "capex2003-2025.csv",
    "indicators": "indicators.csv",
}

# --- helpers -----------------------------------------------------------------

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
    # exact-insensitive
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    # contains
    for cand in cands:
        for col in cols:
            if cand.lower() in str(col).lower():
                return col
    return None

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

# --- data loading ------------------------------------------------------------

@st.cache_data(show_spinner=True)
def _load_combined_panel() -> pd.DataFrame:
    """
    Try to load a combined panel of CAPEX + indicators.
    Expected columns (case-insensitive):
      Country, Year, CAPEX, plus exogenous columns.
    If not present, merge capex_long + indicators on [Country, Year].
    Returns tidy DataFrame with:
      country, year, capex, and all remaining indicator columns.
    """
    # 1) Preferred combined file
    try:
        df = pd.read_csv(_raw(FILES["combined"]))
        # normalize columns
        ctry = _find_col(df.columns, "country")
        year = _find_col(df.columns, "year")
        cap  = _find_col(df.columns, "capex")
        if not (ctry and year and cap):
            raise ValueError("Combined file missing Country/Year/CAPEX columns.")
        df = df.rename(columns={ctry:"country", year:"year", cap:"capex"}).copy()
        df["country"] = df["country"].astype(str).str.strip()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["capex"] = df["capex"].map(_numify)
        return df
    except Exception:
        pass

    # 2) Fallback: merge capex_long + indicators
    # capex wide -> long
    try:
        cap = pd.read_csv(_raw(FILES["capex_long"]))
        # find country column
        ctry = _find_col(cap.columns, "Source Country", "Country", "source_country", "country", "Source Co")
        if not ctry:
            for c in cap.columns:
                if "country" in str(c).lower():
                    ctry = c; break
        if not ctry:
            raise ValueError("No country column in capex file.")
        year_cols = [c for c in cap.columns if re.fullmatch(r"\d{4}", str(c))]
        if not year_cols:
            raise ValueError("No 4-digit year columns in capex file.")
        m = cap.melt(id_vars=[ctry], value_vars=year_cols, var_name="year", value_name="capex")
        m = m.rename(columns={ctry:"country"})
        m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
        m["capex"] = m["capex"].map(_numify)
        m["country"] = m["country"].astype(str).str.strip()
        cap_long = m.dropna(subset=["year"]).copy()
    except Exception as e:
        raise RuntimeError(f"Could not load capex: {e}")

    try:
        ind = pd.read_csv(_raw(FILES["indicators"]))
        ctry_i = _find_col(ind.columns, "country")
        year_i = _find_col(ind.columns, "year")
        if not (ctry_i and year_i):
            raise ValueError("Indicators missing Country/Year columns.")
        ind = ind.rename(columns={ctry_i:"country", year_i:"year"})
        ind["country"] = ind["country"].astype(str).str.strip()
        ind["year"] = pd.to_numeric(ind["year"], errors="coerce").astype("Int64")
    except Exception as e:
        # if indicators missing, just return capex
        return cap_long

    df = cap_long.merge(ind, on=["country","year"], how="left")
    return df

# --- modeling ----------------------------------------------------------------

EXOG_DEFAULT = [
    "Exports of goods and services (% of GDP)",
    "GDP per capita (current US$)",
    "Imports of goods and services (% of GDP)",
    "Political Stability and Absence of Violence/Terrorism: Estimate",
]

def _prep_country(df_all: pd.DataFrame, country: str):
    """Return series endog (log CAPEX), train/test split, standardized exog, and future exog."""
    d = df_all[df_all["country"] == country].copy()
    d = d.dropna(subset=["year"]).sort_values("year")
    # endog
    d["capex"] = d["capex"].map(_numify)
    d = d.dropna(subset=["capex"])
    d = d[d["capex"] > 0]  # avoid log(0)
    if d.shape[0] < 6:
        raise ValueError("Not enough datapoints (need ≥ 6 after cleaning).")

    # try to find exog columns; use defaults if available, else use all numeric leftovers (excluding capex)
    exog_cols = []
    for col in EXOG_DEFAULT:
        hit = _find_col(d.columns, col)
        if hit: exog_cols.append(hit)
    if not exog_cols:
        # fallback: any numeric, excluding capex/year
        candidates = [c for c in d.columns if c not in {"country","year","capex"}]
        numers = []
        for c in candidates:
            vc = pd.to_numeric(d[c], errors="coerce")
            if vc.notna().sum() >= 6:
                d[c] = vc
                numers.append(c)
        exog_cols = numers

    # align index by year (int)
    d["year"] = d["year"].astype(int)
    d = d.set_index("year")
    endog_log = np.log(d["capex"])

    exog = None
    if exog_cols:
        exog = d[exog_cols].apply(pd.to_numeric, errors="coerce")
        # simple forward/backward fill for internal gaps
        exog = exog.sort_index().interpolate(limit_direction="both")

    # split: last 4 years test (min 2)
    split_years = min(4, max(2, int(np.ceil(0.15 * len(endog_log)))))
    train_endog, test_endog = endog_log.iloc[:-split_years], endog_log.iloc[-split_years:]
    if exog is not None and not exog.empty:
        train_exog, test_exog = exog.loc[train_endog.index], exog.loc[test_endog.index]
        # standardize using train stats ONLY
        mean = train_exog.mean()
        std = train_exog.std(ddof=0).replace(0, np.nan)
        train_exog_std = (train_exog - mean) / std
        test_exog_std  = (test_exog  - mean) / std
        # future exog: repeat last observed exog row
        last_row = exog.loc[[exog.index.max()]]
        future_years = 3
        future_index = pd.Index([int(d.index.max()) + i for i in range(1, future_years + 1)], name="year")
        future_exog_raw = pd.DataFrame(np.repeat(last_row.values, repeats=future_years, axis=0),
                                       columns=exog_cols, index=future_index)
        future_exog_std = (future_exog_raw - mean) / std
    else:
        train_exog_std = test_exog_std = future_exog_std = None
        future_index = pd.Index([int(d.index.max()) + i for i in range(1, 3 + 1)], name="year")

    return {
        "full_endog_log": endog_log,
        "train_endog": train_endog, "test_endog": test_endog,
        "train_exog": train_exog_std, "test_exog": test_exog_std,
        "future_index": future_index, "future_exog": future_exog_std,
        "exog_cols": exog_cols,
        "capex_actual": d["capex"],
    }

def _fit_eval_arima(train_y, test_y):
    best = {"name":"ARIMA", "order": None, "rmse": np.inf, "fit": None}
    for p in range(0, 3):
        for d in range(0, 3):
            for q in range(0, 3):
                try:
                    model = ARIMA(train_y, order=(p,d,q))
                    res = model.fit()
                    pred_log = res.forecast(steps=len(test_y))
                    rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                    if rmse < best["rmse"]:
                        best.update({"order": (p,d,q), "rmse": rmse, "fit": res})
                except Exception:
                    continue
    return best

def _fit_eval_arimax(train_y, test_y, train_x, test_x):
    if train_x is None or test_x is None:
        return {"name":"ARIMAX", "order": None, "rmse": np.inf, "fit": None}
    best = {"name":"ARIMAX", "order": None, "rmse": np.inf, "fit": None}
    for p in range(0, 3):
        for d in range(0, 3):
            for q in range(0, 3):
                try:
                    model = ARIMA(train_y, exog=train_x, order=(p,d,q))
                    res = model.fit()
                    pred_log = res.forecast(steps=len(test_y), exog=test_x)
                    rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                    if rmse < best["rmse"]:
                        best.update({"order": (p,d,q), "rmse": rmse, "fit": res})
                except Exception:
                    continue
    return best

def _fit_eval_sarima(train_y, test_y):
    best = {"name":"SARIMA", "order": None, "seasonal": None, "rmse": np.inf, "fit": None}
    for p in range(0, 2):
        for d in range(0, 2):
            for q in range(0, 2):
                for P in range(0, 2):
                    for D in range(0, 2):
                        for Q in range(0, 2):
                            try:
                                model = SARIMAX(train_y, order=(p,d,q), seasonal_order=(P,D,Q,1), enforce_stationarity=False, enforce_invertibility=False)
                                res = model.fit(disp=False)
                                pred_log = res.forecast(steps=len(test_y))
                                rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                                if rmse < best["rmse"]:
                                    best.update({"order": (p,d,q), "seasonal": (P,D,Q,1), "rmse": rmse, "fit": res})
                            except Exception:
                                continue
    return best

def _fit_eval_sarimax(train_y, test_y, train_x, test_x):
    if train_x is None or test_x is None:
        return {"name":"SARIMAX", "order": None, "seasonal": None, "rmse": np.inf, "fit": None}
    best = {"name":"SARIMAX", "order": None, "seasonal": None, "rmse": np.inf, "fit": None}
    for p in range(0, 2):
        for d in range(0, 2):
            for q in range(0, 2):
                for P in range(0, 2):
                    for D in range(0, 2):
                        for Q in range(0, 2):
                            try:
                                model = SARIMAX(train_y, exog=train_x, order=(p,d,q), seasonal_order=(P,D,Q,1),
                                                enforce_stationarity=False, enforce_invertibility=False)
                                res = model.fit(disp=False)
                                pred_log = res.forecast(steps=len(test_y), exog=test_x)
                                rmse = _rmse(np.exp(test_y.values), np.exp(pred_log.values))
                                if rmse < best["rmse"]:
                                    best.update({"order": (p,d,q), "seasonal": (P,D,Q,1), "rmse": rmse, "fit": res})
                            except Exception:
                                continue
    return best

def _refit_and_forecast_full(best_model: dict, full_endog_log: pd.Series, exog_full: pd.DataFrame, future_index: pd.Index, future_exog: pd.DataFrame):
    name = best_model["name"]
    if name in ("ARIMAX","SARIMAX"):
        exog_full_std = exog_full.loc[full_endog_log.index] if exog_full is not None else None
    else:
        exog_full_std = None

    if name == "ARIMA":
        final = ARIMA(full_endog_log, order=best_model["order"]).fit()
        fitted_log = pd.Series(final.fittedvalues, index=full_endog_log.index)
        future_log = final.forecast(steps=len(future_index))
    elif name == "ARIMAX":
        final = ARIMA(full_endog_log, exog=exog_full_std, order=best_model["order"]).fit()
        fitted_log = pd.Series(final.fittedvalues, index=full_endog_log.index)
        future_log = final.forecast(steps=len(future_index), exog=future_exog)
    elif name == "SARIMA":
        final = SARIMAX(full_endog_log, order=best_model["order"], seasonal_order=best_model["seasonal"],
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fitted_log = pd.Series(final.fittedvalues, index=full_endog_log.index)
        future_log = final.forecast(steps=len(future_index))
    else:  # SARIMAX
        final = SARIMAX(full_endog_log, exog=exog_full_std, order=best_model["order"], seasonal_order=best_model["seasonal"],
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fitted_log = pd.Series(final.fittedvalues, index=full_endog_log.index)
        future_log = final.forecast(steps=len(future_index), exog=future_exog)

    return np.exp(fitted_log), np.exp(future_log)

def _plot_result(country: str, actual: pd.Series, fitted: pd.Series,
                 test_idx: pd.Index, test_pred: np.ndarray,
                 future_idx: pd.Index, future_pred: np.ndarray,
                 best_name: str, rmse: float, order=None, seasonal=None):

    fig = go.Figure()
    # Actual
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                             mode="lines", name="Actual CAPEX", line=dict(color="blue")))
    # Fitted
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values,
                             mode="lines", name="Fitted", line=dict(color="green")))
    # Test prediction
    if len(test_idx) > 0:
        fig.add_trace(go.Scatter(x=test_idx, y=test_pred,
                                 mode="lines", name="Test Set Prediction",
                                 line=dict(color="red", dash="dash")))
    # Forecast
    if len(future_idx) > 0:
        fig.add_trace(go.Scatter(x=future_idx, y=future_pred,
                                 mode="lines", name="Future Forecast (3 yrs)",
                                 line=dict(color="orange", dash="dash")))

    extra = f" | order {order}" if order is not None else ""
    if seasonal is not None:
        extra += f", seasonal {seasonal}"
    fig.update_layout(
        title=f"{best_name} Forecast for {country}{extra} | RMSE: {rmse:.2f}",
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10), height=500, showlegend=True
    )
    return fig

# --- public entrypoint -------------------------------------------------------

def render_forecasting_tab():
    st.caption("Forecasting (ARIMA / ARIMAX / SARIMA / SARIMAX) • selects the best model by lowest RMSE")

    panel = _load_combined_panel()
    if panel.empty:
        st.info("No forecasting data available.")
        return

    countries = sorted(panel["country"].dropna().unique().tolist())
    c1, _ = st.columns([1.4, 2], gap="small")
    with c1:
        sel_country = st.selectbox("Country", countries, index=0, key="forecast_country")

    # Prep data
    try:
        prep = _prep_country(panel, sel_country)
    except Exception as e:
        st.error(f"Could not prepare data: {e}")
        return

    train_y, test_y = prep["train_endog"], prep["test_endog"]
    train_x, test_x = prep["train_exog"], prep["test_exog"]

    # Fit/evaluate 4 families
    cand = []
    cand.append(_fit_eval_arima(train_y, test_y))
    cand.append(_fit_eval_arimax(train_y, test_y, train_x, test_x))
    cand.append(_fit_eval_sarima(train_y, test_y))
    cand.append(_fit_eval_sarimax(train_y, test_y, train_x, test_x))

    # Choose best by RMSE
    best = min(cand, key=lambda d: d["rmse"])
    best_name = best["name"]

    # Test predictions from best fit (already computed during search)
    if best_name in ("ARIMA","SARIMA"):
        test_pred_log = best["fit"].forecast(steps=len(test_y))
    else:
        test_pred_log = best["fit"].forecast(steps=len(test_y), exog=test_x)
    test_pred = np.exp(test_pred_log.values)

    # Refit on full data & 3-year forecast
    fitted, future_pred = _refit_and_forecast_full(
        best, prep["full_endog_log"], prep["train_exog"] if prep["train_exog"] is not None else None,
        prep["future_index"], prep["future_exog"]
    )

    fig = _plot_result(
        sel_country,
        actual=prep["capex_actual"],
        fitted=fitted,
        test_idx=test_y.index,
        test_pred=test_pred,
        future_idx=prep["future_index"],
        future_pred=future_pred,
        best_name=best_name,
        rmse=best["rmse"],
        order=best.get("order"),
        seasonal=best.get("seasonal")
    )
    st.plotly_chart(fig, use_container_width=True)

    # small summary
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Best model:** `{best_name}`")
        if best.get("order") is not None:
            st.markdown(f"**Order:** `{best['order']}`")
        if best.get("seasonal") is not None:
            st.markdown(f"**Seasonal:** `{best['seasonal']}`")
    with right:
        st.markdown(f"**RMSE (test):** `{best['rmse']:.2f}`")
        st.markdown(f"**Forecast horizon:** `{int(prep['future_index'][0])}–{int(prep['future_index'][-1])}`")
