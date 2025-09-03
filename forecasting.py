# forecasting.py
# ─────────────────────────────────────────────────────────────────────────────
# Make the app's behavior match the notebook:
# - Data prep matches notebook (drop 2003, drop countries with any missing CAPEX)
# - Prefer final_capex_and_indicators_cleaned.csv (or forecasting_... file)
# - Scale exog on the FULL series (pre-split) to replicate notebook results
# - Split: last 4 years
# - Forecast horizon: 3 years
# - Plot: Actual, Fitted, Test Prediction, Future Forecast (single chart)
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

# ── data loading (notebook-aligned) ──────────────────────────────────────────

@st.cache_data(show_spinner=True)
def _load_notebook_style_panel() -> pd.DataFrame:
    """
    Try to load exactly what the notebook trains on, then fall back to older sources.
    Normalize to columns: Country, Year, CAPEX, plus indicators/exog columns.
    Apply notebook cleaning (drop 2003, remove countries with any missing CAPEX).
    """
    # 0) Preferred "final" clean files produced in the notebook
    for key in ("final_clean", "forecasting_final_clean", "combined"):
        try:
            df = pd.read_csv(_raw(FILES[key]))
            ctry = _find_col(df.columns, "Country", "country")
            year = _find_col(df.columns, "Year", "year")
            cap  = _find_col(df.columns, "CAPEX", "capex")
            if not (ctry and year and cap):
                raise ValueError("Missing Country/Year/CAPEX columns.")
            df = df.rename(columns={ctry: "Country", year: "Year", cap: "CAPEX"}).copy()
            # number-ify CAPEX
            df["CAPEX"] = df["CAPEX"].map(_numify)
            break
        except Exception:
            df = None
    if df is None:
        # 1) Fallback: merge capex_long + indicators (as in app)
        # capex wide -> long
        cap = pd.read_csv(_raw(FILES["capex_long"]))
        ctry = _find_col(cap.columns, "Source Country", "Country", "source_country", "Source Co")
        if not ctry:
            for c in cap.columns:
                if "country" in str(c).lower():
                    ctry = c
                    break
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

    # - Countries with exactly one missing CAPEX and it's 2003 would have been removed
    #   (after the 2003 drop, just remove any country that still has a missing CAPEX)
    miss = df[df["CAPEX"].isna()]
    if not miss.empty:
        bad_countries = miss["Country"].unique().tolist()
        df = df[~df["Country"].isin(bad_countries)].copy()

    # Coerce year to int
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Final tidy
    df["Country"] = df["Country"].astype(str).str.strip()
    return df

# ── modeling (notebook grids & choices) ──────────────────────────────────────

def _prep_country_notebook(df_all: pd.DataFrame, country: str):
    """
    Matches notebook prep:
      - endog = log(CAPEX) with CAPEX>0
      - exog = StandardScaler fit on FULL series (pre-split)
      - split = last 4 years for test
      - future = 3 years repeating last exog row (scaled with same scaler)
    """
    d = df_all[df_all["Country"] == country].copy().sort_values("Year")
    d["CAPEX"] = d["CAPEX"].map(_numify)
    d = d.dropna(subset=["CAPEX"])
    d = d[d["CAPEX"] > 0]

    if d.shape[0] < 8:
        raise ValueError("Not enough datapoints after cleaning (need ≥ 8 for a 4-year test).")

    endog_log = np.log(d["CAPEX"]).rename("log_CAPEX")
    exog_cols = []
    for col in EXOG_DEFAULT:
        hit = _find_col(d.columns, col)
        if hit:
            exog_cols.append(hit)
    if exog_cols:
        exog_raw = d[exog_cols].apply(pd.to_numeric, errors="coerce")
        # Notebook did not interpolate; keep as is (drop rows with all-NaN will break index).
        # We’ll forward/back fill to avoid model errors, but keep it conservative.
        exog_raw = exog_raw.copy()
        if exog_raw.isna().any().any():
            exog_raw = exog_raw.interpolate(limit_direction="both")
        scaler = StandardScaler()
        exog_full = pd.DataFrame(
            scaler.fit_transform(exog_raw),  # fit on FULL series (intentional leakage to match notebook)
            columns=exog_raw.columns,
            index=d["Year"].values
        )
    else:
        exog_full = None

    # Split: last 4 years test
    n = len(endog_log)
    split_years = 4
    train_idx = d["Year"].iloc[: n - split_years].values
    test_idx  = d["Year"].iloc[n - split_years :].values

    train_y = pd.Series(endog_log.values[: n - split_years], index=train_idx)
    test_y  = pd.Series(endog_log.values[n - split_years :], index=test_idx)

    if exog_full is not None:
        train_x = exog_full.loc[train_idx]
        test_x  = exog_full.loc[test_idx]
        # Future exog: repeat last observed row (already scaled)
        last_row = exog_full.loc[[exog_full.index.max()]]
    else:
        train_x = test_x = None
        last_row = None

    # Future horizon: 3 years
    last_year = int(d["Year"].max())
    future_index = pd.Index([last_year + i for i in range(1, 3 + 1)], name="Year")
    if last_row is not None:
        future_exog = pd.DataFrame(
            np.repeat(last_row.values, repeats=3, axis=0),
            columns=last_row.columns,
            index=future_index
        )
    else:
        future_exog = None

    return {
        "capex_actual": pd.Series(d["CAPEX"].values, index=d["Year"].values, name="CAPEX"),
        "endog_log": pd.Series(endog_log.values, index=d["Year"].values, name="log_CAPEX"),
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
    for p in range(0, 4):        # notebook had wider ARIMAX grid
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
                                res = SARIMAX(train_y, exog=train_x, order=(p, d, q), seasonal_order=(P, D, Q, 1)).fit(disp=False)
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
    name = best_model["name"]
    if name == "ARIMA":
        final = ARIMA(endog_log, order=best_model["order"]).fit()
        fitted_log = pd.Series(final.fittedvalues, index=endog_log.index)
        future_log = final.forecast(steps=len(future_index))
    elif name == "ARIMAX":
        final = ARIMA(endog_log, exog=exog_full.loc[endog_log.index], order=best_model["order"]).fit()
        fitted_log = pd.Series(final.fittedvalues, index=endog_log.index)
        future_log = final.forecast(steps=len(future_index), exog=future_exog)
    elif name == "SARIMA":
        final = SARIMAX(endog_log, order=best_model["order"], seasonal_order=best_model["seasonal"]).fit(disp=False)
        fitted_log = pd.Series(final.fittedvalues, index=endog_log.index)
        future_log = final.forecast(steps=len(future_index))
    else:  # SARIMAX
        final = SARIMAX(endog_log, exog=exog_full.loc[endog_log.index],
                        order=best_model["order"], seasonal_order=best_model["seasonal"]).fit(disp=False)
        fitted_log = pd.Series(final.fittedvalues, index=endog_log.index)
        future_log = final.forecast(steps=len(future_index), exog=future_exog)

    fitted = pd.Series(np.exp(fitted_log).values, index=fitted_log.index, name="fitted")
    future = pd.Series(np.exp(future_log).values, index=future_log.index, name="forecast")
    return fitted, future

def _plot_like_notebook(country: str,
                        actual: pd.Series,
                        fitted: pd.Series,
                        test_idx: pd.Index,
                        test_pred: np.ndarray,
                        future_idx: pd.Index,
                        future_pred: pd.Series,
                        best_name: str,
                        rmse: float):
    fig = go.Figure()
    # Actual
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                             mode="lines", name="Actual CAPEX"))
    # Fitted
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values,
                             mode="lines", name="Fitted"))
    # Test prediction
    if len(test_idx) > 0:
        fig.add_trace(go.Scatter(x=test_idx, y=test_pred,
                                 mode="lines", name="Test Set Prediction",
                                 line=dict(dash="dash")))
    # Future forecast (3 yrs)
    if len(future_idx) > 0:
        fig.add_trace(go.Scatter(x=future_idx, y=future_pred.values,
                                 mode="lines", name="Future Forecast (3 yrs)",
                                 line=dict(dash="dot")))
    fig.update_layout(
        title=f"{best_name} Forecast | {country} | RMSE: {rmse:.2f}",
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10),
        height=520, showlegend=True
    )
    return fig

# ── public entrypoint ────────────────────────────────────────────────────────

def render_forecasting_tab():
    st.caption("Forecasting (Notebook-aligned): ARIMA / ARIMAX / SARIMA / SARIMAX • last-4 test • 3-yr horizon")

    panel = _load_notebook_style_panel()
    if panel.empty:
        st.info("No forecasting data available.")
        return

    countries = sorted(panel["Country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", countries, index=0, key="forecast_country_notebook")

    # Prep data (notebook style)
    try:
        prep = _prep_country_notebook(panel, sel_country)
    except Exception as e:
        st.error(f"Could not prepare data: {e}")
        return

    train_y, test_y = prep["train_y"], prep["test_y"]
    train_x, test_x = prep["train_x"], prep["test_x"]

    # Fit/evaluate 4 families (notebook grids)
    cand = []
    cand.append(_fit_eval_arima(train_y, test_y))
    cand.append(_fit_eval_arimax(train_y, test_y, train_x, test_x))
    cand.append(_fit_eval_sarima(train_y, test_y))
    cand.append(_fit_eval_sarimax(train_y, test_y, train_x, test_x))

    best = min(cand, key=lambda d: d["rmse"])
    best_name = best["name"]

    # Test predictions from best fit
    if best_name in ("ARIMA", "SARIMA"):
        test_pred_log = best["fit"].forecast(steps=len(test_y))
    else:
        test_pred_log = best["fit"].forecast(steps=len(test_y), exog=test_x)
    test_pred = np.exp(test_pred_log.values)

    # Refit on full & 3-yr forecast
    fitted, future_pred = _refit_and_forecast_full(
        best, prep["endog_log"], prep["exog_full"], prep["future_index"], prep["future_exog"]
    )

    # Plot like the notebook (single figure with all layers)
    fig = _plot_like_notebook(
        sel_country, prep["capex_actual"], fitted,
        test_y.index, test_pred,
        prep["future_index"], future_pred,
        best_name, best["rmse"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Small summary (matches notebook information density)
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
