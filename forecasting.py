# forecasting.py
# -----------------------------------------------------------------------------
# New "Forecasting" tab: ARIMA model selection by lowest RMSE and 3-year forecast
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    # preferred (longer history, if present in your repo)
    "capex_long": "capex2003-2025.csv",
    # fallbacks (already used in the main app)
    "cap_csv": "capex_EDA_cleaned_filled.csv",
    "cap_xlsx": "capex_EDA.xlsx",
}

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

@st.cache_data(show_spinner=True)
def _load_capex_wide() -> pd.DataFrame:
    """
    Load a 'wide' CAPEX table with columns:
      - Country (or Source Country)
      - 2003, 2004, ..., 2025  (year columns)
    Tries capex2003-2025.csv first, then falls back to the existing files.
    Returns a DataFrame with standardized columns:
      - country (str)
      - year (int)
      - capex (float)
    """
    # helper to melt wide → long
    def _melt(df):
        # find country column
        cands = ["Source Country", "Country", "Source Co", "source_country", "country"]
        country_col = None
        for c in df.columns:
            if str(c).strip() in cands:
                country_col = c
                break
        if country_col is None:
            # try fuzzy match
            for c in df.columns:
                if "country" in str(c).lower():
                    country_col = c
                    break
        if not country_col:
            raise ValueError("Could not find a country column in CAPEX file.")

        # year columns = 4-digit
        year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        if not year_cols:
            raise ValueError("No 4-digit year columns detected in CAPEX file.")

        m = df.melt(id_vars=[country_col], value_vars=year_cols,
                    var_name="year", value_name="capex")
        m = m.rename(columns={country_col: "country"})
        m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
        m["capex"] = m["capex"].map(_numify)
        m["country"] = m["country"].astype(str).str.strip()
        m = m.dropna(subset=["year"]).copy()
        return m[["country", "year", "capex"]]

    # 1) Preferred: capex2003-2025.csv
    try:
        df = pd.read_csv(_raw(FILES["capex_long"]))
        return _melt(df)
    except Exception:
        pass

    # 2) Fallback: capex_EDA_cleaned_filled.csv
    try:
        df = pd.read_csv(_raw(FILES["cap_csv"]))
        return _melt(df)
    except Exception:
        pass

    # 3) Fallback: capex_EDA.xlsx
    try:
        df = pd.read_excel(_raw(FILES["cap_xlsx"]), sheet_name=0)
        return _melt(df)
    except Exception as e:
        raise RuntimeError(f"Could not load any CAPEX source: {e}")

def _best_arima_forecast(series: pd.Series, years: pd.Index, test_h: int = 3):
    """
    Given a numeric series (indexed by year), split into train/test (last test_h years for test),
    try several ARIMA orders, choose the lowest RMSE on the test set.
    Returns dict with:
      - order, rmse, fitted_index, fitted_values
      - test_index, test_pred
      - future_index, future_pred
    """
    # ensure no NaNs at tail (trim) and interpolate inside
    s = series.copy().astype(float)
    # Drop leading/trailing NaNs, interpolate internal gaps
    s = s.dropna()
    s = s.sort_index()
    if s.size < 6:
        raise ValueError("Not enough data points after cleaning to fit ARIMA (need at least 6).")

    # test horizon
    test_h = min(test_h, max(2, int(np.ceil(0.15 * len(s)))))  # adaptive, but at least 2
    train = s.iloc[:-test_h]
    test  = s.iloc[-test_h:]

    # candidate orders (4 models)
    candidates = [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]

    best = None
    for order in candidates:
        try:
            model = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit()
            # in-sample fitted (for plotting)
            fitted = pd.Series(res.fittedvalues, index=train.index)
            # predict test horizon
            test_pred = res.forecast(steps=test_h)
            rmse = sqrt(mean_squared_error(test.values, test_pred.values))
            info = {
                "order": order,
                "rmse": rmse,
                "fitted_index": fitted.index, "fitted_values": fitted.values,
                "test_index": test.index, "test_pred": test_pred.values,
            }
            if (best is None) or (rmse < best["rmse"]):
                best = info
        except Exception:
            # skip failures
            continue

    if best is None:
        raise ValueError("All ARIMA candidates failed to fit.")

    # Refit best order on full data and forecast next 3 years
    try:
        final = ARIMA(s, order=best["order"], enforce_stationarity=False, enforce_invertibility=False).fit()
        future_steps = 3
        last_year = int(s.index.max())
        future_index = pd.Index([last_year + i for i in range(1, future_steps + 1)], name="year")
        future_pred = final.forecast(steps=future_steps)
        best.update({
            "future_index": future_index,
            "future_pred": future_pred.values
        })
    except Exception:
        # If refit fails, keep what we have and no future forecast
        best.update({"future_index": pd.Index([], name="year"), "future_pred": np.array([])})

    # Also store actual slices for convenience
    best.update({
        "actual_index": s.index, "actual_values": s.values
    })
    return best

def _plot_best(country: str, result: dict):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Actual series
    ax.plot(result["actual_index"], result["actual_values"], label="Actual CAPEX", color="blue")

    # In-sample fitted on train
    ax.plot(result["fitted_index"], result["fitted_values"], label="Fitted", color="green")

    # Test predictions (dashed)
    if len(result["test_index"]) > 0:
        ax.plot(result["test_index"], result["test_pred"], label="Test Set Prediction", color="red", linestyle="--")

    # Future forecast (dashed, orange)
    if len(result.get("future_index", [])) > 0:
        ax.plot(result["future_index"], result["future_pred"], label="Future Forecast (3 yrs)", color="orange", linestyle="--")

    ax.set_title(f"ARIMA Forecast for {country} | Best order {result['order']} | RMSE: {result['rmse']:.2f}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="upper right")
    ax.grid(False)
    fig.tight_layout()
    return fig

def render_forecasting_tab():
    st.caption("Forecasting (ARIMA) • selects the best model by lowest RMSE on a test split")

    # Load and prep data
    capx = _load_capex_wide()  # columns: country, year, capex

    # Countries list
    countries = sorted(capx["country"].dropna().unique().tolist())
    if not countries:
        st.info("No CAPEX data available.")
        return

    # UI
    c1, c2 = st.columns([1.2, 2], gap="small")
    with c1:
        sel_country = st.selectbox("Country", countries, index=0, key="forecast_country")
        # optional: choose test horizon; keep simple/hidden for now

    # Build yearly series for selected country
    s = (capx.loc[capx["country"] == sel_country, ["year", "capex"]]
              .dropna(subset=["year"])
              .groupby("year", as_index=True)["capex"].sum()
              .sort_index())

    if s.empty or s.dropna().shape[0] < 6:
        st.info("Not enough data points to fit ARIMA for this country (need ≥ 6).")
        return

    # Choose best among 4 ARIMA candidates
    try:
        result = _best_arima_forecast(s, s.index)
    except Exception as e:
        st.error(f"Could not fit ARIMA models: {e}")
        return

    # Plot
    fig = _plot_best(sel_country, result)
    st.pyplot(fig)

    # Small metrics block
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Best ARIMA order:** `{result['order']}`")
        st.markdown(f"**RMSE (test):** `{result['rmse']:.2f}`")
    with right:
        tr_start, tr_end = int(result["fitted_index"][0]), int(result["fitted_index"][-1]) if len(result["fitted_index"]) else (None, None)
        if len(result["test_index"]) > 0:
            ts_start, ts_end = int(result["test_index"][0]), int(result["test_index"][-1])
            st.markdown(f"**Train:** {tr_start}–{tr_end}  •  **Test:** {ts_start}–{ts_end}")
        if len(result.get("future_index", [])) > 0:
            fut_str = f"{int(result['future_index'][0])}–{int(result['future_index'][-1])}"
            st.markdown(f"**Forecast horizon:** {fut_str}")
