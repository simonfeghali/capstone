# forecasting.py
# ─────────────────────────────────────────────────────────────────────────────
# New "Forecasting" tab: ARIMA model selection by lowest RMSE and 3-year forecast
# Uses Plotly (no matplotlib / sklearn).
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

RAW_BASE = "https://raw.githubusercontent.com/simonfeghali/capstone/main"
FILES = {
    # preferred (longer history if present)
    "capex_long": "capex2003-2025.csv",
    # fallbacks you already use
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
    Return DataFrame with columns: country, year, capex
    Tries capex2003-2025.csv first; falls back to your existing files.
    """
    def _melt(df):
        # find a country column
        cands = ["Source Country", "Country", "Source Co", "source_country", "country"]
        country_col = None
        for c in df.columns:
            if str(c).strip() in cands:
                country_col = c
                break
        if country_col is None:
            for c in df.columns:
                if "country" in str(c).lower():
                    country_col = c
                    break
        if not country_col:
            raise ValueError("Could not find a country column in CAPEX file.")

        year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        if not year_cols:
            raise ValueError("No 4-digit year columns detected in CAPEX file.")

        m = df.melt(id_vars=[country_col], value_vars=year_cols,
                    var_name="year", value_name="capex").rename(columns={country_col: "country"})
        m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
        m["capex"] = m["capex"].map(_numify)
        m["country"] = m["country"].astype(str).str.strip()
        m = m.dropna(subset=["year"]).copy()
        return m[["country", "year", "capex"]]

    # 1) Preferred
    try:
        df = pd.read_csv(_raw(FILES["capex_long"]))
        return _melt(df)
    except Exception:
        pass

    # 2) Fallback CSV
    try:
        df = pd.read_csv(_raw(FILES["cap_csv"]))
        return _melt(df)
    except Exception:
        pass

    # 3) Fallback Excel
    try:
        df = pd.read_excel(_raw(FILES["cap_xlsx"]), sheet_name=0)
        return _melt(df)
    except Exception as e:
        raise RuntimeError(f"Could not load any CAPEX source: {e}")

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

def _best_arima_forecast(series: pd.Series, test_h: int = 3):
    """
    Split series into train/test (last test_h years), try several ARIMA orders,
    choose the one with lowest RMSE on the test. Then refit on full data and
    forecast +3 years.
    """
    s = series.copy().astype(float).dropna().sort_index()
    if s.size < 6:
        raise ValueError("Not enough data points after cleaning to fit ARIMA (need at least 6).")

    # adaptive test horizon: at least 2 points, ~15% of series length
    test_h = min(test_h, max(2, int(np.ceil(0.15 * len(s)))))
    train = s.iloc[:-test_h]
    test  = s.iloc[-test_h:]

    candidates = [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]
    best = None

    for order in candidates:
        try:
            model = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit()
            fitted = pd.Series(res.fittedvalues, index=train.index)
            test_pred = res.forecast(steps=test_h)
            score = _rmse(test.values, test_pred.values)
            info = {
                "order": order,
                "rmse": score,
                "fitted_index": fitted.index, "fitted_values": fitted.values,
                "test_index": test.index, "test_pred": test_pred.values,
            }
            if (best is None) or (score < best["rmse"]):
                best = info
        except Exception:
            continue

    if best is None:
        raise ValueError("All ARIMA candidates failed to fit.")

    # Refit on full series and forecast 3 future years
    try:
        final = ARIMA(s, order=best["order"], enforce_stationarity=False, enforce_invertibility=False).fit()
        future_steps = 3
        last_year = int(s.index.max())
        future_index = pd.Index([last_year + i for i in range(1, future_steps + 1)], name="year")
        future_pred = final.forecast(steps=future_steps).values
    except Exception:
        future_index = pd.Index([], name="year")
        future_pred = np.array([])

    best.update({
        "actual_index": s.index, "actual_values": s.values,
        "future_index": future_index, "future_pred": future_pred
    })
    return best

def _plot_best(country: str, result: dict):
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=result["actual_index"], y=result["actual_values"],
        mode="lines", name="Actual CAPEX", line=dict(color="blue")
    ))
    # Fitted (train)
    fig.add_trace(go.Scatte r(
        x=result["fitted_index"], y=result["fitted_values"],
        mode="lines", name="Fitted", line=dict(color="green")
    ))
    # Test predictions
    if len(result["test_index"]) > 0:
        fig.add_trace(go.Scatter(
            x=result["test_index"], y=result["test_pred"],
            mode="lines", name="Test Set Prediction",
            line=dict(color="red", dash="dash")
        ))
    # Future forecast
    if len(result.get("future_index", [])) > 0:
        fig.add_trace(go.Scatter(
            x=result["future_index"], y=result["future_pred"],
            mode="lines", name="Future Forecast (3 yrs)",
            line=dict(color="orange", dash="dash")
        ))

    fig.update_layout(
        title=f"ARIMA Forecast for {country} | Best order {result['order']} | RMSE: {result['rmse']:.2f}",
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10), height=480,
        showlegend=True
    )
    return fig

def render_forecasting_tab():
    st.caption("Forecasting (ARIMA) • selects the best model by lowest RMSE on a test split")

    capx = _load_capex_wide()  # columns: country, year, capex
    countries = sorted(capx["country"].dropna().unique().tolist())
    if not countries:
        st.info("No CAPEX data available.")
        return

    c1, _ = st.columns([1.2, 2], gap="small")
    with c1:
        sel_country = st.selectbox("Country", countries, index=0, key="forecast_country")

    # Build yearly series for selected country
    s = (capx.loc[capx["country"] == sel_country, ["year", "capex"]]
              .dropna(subset=["year"])
              .groupby("year", as_index=True)["capex"].sum()
              .sort_index())

    if s.empty or s.dropna().shape[0] < 6:
        st.info("Not enough data points to fit ARIMA for this country (need ≥ 6).")
        return

    try:
        result = _best_arima_forecast(s)
    except Exception as e:
        st.error(f"Could not fit ARIMA models: {e}")
        return

    fig = _plot_best(sel_country, result)
    st.plotly_chart(fig, use_container_width=True)

    # small summary
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Best ARIMA order:** `{result['order']}`")
        st.markdown(f"**RMSE (test):** `{result['rmse']:.2f}`")
    with right:
        if len(result["fitted_index"]) > 0 and len(result["test_index"]) > 0:
            st.markdown(f"**Train:** {int(result['fitted_index'][0])}–{int(result['fitted_index'][-1])}  •  "
                        f"**Test:** {int(result['test_index'][0])}–{int(result['test_index'][-1])}")
        if len(result.get("future_index", [])) > 0:
            st.markdown(f"**Forecast horizon:** {int(result['future_index'][0])}–{int(result['future_index'][-1])}")
