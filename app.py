# app.py
# Momentum-RoboAdvisor v1.3.3
# - Analyse: Momentum-Score (260/130), RS z, Volumen-Score, 52W-Drawdown, Volatilität
# - Handlungsempfehlungen: Kaufen/Halten/Verkaufen + Marktübersicht mit GD200-Anteil und Rundungsregel
# - Backtest: monatliches Rebalancing mit Exposure (optional)
# - Filter: Volumen, Drawdown, Volatilität, RS > Benchmark

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Momentum-RoboAdvisor", layout="wide")

# ============================================================================
# Hilfsfunktionen
# ============================================================================

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """Lädt Kurs- und Volumendaten."""
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [str(t).strip() for t in ticker_list if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    try:
        data = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        st.error(f"Fehler beim Download: {e}")
        return pd.DataFrame(), pd.DataFrame()

    close_dict, vol_dict = {}, {}
    for t in tickers:
        try:
            df = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        except Exception:
            df = data.copy()
        if df.empty:
            continue
        closes = (df["Adj Close"] if "Adj Close" in df.columns else df.get("Close"))
        vols = df.get("Volume")
        if closes is None or closes.dropna().empty:
            continue
        close_dict[t] = closes.rename(t)
        vol_dict[t] = (vols.rename(t) if vols is not None else pd.Series(dtype=float, name=t))

    price = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    volume = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()
    price = price.sort_index().dropna(how="all")
    volume = volume.reindex(price.index)
    return price, volume


def pct_change_over_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    start_val = s.iloc[-(days+1)]
    end_val = s.iloc[-1]
    if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0:
        return np.nan
    return (end_val / start_val - 1.0) * 100.0


def safe_sma(series: pd.Series, window: int) -> pd.Series:
    if series is None or series.empty:
        return series
    return series.rolling(window=window, min_periods=max(5, window // 5)).mean()


def zscore_last(value: float, mean: float, std: float) -> float:
    if std is None or std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std


def volume_score(vol_series: pd.Series, lookback=60):
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base is None or base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    return float(np.clip(cur / base, 0.5, 2.0))


def logp(x):
    if pd.isna(x):
        return np.nan
    return np.sign(x) * np.log1p(abs(x))

# ============================================================================
# Kennzahlen / Momentum-Scores
# ============================================================================

def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame, benchmark_df=None):
    results = []

    mom130_universe = {t: pct_change_over_window(price_df[t], 130) for t in price_df.columns}
    mom130_series = pd.Series(mom130_universe).astype(float)
    mu, sigma = mom130_series.mean(), mom130_series.std(ddof=0)

    bm_return = None
    if benchmark_df is not None and not benchmark_df.empty:
        bm_return = pct_change_over_window(benchmark_df.iloc[:, 0], 130)

    for t in price_df.columns:
        s = price_df[t].dropna()
        v = volume_df[t].dropna() if (isinstance(volume_df, pd.DataFrame) and t in volume_df) else pd.Series(dtype=float)
        if s.empty or len(s) < 200:
            continue

        last = s.iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        mom130 = pct_change_over_window(s, 130)
        rs_z = zscore_last(mom130, mu, sigma)
        vol_sc = volume_score(v, 60)
        avg_vol = v.rolling(60).mean().iloc[-1] if not v.empty else np.nan

        d50 = (last / sma50 - 1.0) * 100.0 if sma50 else np.nan
        d200 = (last / sma200 - 1.0) * 100.0 if sma200 else np.nan
        sig50 = "Über GD50" if last >= sma50 else "Unter GD50"
        sig200 = "Über GD200" if last >= sma200 else "Unter GD200"

        high52 = s[-260:].max()
        dd52 = (last / high52 - 1.0) * 100.0 if high52 else np.nan
        vol = s.pct_change().std() * np.sqrt(252)

        score = (
            0.40 * logp(mom260)
            + 0.30 * logp(mom130)
            + 0.20 * rs_z
            + 0.10 * (vol_sc - 1.0 if not np.isnan(vol_sc) else 0)
        )
        score = 0.0 if np.isnan(score) else float(score)

        results.append({
            "Ticker": t,
            "Kurs aktuell": round(last, 2),
            "MOM260 (%)": round(mom260, 2),
            "MOM130 (%)": round(mom130, 2),
            "RS z-Score": round(rs_z, 2),
            "Volumen-Score": round(vol_sc, 2),
            "Ø Volumen (60T)": round(avg_vol, 0),
            "Abstand GD50 (%)": round(d50, 2),
            "Abstand GD200 (%)": round(d200, 2),
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "52W-Drawdown (%)": round(dd52, 2),
            "Volatilität (ann.)": round(vol, 2),
            "Momentum-Score": round(score, 3),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
start_date = st.sidebar.date_input("Startdatum (Datenabruf)", value=datetime.today() - timedelta(days=900))
end_date = st.sidebar.date_input("Enddatum", value=datetime.today())
min_volume = st.sidebar.number_input("Min. Ø Volumen (60T)", min_value=0, value=5_000_000, step=100_000)
benchmark_ticker = st.sidebar.text_input("Benchmark-Ticker", "SPY")

# ============================================================================
# Daten laden
# ============================================================================

st.title("📈 Momentum-Analyse mit GD200-Marktübersicht")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name**", type=["csv"])
tickers_txt = st.text_input("Oder Ticker (kommagetrennt):", "AAPL, MSFT, NVDA, META, LLY, JPM")
portfolio_txt = st.text_input("(Optional) Aktuelle Portfolio-Ticker:", "")

name_map = {}
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" in df_in.columns:
            if "Name" in df_in.columns:
                name_map = dict(zip(df_in["Ticker"], df_in["Name"]))
            tickers_txt = ", ".join(df_in["Ticker"].astype(str).tolist())
            st.success(f"{len(df_in)} Ticker aus CSV geladen.")
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")

tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
portfolio = set([t.strip().upper() for t in portfolio_txt.split(",") if t.strip()])

if not tickers:
    st.info("Bitte Ticker eingeben oder CSV laden.")
    st.stop()

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)
    bm_prices, _ = fetch_ohlc([benchmark_ticker], start_date, end_date)

if prices.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

# ============================================================================
# Analyse
# ============================================================================

df = compute_indicators(prices, volumes, benchmark_df=bm_prices)
if df.empty:
    st.warning("Keine Kennzahlen berechnet.")
    st.stop()

df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])

# ============================================================================
# Tabs
# ============================================================================

tab1, tab2 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen"])

with tab1:
    st.subheader("Analyse – Kennzahlen")
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen")

    # --- Marktübersicht ---
    total = len(df)
    gd200_above = int((df["GD200-Signal"] == "Über GD200").sum())
    share = gd200_above / total if total > 0 else 0

    invest_raw = share * top_n
    invest_n = int(math.floor(invest_raw + 0.5))
    invest_n = max(0, min(invest_n, top_n))

    st.markdown(
        f"**Marktübersicht:** {share:.0%} der analysierten Aktien "
        f"({gd200_above}/{total}) liegen **über dem GD200**."
    )
    st.markdown(
        f"**Investiere:** **{invest_n}** von **Top-{top_n}** "
        f"(normales Runden: {share:.2f}×{top_n} = {invest_raw:.2f} → {invest_n})."
    )

    st.divider()

    # Handlungsempfehlungen
    def rec_row(row, in_port, top_limit=10):
        t = row["Ticker"]
        rank = row["Rank"]
        over50 = row["GD50-Signal"].startswith("Über")
        over200 = row["GD200-Signal"].startswith("Über")

        if t in in_port:
            if not over50:
                return "🔴 Verkaufen (unter GD50)"
            if rank <= top_limit:
                return "🟡 Halten"
            return "🔴 Verkaufen (nicht mehr Top)"
        else:
            if rank <= top_limit and over50 and over200:
                return "🟢 Kaufen"
            return "—"

    rec_df = df.copy()
    rec_df["Handlung"] = rec_df.apply(lambda r: rec_row(r, portfolio, top_limit=max(invest_n, 0)), axis=1)

    cols = ["Rank", "Ticker", "Name", "Momentum-Score", "GD50-Signal", "GD200-Signal", "Handlung"]
    st.dataframe(rec_df[cols], use_container_width=True)

st.caption("Nur Informations- und Ausbildungszwecke. Keine Anlageempfehlung.")
