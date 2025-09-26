# app.py
# Momentum-Screener mit Handlungsempfehlungen (Kaufen/Halten/Verkaufen)
# Robuster OHLCV-Downloader mit Batch-Handling
# Universum kann sehr groß sein (z. B. 750 Aktien)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Momentum-Screener", page_icon="📈", layout="wide")

# ---------------------------------------------------
# ROBUSTER OHLCV-DOWNLOADER
# ---------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end, batch_size: int = 150):
    """
    Lädt OHLCV-Daten robust in Batches.
    Gibt (Preise, Volumen) zurück.
    """
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [str(t).strip() for t in ticker_list if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    close_dict, vol_dict = {}, {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                tickers=" ".join(batch),
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            st.warning(f"Fehler beim Download Batch {i}: {e}")
            continue

        for t in batch:
            try:
                df = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
            except Exception:
                df = data.copy()

            if df is None or df.empty:
                continue

            closes = (df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")).rename(t)
            vols = df.get("Volume", pd.Series(dtype=float)).rename(t)

            close_dict[t] = closes
            vol_dict[t] = vols

    price = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    volume = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()
    price = price.sort_index().dropna(how="all")
    volume = volume.reindex_like(price)
    return price, volume

# ---------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------
def pct_change_over_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    start_val, end_val = s.iloc[-(days+1)], s.iloc[-1]
    if start_val <= 0 or pd.isna(start_val) or pd.isna(end_val):
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
    """Volume-Multiplikator: aktuelles Vol / SMA(lookback). Caps (0.5 – 2.0)."""
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base is None or base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    ratio = cur / base
    return float(np.clip(ratio, 0.5, 2.0))

# ---------------------------------------------------
# Indikatoren & Score
# ---------------------------------------------------
def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    results = []

    mom90_universe = {t: pct_change_over_window(price_df[t], 90) for t in price_df.columns}
    mom90_series = pd.Series(mom90_universe).astype(float)
    mu, sigma = mom90_series.mean(), mom90_series.std(ddof=0)

    for t in price_df.columns:
        s = price_df[t].dropna()
        if s.empty or len(s) < 60:
            continue

        last = s.iloc[-1]
        sma50, sma200 = safe_sma(s, 50).iloc[-1], safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        momJT  = pct_change_over_window(s, 90)

        rs_90, rs_z = mom90_series.get(t, np.nan), np.nan
        if not np.isnan(rs_90):
            rs_z = zscore_last(rs_90, mu, sigma)

        vol_sc = volume_score(volume_df.get(t, pd.Series(dtype=float)), lookback=60)

        def dist(p, m): return (p / m - 1.0) * 100.0 if (not pd.isna(p) and not pd.isna(m) and m != 0) else np.nan
        d50, d200 = dist(last, sma50), dist(last, sma200)

        sig50  = "Über GD50" if (not pd.isna(last) and not pd.isna(sma50) and last >= sma50) else "Unter GD50"
        sig200 = "Über GD200" if (not pd.isna(last) and not pd.isna(sma200) and last >= sma200) else "Unter GD200"

        def logp(x): return np.sign(x) * np.log1p(abs(x)) if not pd.isna(x) else np.nan

        mom_part = 0.40 * logp(mom260) + 0.30 * logp(momJT)
        rs_part  = 0.20 * (0 if pd.isna(rs_z) else rs_z)
        vol_part = 0.10 * (0 if pd.isna(vol_sc) else (vol_sc - 1.0))

        score = mom_part + rs_part + vol_part
        score = 0.0 if pd.isna(score) else float(score)

        results.append({
            "Ticker": t,
            "Kurs aktuell": last,
            "MOM260 (%)": mom260,
            "MOMJT (%)":  momJT,
            "Relative Stärke (%)": rs_90,
            "RS z-Score": rs_z,
            "Volumen-Score": vol_sc,
            "Abstand GD50 (%)": d50,
            "Abstand GD200 (%)": d200,
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "Momentum-Score": score,
        })

    df = pd.DataFrame(results)
    if df.empty: return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

# ---------------------------------------------------
# Handlungsempfehlungen
# ---------------------------------------------------
def rec_row(row, in_port, top_n=10, reserve=2):
    t, rank = row["Ticker"], row["Rank"]
    over50, over200 = row["GD50-Signal"].startswith("Über"), row["GD200-Signal"].startswith("Über")

    if t in in_port:
        if not over50: return "🔴 Verkaufen (unter GD50)"
        if rank <= top_n: return "🟡 Halten"
        if rank <= top_n + reserve and over200: return "🟡 Halten (Reserve)"
        return "🔴 Verkaufen (nicht mehr Top)"
    else:
        if rank <= top_n and over50 and over200: return "🟢 Kaufen"
        if rank <= top_n + reserve and over50 and over200: return "🟡 Beobachten (Reserve)"
        return "—"

def dot(color: str) -> str:
    return f"<span style='font-size:18px;color:{color}'>●</span>"

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", 3, 50, 10, 1)
reserve_m = st.sidebar.number_input("Reserven (Nachrücker)", 0, 20, 2, 1)
start_date = st.sidebar.date_input("Startdatum", value=datetime.today() - timedelta(days=900))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())

st.title("📊 Momentum-Analyse mit Handlungsempfehlungen (GD50/GD200)")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name** hochladen", type=["csv"])
tickers_txt = st.text_input("Oder Ticker eingeben (Yahoo Finance, komma-getrennt):", "AAPL, MSFT, TSLA, NVDA")
portfolio_txt = st.text_input("(Optional) Portfolio-Ticker:", "AAPL")

name_map = {}
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" not in df_in.columns:
            st.error("CSV muss Spalte **Ticker** enthalten.")
        else:
            if "Name" in df_in.columns:
                name_map = dict(zip(df_in["Ticker"].astype(str), df_in["Name"].astype(str)))
            tickers_txt = ", ".join(df_in["Ticker"].astype(str).tolist())
            st.success(f"{len(df_in)} Ticker aus CSV geladen.")
    except Exception as e:
        st.error(f"Fehler beim Einlesen: {e}")

tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
in_port = set([t.strip().upper() for t in portfolio_txt.split(",") if t.strip()])

if not tickers:
    st.info("Bitte Ticker eingeben oder CSV laden.")
    st.stop()

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)

if prices.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

df = compute_indicators(prices, volumes)
if df.empty:
    st.warning("Kennzahlen konnten nicht berechnet werden.")
    st.stop()

df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])
df["_GD50_dot"]  = df["GD50-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))
df["_GD200_dot"] = df["GD200-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen"])

with tab1:
    st.subheader("Analyse – alle Kennzahlen")
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen – Kaufen / Halten / Verkaufen")

    rec_df = df.copy()
    rec_df["Handlung"] = rec_df.apply(lambda r: rec_row(r, in_port, top_n=top_n, reserve=reserve_m), axis=1)
    rec_df = rec_df.sort_values("Rank").reset_index(drop=True)
    rec_df["GD50"]  = rec_df["_GD50_dot"]
    rec_df["GD200"] = rec_df["_GD200_dot"]

    buy_df  = rec_df[rec_df["Handlung"].str.startswith("🟢")]
    hold_df = rec_df[rec_df["Handlung"].str.startswith("🟡")]
    sell_df = rec_df[rec_df["Handlung"].str.startswith("🔴")]

    def render_block(title, frame):
        st.markdown(f"### {title}")
        cols = ["Rank", "Ticker", "Name", "Momentum-Score", "GD50", "GD200", "Handlung"]
        if frame.empty:
            st.write("—")
        else:
            st.write(frame[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

    render_block("🟢 Kaufen", buy_df)
    render_block("🟡 Halten / Beobachten", hold_df)
    render_block("🔴 Verkaufen", sell_df)

st.caption("Hinweis: Nur zu Informations- und Ausbildungszwecken.")
