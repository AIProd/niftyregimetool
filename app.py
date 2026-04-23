
import io
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Nifty Regime Tool", layout="wide")

INDEX_CONFIG = {
    "Nifty 50": {
        "price_ticker": "^NSEI",
        "default_valuation_file": "data/nifty50_valuation.csv",
        "description": "Core large-cap India index.",
    },
    "Nifty Next 50": {
        "price_ticker": "^NSMIDCP",
        "default_valuation_file": "data/nifty_next50_valuation.csv",
        "description": "Higher-beta future-leaders basket.",
    },
}

MACRO_TICKERS = {
    "S&P 500": "^GSPC",
    "Brent Crude": "BZ=F",
    "USD/INR": "INR=X",
    "US 10Y Yield": "^TNX",
    "VIX": "^VIX",
}

DEFAULT_EVENTS = [
    {"date": "2008-09-15", "event": "Lehman collapse / GFC"},
    {"date": "2013-05-22", "event": "Fed taper tantrum begins"},
    {"date": "2016-11-08", "event": "US election / risk repricing"},
    {"date": "2020-03-11", "event": "COVID declared pandemic"},
    {"date": "2022-02-24", "event": "Russia-Ukraine war"},
    {"date": "2025-04-02", "event": "Tariff shock / global risk-off"},
]

ZONE_ORDER = [
    "Extreme Buy",
    "Buy",
    "Accumulate",
    "Neutral",
    "Reduce",
    "Sell",
    "Extreme Sell",
]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def download_price_data(ticker: str, start: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=True,
        actions=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.rename(columns=str.title)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.reset_index().rename(columns={"Date": "date"})
    if "Close" not in data.columns:
        raise ValueError(f"Close column missing for {ticker}")
    return data[["date", "Open", "High", "Low", "Close", "Volume"]].copy()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def download_macro_series(start: str) -> pd.DataFrame:
    frames = []
    for label, ticker in MACRO_TICKERS.items():
        try:
            data = download_price_data(ticker, start)
            data = data[["date", "Close"]].rename(columns={"Close": label})
            frames.append(data)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["date"])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def parse_valuation_csv(file_obj) -> pd.DataFrame:
    if file_obj is None:
        return pd.DataFrame()

    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, bytes):
            content = io.BytesIO(raw)
        else:
            content = io.StringIO(raw)
        df = pd.read_csv(content)
    else:
        df = pd.read_csv(file_obj)

    df.columns = [c.strip() for c in df.columns]
    column_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in {"date"}:
            column_map[col] = "date"
        elif lower in {"p/e", "pe", "p e"}:
            column_map[col] = "pe"
        elif lower in {"p/b", "pb", "p b"}:
            column_map[col] = "pb"
        elif lower in {"div yield %", "dividend yield", "div yield", "dy", "div_yield"}:
            column_map[col] = "dy"
        elif lower in {"close", "index close", "price"}:
            column_map[col] = "close"
        elif lower in {"indexname", "index name"}:
            column_map[col] = "index_name"

    df = df.rename(columns=column_map)
    required_any = {"date", "pe", "pb", "dy"}
    if "date" not in df.columns or len(required_any.intersection(set(df.columns))) < 2:
        raise ValueError(
            "Valuation CSV must have Date and at least two of P/E, P/B, Div Yield % columns."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    keep_cols = [c for c in ["date", "pe", "pb", "dy", "close", "index_name"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["date"]).sort_values("date").copy()

    for col in ["pe", "pb", "dy", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_default_valuation(index_name: str) -> pd.DataFrame:
    path = Path(INDEX_CONFIG[index_name]["default_valuation_file"])
    if not path.exists():
        return pd.DataFrame()
    return parse_valuation_csv(path)


def prepare_events_df(custom_file) -> pd.DataFrame:
    if custom_file is not None:
        df = pd.read_csv(custom_file)
        if {"date", "event"}.difference({c.lower() for c in df.columns}):
            df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["date", "event"]).copy()
        return df[["date", "event"]].sort_values("date")

    df = pd.DataFrame(DEFAULT_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def winsorize_series(s: pd.Series, lower=0.02, upper=0.98) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def label_from_score(score: float) -> str:
    if score >= 2.5:
        return "Extreme Buy"
    if score >= 1.5:
        return "Buy"
    if score >= 0.5:
        return "Accumulate"
    if score > -0.5:
        return "Neutral"
    if score > -1.5:
        return "Reduce"
    if score > -2.5:
        return "Sell"
    return "Extreme Sell"


def compute_percentile_score(
    value: float,
    history: pd.Series,
    invert: bool = False,
    strong_buy_q: float = 0.20,
    buy_q: float = 0.35,
    reduce_q: float = 0.65,
    sell_q: float = 0.80,
) -> float:
    hist = history.dropna()
    if len(hist) < 50 or pd.isna(value):
        return 0.0

    q20 = hist.quantile(strong_buy_q)
    q35 = hist.quantile(buy_q)
    q65 = hist.quantile(reduce_q)
    q80 = hist.quantile(sell_q)

    score = 0.0
    if not invert:
        if value <= q20:
            score = 1.0
        elif value <= q35:
            score = 0.5
        elif value >= q80:
            score = -1.0
        elif value >= q65:
            score = -0.5
    else:
        if value >= q80:
            score = 1.0
        elif value >= q65:
            score = 0.5
        elif value <= q20:
            score = -1.0
        elif value <= q35:
            score = -0.5
    return score


def compute_regime_frame(
    price_df: pd.DataFrame,
    valuation_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    valuation_weight: float = 1.0,
    technical_weight: float = 1.0,
    macro_weight: float = 1.0,
    score_smoothing: int = 10,
) -> pd.DataFrame:
    df = price_df.copy().sort_values("date").reset_index(drop=True)
    df["ret_1d"] = df["Close"].pct_change()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()
    df["ath"] = df["Close"].cummax()
    df["drawdown"] = (df["Close"] / df["ath"]) - 1.0

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Technical score: cheapness from drawdown, caution if trend weak, greed/fear via RSI
    tech_drawdown = np.select(
        [
            df["drawdown"] <= -0.30,
            df["drawdown"] <= -0.18,
            df["drawdown"] >= -0.03,
        ],
        [1.0, 0.5, -0.5],
        default=0.0,
    )
    tech_trend = np.select(
        [
            df["Close"] >= df["ma_200"] * 1.08,
            df["Close"] < df["ma_200"] * 0.92,
            df["Close"] < df["ma_200"],
        ],
        [-0.5, 0.25, 0.15],
        default=0.0,
    )
    tech_rsi = np.select(
        [
            df["rsi_14"] <= 30,
            df["rsi_14"] <= 40,
            df["rsi_14"] >= 75,
            df["rsi_14"] >= 65,
        ],
        [0.5, 0.25, -0.5, -0.25],
        default=0.0,
    )
    df["technical_score"] = technical_weight * (tech_drawdown + tech_trend + tech_rsi)

    if not valuation_df.empty:
        val = valuation_df.copy().sort_values("date")
        val_cols = [c for c in ["pe", "pb", "dy", "close"] if c in val.columns]
        val = val[["date"] + val_cols].drop_duplicates(subset=["date"], keep="last")
        df = pd.merge_asof(df.sort_values("date"), val.sort_values("date"), on="date", direction="backward")

        for col in ["pe", "pb", "dy"]:
            if col in df.columns:
                df[col] = winsorize_series(df[col])

        pe_score = (
            df["pe"].expanding().apply(
                lambda x: compute_percentile_score(x.iloc[-1], x, invert=False), raw=False
            )
            if "pe" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        pb_score = (
            df["pb"].expanding().apply(
                lambda x: compute_percentile_score(x.iloc[-1], x, invert=False), raw=False
            )
            if "pb" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        dy_score = (
            df["dy"].expanding().apply(
                lambda x: compute_percentile_score(x.iloc[-1], x, invert=True), raw=False
            )
            if "dy" in df.columns
            else pd.Series(0.0, index=df.index)
        )

        available_counts = (
            (~pe_score.eq(0)).astype(int) + (~pb_score.eq(0)).astype(int) + (~dy_score.eq(0)).astype(int)
        ).replace(0, np.nan)
        df["valuation_score"] = valuation_weight * ((pe_score.fillna(0) + pb_score.fillna(0) + dy_score.fillna(0)) / available_counts).fillna(0)
    else:
        df["valuation_score"] = 0.0

    if not macro_df.empty:
        macro = macro_df.copy().sort_values("date")
        df = pd.merge_asof(df.sort_values("date"), macro.sort_values("date"), on="date", direction="backward")
        macro_signal_parts = []

        if "Brent Crude" in df.columns:
            brent_ma = df["Brent Crude"].rolling(200).mean()
            macro_signal_parts.append(np.where(df["Brent Crude"] > brent_ma * 1.08, -0.35, 0.0))
            macro_signal_parts.append(np.where(df["Brent Crude"] < brent_ma * 0.95, 0.15, 0.0))

        if "USD/INR" in df.columns:
            fx_ma = df["USD/INR"].rolling(200).mean()
            macro_signal_parts.append(np.where(df["USD/INR"] > fx_ma * 1.03, -0.30, 0.0))
            macro_signal_parts.append(np.where(df["USD/INR"] < fx_ma * 0.98, 0.10, 0.0))

        if "S&P 500" in df.columns:
            sp_ma = df["S&P 500"].rolling(200).mean()
            macro_signal_parts.append(np.where(df["S&P 500"] < sp_ma, -0.25, 0.10))

        if "US 10Y Yield" in df.columns:
            us10_ma = df["US 10Y Yield"].rolling(200).mean()
            macro_signal_parts.append(np.where(df["US 10Y Yield"] > us10_ma * 1.05, -0.15, 0.0))

        if "VIX" in df.columns:
            vix_ma = df["VIX"].rolling(200).mean()
            macro_signal_parts.append(np.where(df["VIX"] > vix_ma * 1.15, -0.20, 0.0))
            macro_signal_parts.append(np.where(df["VIX"] > 30, 0.10, 0.0))  # panic can create opportunity

        if macro_signal_parts:
            df["macro_score"] = macro_weight * np.sum(macro_signal_parts, axis=0)
        else:
            df["macro_score"] = 0.0
    else:
        df["macro_score"] = 0.0

    df["raw_score"] = df["valuation_score"] + df["technical_score"] + df["macro_score"]
    df["score"] = df["raw_score"].ewm(span=score_smoothing, adjust=False).mean()
    df["zone"] = df["score"].apply(label_from_score)

    # Forward returns for diagnostics
    for days, label in [(63, "fwd_3m"), (126, "fwd_6m"), (252, "fwd_12m")]:
        df[label] = df["Close"].shift(-days) / df["Close"] - 1.0

    return df


def summarize_current_state(df: pd.DataFrame) -> Dict[str, float]:
    latest = df.dropna(subset=["score"]).iloc[-1]
    return {
        "date": latest["date"],
        "close": float(latest["Close"]),
        "score": float(latest["score"]),
        "zone": latest["zone"],
        "valuation_score": float(latest.get("valuation_score", 0.0)),
        "technical_score": float(latest.get("technical_score", 0.0)),
        "macro_score": float(latest.get("macro_score", 0.0)),
        "drawdown_pct": float(latest.get("drawdown", np.nan) * 100),
        "rsi_14": float(latest.get("rsi_14", np.nan)),
        "pe": float(latest["pe"]) if "pe" in latest.index and pd.notna(latest["pe"]) else np.nan,
        "pb": float(latest["pb"]) if "pb" in latest.index and pd.notna(latest["pb"]) else np.nan,
        "dy": float(latest["dy"]) if "dy" in latest.index and pd.notna(latest["dy"]) else np.nan,
    }


def build_price_figure(df: pd.DataFrame, events_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["Close"],
            mode="lines",
            name="Close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma_200"],
            mode="lines",
            name="200 DMA",
            opacity=0.5,
        )
    )

    for _, row in events_df.iterrows():
        fig.add_vline(x=row["date"], line_dash="dot", opacity=0.2)
        fig.add_annotation(
            x=row["date"],
            y=df["Close"].max(),
            text=row["event"],
            textangle=-90,
            yanchor="top",
            showarrow=False,
            opacity=0.6,
            font=dict(size=10),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Index level",
        height=500,
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_score_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["score"], mode="lines", name="Composite score")
    )
    threshold_lines = [
        (2.5, "Extreme Buy"),
        (1.5, "Buy"),
        (0.5, "Accumulate"),
        (-0.5, "Neutral"),
        (-1.5, "Reduce"),
        (-2.5, "Sell"),
    ]
    for y, label in threshold_lines:
        fig.add_hline(y=y, line_dash="dot", opacity=0.3, annotation_text=label)

    fig.update_layout(
        title="Composite valuation / regime score",
        xaxis_title="Date",
        yaxis_title="Score",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_valuation_figure(df: pd.DataFrame) -> Optional[go.Figure]:
    series = []
    for col, label in [("pe", "P/E"), ("pb", "P/B"), ("dy", "Dividend Yield %")]:
        if col in df.columns and df[col].notna().sum() > 10:
            series.append((col, label))

    if not series:
        return None

    fig = go.Figure()
    for col, label in series:
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=label))
    fig.update_layout(
        title="Valuation history",
        xaxis_title="Date",
        yaxis_title="Value",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_zone_return_table(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("zone")[["fwd_3m", "fwd_6m", "fwd_12m"]].mean().reindex(ZONE_ORDER)
    counts = df["zone"].value_counts().reindex(ZONE_ORDER)
    out = grp.copy()
    out["observations"] = counts
    out = out.dropna(how="all")
    for col in ["fwd_3m", "fwd_6m", "fwd_12m"]:
        if col in out.columns:
            out[col] = (out[col] * 100).round(2)
    out["observations"] = out["observations"].fillna(0).astype(int)
    return out.reset_index().rename(columns={"index": "zone", "zone": "Zone"})


def explain_zone(zone: str) -> str:
    mapping = {
        "Extreme Buy": "Valuation and/or panic conditions are unusually favorable. Best used for staged but aggressive accumulation.",
        "Buy": "Good risk/reward. Start or add meaningfully, but still in tranches.",
        "Accumulate": "Reasonable zone for SIPs or moderate additions.",
        "Neutral": "No edge. Let fresh data or a better price improve the setup.",
        "Reduce": "Starting to look stretched. Avoid fresh aggressive buying.",
        "Sell": "Rich setup. Trim if your process allows tactical rebalancing.",
        "Extreme Sell": "Euphoric / overheated regime. Strong rebalance candidate.",
    }
    return mapping.get(zone, "")


def load_text_template(index_name: str) -> str:
    return f"""
How to use this app:
1. Pick {index_name}.
2. Leave valuation upload empty to run in price+macro mode.
3. Upload official NSE valuation history CSV later to activate PE/PB/dividend-yield scoring.
4. Read the current zone, then inspect the historical score chart and forward-return table.
"""


# Sidebar
st.sidebar.title("Nifty Regime Tool")
index_name = st.sidebar.selectbox("Index", list(INDEX_CONFIG.keys()))
start_date = st.sidebar.date_input("Start date", pd.Timestamp("2005-01-01"))
score_smoothing = st.sidebar.slider("Score smoothing (days)", 3, 30, 10)
st.sidebar.markdown("### Weights")
valuation_weight = st.sidebar.slider("Valuation weight", 0.0, 2.0, 1.0, 0.1)
technical_weight = st.sidebar.slider("Technical weight", 0.0, 2.0, 1.0, 0.1)
macro_weight = st.sidebar.slider("Macro weight", 0.0, 2.0, 1.0, 0.1)

st.sidebar.markdown("### Optional inputs")
valuation_upload = st.sidebar.file_uploader(
    "Upload valuation CSV",
    type=["csv"],
    help="Use official NSE export with Date, P/E, P/B, Div Yield %.",
)
events_upload = st.sidebar.file_uploader(
    "Upload custom events CSV",
    type=["csv"],
    help="CSV with columns: date,event",
)
show_event_markers = st.sidebar.checkbox("Show event markers", value=True)

st.title("Nifty 50 / Nifty Next 50 Buy-Sell Regime Tool")
st.caption(
    "Composite engine combining valuation, technical stress, and macro overlays. "
    "Works in price+macro mode even if valuation history is unavailable."
)
st.info(load_text_template(index_name))

try:
    price_df = download_price_data(INDEX_CONFIG[index_name]["price_ticker"], str(start_date))
    macro_df = download_macro_series(str(start_date))
    if valuation_upload is not None:
        valuation_df = parse_valuation_csv(valuation_upload)
    else:
        valuation_df = load_default_valuation(index_name)

    events_df = prepare_events_df(events_upload)
    regime_df = compute_regime_frame(
        price_df=price_df,
        valuation_df=valuation_df,
        macro_df=macro_df,
        valuation_weight=valuation_weight,
        technical_weight=technical_weight,
        macro_weight=macro_weight,
        score_smoothing=score_smoothing,
    )

    state = summarize_current_state(regime_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current zone", state["zone"])
    c2.metric("Composite score", f'{state["score"]:.2f}')
    c3.metric("Drawdown from ATH", f'{state["drawdown_pct"]:.1f}%')
    c4.metric("RSI(14)", f'{state["rsi_14"]:.1f}')

    c5, c6, c7 = st.columns(3)
    pe_text = "NA" if np.isnan(state["pe"]) else f'{state["pe"]:.2f}'
    pb_text = "NA" if np.isnan(state["pb"]) else f'{state["pb"]:.2f}'
    dy_text = "NA" if np.isnan(state["dy"]) else f'{state["dy"]:.2f}%'
    c5.metric("P/E", pe_text)
    c6.metric("P/B", pb_text)
    c7.metric("Div Yield", dy_text)

    c8, c9, c10 = st.columns(3)
    c8.metric("Valuation score", f'{state["valuation_score"]:.2f}')
    c9.metric("Technical score", f'{state["technical_score"]:.2f}')
    c10.metric("Macro score", f'{state["macro_score"]:.2f}')

    st.markdown(f"**Interpretation:** {explain_zone(state['zone'])}")

    price_fig = build_price_figure(
        regime_df,
        events_df if show_event_markers else pd.DataFrame(columns=["date", "event"]),
        f"{index_name} price vs major events",
    )
    st.plotly_chart(price_fig, use_container_width=True)

    score_fig = build_score_figure(regime_df)
    st.plotly_chart(score_fig, use_container_width=True)

    val_fig = build_valuation_figure(regime_df)
    if val_fig is not None:
        st.plotly_chart(val_fig, use_container_width=True)
    else:
        st.warning(
            "Valuation history not found. The app is running in price+macro mode. "
            "Upload official NSE valuation CSV to activate PE/PB/dividend-yield scoring."
        )

    st.subheader("Historical forward returns by zone")
    zone_table = build_zone_return_table(regime_df)
    st.dataframe(zone_table, use_container_width=True)

    st.subheader("Latest observations")
    display_cols = [
        "date", "Close", "drawdown", "rsi_14", "valuation_score", "technical_score",
        "macro_score", "score", "zone"
    ]
    for col in ["pe", "pb", "dy"]:
        if col in regime_df.columns:
            display_cols.append(col)
    latest_view = regime_df[display_cols].tail(1000).copy()
    if "drawdown" in latest_view.columns:
        latest_view["drawdown"] = (latest_view["drawdown"] * 100).round(2)
    st.dataframe(latest_view, use_container_width=True)

    with st.expander("Methodology / scoring logic"):
        st.markdown("""
**Valuation**
- Low P/E and low P/B relative to their own history are bullish.
- High dividend yield relative to its own history is bullish.
- These are scored on historical percentile bands, not fixed magic numbers.

**Technical**
- Deep drawdowns increase buy-score.
- Very strong price far above the 200 DMA reduces score.
- RSI captures short-term panic/euphoria.

**Macro**
- India-unfriendly macro conditions such as expensive Brent crude, rupee weakness,
  rising US yields, risk-off in S&P 500, and elevated VIX reduce the score.
- Panic conditions can still create opportunity, so the macro block is not purely one-way.

**Important**
- This is a regime tool, not an exact-top exact-bottom machine.
- Best use: staged allocation, rebalancing, and comparing today's setup with history.
        """)

except Exception as exc:
    st.error(f"App failed: {exc}")
    st.stop()
