# app.py
# Crypto Predator – Web (Binance/MEXC, REST-only, mirror+fallback)
# Author: Uğurcan & Hacker
# How to run: streamlit run app.py

from __future__ import annotations
import time, math, random, functools
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import streamlit as st
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# ============ Page & basic opts ============
st.set_page_config(page_title="Crypto Predator – Web", page_icon="🧠", layout="wide")

# ------------- Mirror bases ---------------
BIN_FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi.binance.me",
    "https://fapi.binance.cc",
]
BIN_API_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api.binance.me",
]
MEXC_BASES = [
    "https://api.mexc.com",
    "https://www.mexc.com",
]

# ------------- UI options -----------------
with st.sidebar:
    st.header("Ayarlar")
    enable_mexc_tab = st.checkbox("MEXC 0 Fee sekmesi", value=True)
    row_cap = st.slider("Tablo satır limiti", 100, 1200, 400)
    if st.button("🔄 Yenile"):
        st.cache_data.clear()
        st.experimental_rerun()

# ------------- Small utils ----------------
UA = {"User-Agent": "Mozilla/5.0"}

def _rest_try(bases: List[str], path: str, params: dict | None = None, timeout: int = 7):
    """Try mirrors in random order until 200 OK."""
    assert path.startswith("/"), "path must start with '/…'"
    bases = list(bases)
    random.shuffle(bases)
    last_err = None
    for b in bases:
        url = f"{b}{path}"
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            if r.status_code == 200:
                return r.json()
            last_err = f"{r.status_code} {r.text[:160]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"All mirrors failed for {path}: {last_err}")

def pct_color(v: float) -> str:
    arrow = "▲" if v >= 0 else "▼"
    color = "#22c55e" if v >= 0 else "#ef4444"
    return f"<span style='color:{color};font-weight:600'>{arrow} {v:.2f}%</span>"

# ------------- Indicators -----------------
def ema(x: pd.Series, n: int):
    return x.ewm(span=n, adjust=False).mean()

def rsi(x: pd.Series, n: int = 14):
    d = x.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    rs = up / (dn.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(x: pd.Series):
    e12 = ema(x, 12); e26 = ema(x, 26)
    line = e12 - e26
    sig  = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ------------- Binance REST ---------------
@st.cache_data(ttl=30)
def fetch_tickers(use_futures: bool = True) -> Dict[str, dict]:
    """Return dict: { 'BTC/USDT:USDT': {'last':..,'percentage':..}, ... }"""
    if use_futures:
        rows = _rest_try(BIN_FAPI_BASES, "/fapi/v1/ticker/24hr")
        out = {}
        for t in rows:
            s = t.get("symbol", "")
            if not s.endswith("USDT"):
                continue
            sym = s.replace("USDT", "/USDT:USDT")
            out[sym] = {
                "last": float(t["lastPrice"]),
                "percentage": float(t["priceChangePercent"])
            }
        return out
    else:
        rows = _rest_try(BIN_API_BASES, "/api/v3/ticker/24hr")
        out = {}
        for t in rows:
            s = t.get("symbol", "")
            if not s.endswith("USDT"):
                continue
            sym = s.replace("USDT", "/USDT")
            out[sym] = {
                "last": float(t["lastPrice"]),
                "percentage": float(t["priceChangePercent"])
            }
        return out

@st.cache_data(ttl=300)
def list_symbols(use_futures: bool = True) -> List[str]:
    """USDT çiftleri listesi (display format)."""
    try:
        if use_futures:
            info = _rest_try(BIN_FAPI_BASES, "/fapi/v1/exchangeInfo")
            syms = [s["symbol"] for s in info.get("symbols", []) if s.get("quoteAsset") == "USDT"]
            return [s.replace("USDT", "/USDT:USDT") for s in syms]
        else:
            rows = _rest_try(BIN_API_BASES, "/api/v3/ticker/24hr")
            syms = [t["symbol"] for t in rows if t["symbol"].endswith("USDT")]
            return [s.replace("USDT", "/USDT") for s in syms]
    except Exception:
        # fallback to ticker-derived
        tk = fetch_tickers(use_futures=use_futures)
        return list(tk.keys())

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol: str, tf: str = "5m", limit: int = 210, use_futures: bool = True) -> List[List[float]]:
    """Return klines: [[ts,open,high,low,close,vol], ...]"""
    s = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    if use_futures:
        rows = _rest_try(BIN_FAPI_BASES, "/fapi/v1/klines", {"symbol": s, "interval": tf, "limit": limit})
    else:
        rows = _rest_try(BIN_API_BASES, "/api/v3/klines", {"symbol": s, "interval": tf, "limit": limit})
    return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows]

# ------------- MEXC REST ------------------
@st.cache_data(ttl=30)
def mexc_tickers() -> Dict[str, dict]:
    rows = _rest_try(MEXC_BASES, "/api/v3/ticker/24hr")
    out = {}
    for t in rows:
        s = t.get("symbol", "")
        if s.endswith("USDT"):
            sym = s.replace("USDT", "/USDT")
            out[sym] = {
                "last": float(t["lastPrice"]),
                "percentage": float(t.get("priceChangePercent", 0.0))
            }
    return out

@st.cache_data(ttl=60)
def mexc_ohlcv(symbol: str, tf: str = "5m", limit: int = 210):
    s = symbol.replace("/USDT", "USDT")
    rows = _rest_try(MEXC_BASES, "/api/v3/klines", {"symbol": s, "interval": tf, "limit": limit})
    return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows]

# ------------- Sentiment (very light) -----
@st.cache_data(ttl=120)
def sentiment_score() -> int:
    total = 0
    sources = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    ]
    try:
        for u in sources:
            xml = requests.get(u, headers=UA, timeout=6).text.lower()
            soup = BeautifulSoup(xml, "xml")
            text = " ".join([it.title.text for it in soup.select("item")[:40]]).lower()
            pos = sum(text.count(k) for k in ["etf", "approve", "upgrade", "bull", "adoption"])
            neg = sum(text.count(k) for k in ["hack", "ban", "lawsuit", "liquidation", "exploit"])
            total += (pos - neg)
    except Exception:
        pass
    return int(total)

# ------------- Scoring / Signals ----------
def score_symbol(symbol: str, use_futures: bool, tf: str = "5m") -> dict | None:
    try:
        rows = fetch_ohlcv(symbol, tf, 210, use_futures)
        if not rows or len(rows) < 80:
            return None
        df = pd.DataFrame(rows, columns=["ts","o","h","l","c","v"])
        c = df["c"].astype(float); h = df["h"].astype(float); l = df["l"].astype(float)
        r = float(rsi(c).iloc[-1])
        m_line, m_sig, _ = macd(c)
        macd_v = float(m_line.iloc[-1]); macd_s = float(m_sig.iloc[-1])
        e50 = ema(c, 50).iloc[-1]; m200 = c.rolling(200).mean().iloc[-1]
        trend = "Boğa" if e50 > m200 else ("Ayı" if e50 < m200 else "Nötr")

        s = sentiment_score()
        direction, comment, score = "Bekle", "Teyit beklenmeli.", 0 + s
        if r < 30:   direction, comment, score = "Long",  "RSI düşük → alım", score+40
        elif r > 70: direction, comment, score = "Short", "RSI yüksek → satış", score+40
        elif macd_v > macd_s: direction, comment, score = "Long",  "MACD yukarı kesişim", score+25
        elif macd_v < macd_s: direction, comment, score = "Short", "MACD aşağı kesişim", score+25
        elif trend == "Boğa": direction, comment, score = "Long",  "Trend boğa", score+20
        elif trend == "Ayı":  direction, comment, score = "Short", "Trend ayı", score+20
        price = float(c.iloc[-1])
        return dict(
            Yön=direction, Tarz="Swing", TF=tf, Sembol=symbol, Fiyat=price, Skor=int(score),
            RSI=round(r,2), MACD=round(macd_v,4), Trend=trend, Yorum=comment,
            Lev="5x", SL="3%", TP="15%", Not=""
        )
    except Exception:
        return None

def scan_symbols(symbols: List[str], use_futures: bool, tf: str, cap: int) -> pd.DataFrame:
    out: List[dict] = []
    for s in symbols[:cap*2]:  # biraz fazla dene
        row = score_symbol(s, use_futures, tf)
        if row: out.append(row)
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("Skor", ascending=False).head(cap).reset_index(drop=True)
    return df

def scan_scalp(symbols: List[str], cap: int = 120) -> pd.DataFrame:
    out: List[dict] = []
    s = sentiment_score()
    for sym in symbols[:cap*3]:
        try:
            rows = fetch_ohlcv(sym, "1m", 250, True)
            if not rows or len(rows) < 60: 
                continue
            df = pd.DataFrame(rows, columns=["ts","o","h","l","c","v"]).astype(float)
            c = df["c"]; h=df["h"]; l=df["l"]
            e9, e21 = ema(c,9), ema(c,21)
            a = atr(h,l,c,14); atrp = (a/c).iloc[-1]*100
            direction, comment, score = "Bekle", "Sıkışma takip", s
            if e9.iloc[-1] > e21.iloc[-1] and c.iloc[-1] > e9.iloc[-1]:
                direction, comment, score = "Long",  "EMA9>21 + momentum", score+35
            elif e9.iloc[-1] < e21.iloc[-1] and c.iloc[-1] < e9.iloc[-1]:
                direction, comment, score = "Short", "EMA9<21 + momentum", score+35
            if 0.2 <= atrp <= 1.2: score += 10
            out.append(dict(
                Yön=direction, Tarz="Scalp", TF="1m", Sembol=sym, Fiyat=float(c.iloc[-1]),
                Skor=int(score), RSI=round(float(rsi(c).iloc[-1]),2),
                MACD=round(float((ema(c,12)-ema(c,26)).iloc[-1]),6),
                Trend="—", Yorum=comment, Lev="10x", SL="0.7%", TP="1.5%", Not=""
            ))
        except Exception:
            continue
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("Skor", ascending=False).head(120).reset_index(drop=True)
    return df

def quick_futures_triggers(symbols: List[str], cap: int = 60) -> pd.DataFrame:
    out = []
    s = sentiment_score()
    for sym in symbols[:cap*4]:
        try:
            rows = fetch_ohlcv(sym, "1m", 130, True)
            if not rows or len(rows) < 40: 
                continue
            df = pd.DataFrame(rows, columns=["ts","o","h","l","c","v"]).astype(float)
            c=df["c"]; h=df["h"]; l=df["l"]
            e9, e21 = ema(c,9), ema(c,21)
            m_line, m_sig, _ = macd(c)
            r = float(rsi(c).iloc[-1])
            a = atr(h,l,c,14); atrp = (a/c).iloc[-1]*100
            direction, comment, score = None, None, s
            if e9.iloc[-1] > e21.iloc[-1] and m_line.iloc[-1] > m_sig.iloc[-1] and r>52 and atrp>0.08:
                direction, comment, score = "Long",  "1m momentum long", s+45
            elif e9.iloc[-1] < e21.iloc[-1] and m_line.iloc[-1] < m_sig.iloc[-1] and r<48 and atrp>0.08:
                direction, comment, score = "Short", "1m momentum short", s+45
            if direction:
                out.append(dict(
                    Yön=direction, Tarz="Scalp", TF="1m", Sembol=sym, Fiyat=float(c.iloc[-1]),
                    Skor=int(score), RSI=round(r,2), MACD=round(float(m_line.iloc[-1]),6),
                    Trend="—", Yorum=comment, Lev="10x", SL="0.6%", TP="1.2%", Not=""
                ))
        except Exception:
            continue
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("Skor", ascending=False).head(cap).reset_index(drop=True)
    return df

# ------------- Predefined lists -----------
KAFA_COINLER = ['BTC','ETH','SOL','AVAX','SEI','MAGIC','BNB','XRP','ADA','DOGE','SUI']
KAFA_COINLER = [f"{c}/USDT:USDT" for c in KAFA_COINLER]

MEXC_ZERO = ['SOL','SUI','ADA','PEPE','PUMP','PENGU','LTC','ONDO','HYPE','LDO','AAVE',
             'XLM','POPCAT','ETHFI','APT','TONU','SEI','WLD','TAO','NEAR','SHIB']
MEXC_ZERO = [f"{c}/USDT" for c in MEXC_ZERO]

# ------------- Tabs -----------------------
st.title("Crypto Predator – Web")

tab_all, tab_kafa, tab_live, tab_scalp, tab_fast, tab_mexc, tab_top, tab_news, tab_rm = st.tabs(
    ["📊 Tüm Coinler – Sinyal", "🧠 Kafa Coinler", "💹 Anlık Fiyatlar",
     "⚡ VUR-KAÇ (Scalp)", "🚀 Vadeli Hızlı", "🟧 MEXC 0 Fee",
     "🏁 Top Gainers / Losers", "📰 Haberler", "💼 Kasa Yönetimi (öneri)"]
)

# ---- Tüm Coinler (Spot) ----
with tab_all:
    st.caption("Spot USDT çiftlerinde 5m Swing skoru (REST, mirror+fallback).")
    syms_spot = list_symbols(use_futures=False)
    df_all = scan_symbols(syms_spot, use_futures=False, tf="5m", cap=min(600, row_cap))
    st.dataframe(df_all, use_container_width=True, height=520)

# ---- Kafa Coinler (Futures) ----
with tab_kafa:
    st.caption("Binance Futures – favori/kafa coinler (5m Swing).")
    df_kafa = scan_symbols(KAFA_COINLER, use_futures=True, tf="5m", cap=len(KAFA_COINLER))
    st.dataframe(df_kafa, use_container_width=True, height=520)

# ---- Live prices (Futures) ----
with tab_live:
    st.caption("Binance Futures anlık fiyat ve 24h değişim (oklar yeşil/kırmızı).")
    tks = fetch_tickers(use_futures=True)
    rows = []
    for s, t in tks.items():
        last = t.get("last"); pct = float(t.get("percentage", 0.0))
        rows.append((s, last, pct))
    rows.sort(key=lambda x: x[0])
    dfp = pd.DataFrame(rows, columns=["Sembol","Fiyat","%24h"])
    # HTML renkli oklar
    dfp["%24h"] = [pct_color(v) for v in dfp["%24h"]]
    st.write(
        dfp.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

# ---- Scalp (1m, Futures) ----
with tab_scalp:
    st.caption("1m scalping – EMA9/21 + volatilite filtresi.")
    syms_fut = list_symbols(use_futures=True)
    df_s = scan_scalp(syms_fut, cap=min(200, row_cap))
    st.dataframe(df_s, use_container_width=True, height=520)

# ---- Fast triggers (1m, Futures) ----
with tab_fast:
    st.caption("Çok hızlı 1m momentum tetikleri (daha agresif).")
    syms_fut = list_symbols(use_futures=True)
    df_f = quick_futures_triggers(syms_fut, cap=min(100, row_cap))
    st.dataframe(df_f, use_container_width=True, height=520)

# ---- MEXC 0 Fee (Spot) ----
with tab_mexc:
    if not enable_mexc_tab:
        st.info("Ayarlar > 'MEXC 0 Fee sekmesi' kapalı.")
    else:
        st.caption("MEXC 0 Fee listesi (Spot 5m).")
        out = []
        for s in MEXC_ZERO:
            row = score_symbol(s, use_futures=False, tf="5m")  # MEXC REST OHLCV kullanalım mı? basit olsun diye spot binance'a kalabilir
            # Üstteki satır Binance spot'tan bakar; istersen mexc_ohlcv ile değiştirebilirsin:
            # rows = mexc_ohlcv(s, "5m", 210) ... (score fonksiyonunu ayrıştırmak gerekir)
            if row: out.append(row)
        df_m = pd.DataFrame(out).head(len(MEXC_ZERO))
        st.dataframe(df_m, use_container_width=True, height=520)

# ---- Top gainers / losers ----
with tab_top:
    st.caption("24h değişime göre en çok yükselen/düşen (Futures).")
    tk = fetch_tickers(True)
    arr = []
    for s,t in tk.items():
        if not s.endswith(":USDT"): continue
        pct = float(t.get("percentage", 0.0)); last = float(t.get("last", 0.0))
        arr.append((s,pct,last))
    arr.sort(key=lambda x:x[1], reverse=True)
    gain = arr[:15]
    arr.sort(key=lambda x:x[1])
    lose = arr[:15]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Gainers")
        st.dataframe(pd.DataFrame(gain, columns=["Sembol","%24h","Fiyat"]), use_container_width=True, height=420)
    with c2:
        st.subheader("Top Losers")
        st.dataframe(pd.DataFrame(lose, columns=["Sembol","%24h","Fiyat"]), use_container_width=True, height=420)

# ---- Haberler (TR) ----
@st.cache_data(ttl=90)
def fetch_news_tr() -> List[dict]:
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    ]
    items = []
    try:
        for u in feeds:
            xml = requests.get(u, headers=UA, timeout=6).text
            soup = BeautifulSoup(xml, "xml")
            for it in soup.select("item")[:30]:
                title = (it.title.text if it.title else "").strip()
                link  = (it.link.text if it.link else "").strip()
                pub   = (it.pubDate.text if it.pubDate else "")
                src   = u.split("//")[1].split("/")[0]
                # Basit TR çeviri (google free endpoint) – arıza ederse orijinali göster
                try:
                    tr = requests.get(
                        "https://translate.googleapis.com/translate_a/single",
                        params={"client":"gtx","sl":"auto","tl":"tr","dt":"t","q": title[:5000]},
                        timeout=4,
                    ).json()
                    title_tr = "".join(p[0] for p in tr[0] if p and p[0])
                except Exception:
                    title_tr = title
                items.append({"time": pub, "source": src, "title": title_tr, "link": link})
    except Exception:
        pass
    return items[:60]

with tab_news:
    st.caption("Kripto haber başlıkları (TR). Başlığa tıklayın.")
    news = fetch_news_tr()
    if news:
        dfN = pd.DataFrame(news)
        dfN["Başlık"] = dfN.apply(lambda r: f"[{r['title']}]({r['link']})", axis=1)
        st.markdown(dfN[["time","source","Başlık"]].to_markdown(index=False), unsafe_allow_html=True)
    else:
        st.info("Haberler alınamadı.")

# ---- Kasa Yönetimi (öneri) ----
with tab_rm:
    st.caption("Basit öneri motoru: Sermayeyi sinyal gücüne göre böler (trade açmaz).")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        equity = st.number_input("Sermaye ($)", min_value=50.0, value=500.0, step=50.0)
    with colB:
        max_trades = st.slider("Max pozisyon", 1, 12, 5)
    with colC:
        base_risk = st.slider("Risk % (pozisyon başına)", 1, 10, 3)
    with colD:
        use_src = st.selectbox("Kaynak", ["Vadeli Hızlı", "Scalp", "Kafa Coinler"])

    # Kaynak veriyi getir
    syms_fut = list_symbols(True)
    if use_src == "Vadeli Hızlı":
        base_df = quick_futures_triggers(syms_fut, cap=80)
    elif use_src == "Scalp":
        base_df = scan_scalp(syms_fut, cap=120)
    else:
        base_df = scan_symbols(KAFA_COINLER, True, "5m", cap=len(KAFA_COINLER))

    if base_df.empty:
        st.info("Öneri üretmek için yeterli veri yok.")
    else:
        # Skora göre normalize dağıtım
        df = base_df.head(max_trades).copy()
        w = (df["Skor"] - df["Skor"].min() + 1)
        w = w / w.sum()
        df["Risk%"] = w * base_risk * max_trades
        df["Notional"] = (df["Risk%"]/100.0) * equity
        # Varsayılan kaldıraç ve qty (yaklaşık)
        lev = 10.0
        df["Lev"] = df["Lev"].astype(str)
        df["Margin $"] = df["Notional"] / lev
        df["Adet"] = df["Notional"] / df["Fiyat"]
        show = df[["Sembol","Yön","Tarz","TF","Skor","Fiyat","Risk%","SL","TP","Lev","Notional","Margin $","Adet","Yorum"]]
        st.dataframe(show, use_container_width=True, height=520)

# ---- Footer / heartbeat ----
st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')} • Data via Binance/MEXC public REST (mirrors + fallback) • Cached to reduce rate-limit.")
