# app.py
# Crypto Predator – Web (Streamlit)
# Fallback-first: Binance engellenirse CoinGecko/MEXC ile boş ekran bırakmaz.

import os, time, math, json, random
from datetime import datetime, timezone
from functools import lru_cache

import requests
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup

# ========= Settings =========
BINANCE_SPOT_TICKER = [
    "https://api.binance.com/api/v3/ticker/24hr",
    "https://api1.binance.com/api/v3/ticker/24hr",
    "https://api2.binance.com/api/v3/ticker/24hr",
    "https://api3.binance.com/api/v3/ticker/24hr",
]
BINANCE_FUT_TICKER = [
    "https://fapi.binance.com/fapi/v1/ticker/24hr",
    "https://fapi.binance.com/fapi/v1/ticker/24hr",  # tek domain ama bırakıyoruz
]
BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"   # futures
BINANCE_KLINES_SPOT = "https://api.binance.com/api/v3/klines" # spot

# CoinGecko fallback (no key)
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"

# UI lists
KAFA_COINLER = ['BTC','ETH','SOL','AVAX','SEI','MAGIC','BNB','XRP','ADA','DOGE','SUI']
MEXC_ZERO = ['SOL','SUI','ADA','PEPE','PUMP','PENGU','LTC','ONDO','HYPE','LDO','AAVE','XLM',
             'POPCAT','ETHFI','APT','TONU','SEI','WLD','TAO','NEAR','SHIB']

# Cache TTL’ler
TTL_TICKERS = 60
TTL_NEWS = 120
TTL_OHLC = 60

st.set_page_config(page_title="Crypto Predator – Web", page_icon="🕶️", layout="wide")

# ========== Helpers ==========
def _headers():
    return {"User-Agent": "Mozilla/5.0 (CryptoPredator/1.0)"}

def _try_get(url, params=None, timeout=8):
    try:
        r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise e

def _try_get_text(url, timeout=8):
    r = requests.get(url, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.text

def translate_tr(txt: str) -> str:
    if not txt:
        return txt
    try:
        u = "https://translate.googleapis.com/translate_a/single"
        params = {"client":"gtx","sl":"auto","tl":"tr","dt":"t","q":txt[:5000]}
        j = requests.get(u, params=params, timeout=5).json()
        return "".join(part[0] for part in j[0] if part and part[0])
    except Exception:
        return txt

@st.cache_data(ttl=TTL_TICKERS, show_spinner=False)
def fetch_binance_futures_tickers():
    last_err = None
    for url in BINANCE_FUT_TICKER:
        try:
            return _try_get(url)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Binance futures tickers blocked: {last_err}")

@st.cache_data(ttl=TTL_TICKERS, show_spinner=False)
def fetch_binance_spot_tickers():
    last_err = None
    for url in BINANCE_SPOT_TICKER:
        try:
            return _try_get(url)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Binance spot tickers blocked: {last_err}")

@st.cache_data(ttl=TTL_TICKERS, show_spinner=False)
def fetch_coingecko_markets(per_page=250, page=1):
    params = {
        "vs_currency":"usd",
        "order":"market_cap_desc",
        "per_page":per_page,
        "page":page,
        "price_change_percentage":"24h"
    }
    return _try_get(COINGECKO_MARKETS, params=params)

@st.cache_data(ttl=TTL_OHLC, show_spinner=False)
def fetch_klines(symbol: str, tf="1m", limit=200, futures=True):
    """
    Binance klines; engellenirse Exception atar. UI tarafında fallback/skip var.
    """
    base = BINANCE_KLINES if futures else BINANCE_KLINES_SPOT
    params = {"symbol": symbol.replace("/", ""), "interval": tf, "limit": limit}
    j = _try_get(base, params=params, timeout=8)
    # [ [openTime, open, high, low, close, volume, ...], ... ]
    rows = []
    for k in j:
        rows.append({
            "t": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low":  float(k[3]),
            "close":float(k[4]),
            "vol":  float(k[5]),
        })
    return pd.DataFrame(rows)

# basit teknikler
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    rs = up / (dn.replace(0, np.nan))
    return 100 - (100/(1+rs))

def macd(s):
    e12, e26 = ema(s,12), ema(s,26)
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# ========== UI Sidebar ==========
st.sidebar.header("Ayarlar")
enable_mexc_tab = st.sidebar.checkbox("MEXC 0 Fee sekmesi", value=True)
limit_rows = st.sidebar.slider("Tablo satır limiti", 50, 500, 400, 25)
st.sidebar.caption("Binance bazı ücretsiz host’larda engellenebilir. Fallback açık.")

# ========== Title ==========
st.title("Crypto Predator – Web")

# Tabs
tabs = st.tabs([
    "📊 Tüm Coinler – Sinyal",
    "🧠 Kafa Coinler",
    "💹 Anlık Fiyatlar",
    "⚡ VUR-KAÇ (Scalp)",
    "🚀 Vadeli Hızlı",
    "🪙 MEXC 0 Fee" if enable_mexc_tab else "🪙 MEXC 0 Fee (kapalı)",
    "📈 Top Gainers/Losers",
    "📰 Haberler",
    "💼 Kasa Yönetimi (öneri)"
])

# ========== Prices (with robust fallback) ==========
def read_prices():
    """
    Öncelik: Binance Futures → olmazsa Binance Spot → CoinGecko fallback.
    Dönüş: list(dict(symbol, last, pct))
    """
    try:
        fut = fetch_binance_futures_tickers()
        out = []
        for t in fut:
            sym = t.get("symbol","")
            if not sym.endswith("USDT"):
                continue
            last = float(t.get("lastPrice", t.get("last", 0)) or 0)
            pct = float(t.get("priceChangePercent", 0))
            out.append({"symbol": f"{sym[:-4]}/USDT", "last": last, "pct": pct, "src": "BINANCE FUT"})
        return out
    except Exception as e_fut:
        st.warning(f"Binance Futures engellenmiş görünüyor (fallback aktif). Detay: {e_fut}")
        try:
            spot = fetch_binance_spot_tickers()
            out = []
            for t in spot:
                sym = t.get("symbol","")
                if not sym.endswith("USDT"):
                    continue
                last = float(t.get("lastPrice", 0) or 0)
                pct = float(t.get("priceChangePercent", 0))
                out.append({"symbol": f"{sym[:-4]}/USDT", "last": last, "pct": pct, "src": "BINANCE SPOT"})
            return out
        except Exception as e_spot:
            st.warning(f"Binance Spot da engelli (CoinGecko’ya düştük). Detay: {e_spot}")
            cg = fetch_coingecko_markets(per_page=250, page=1)
            out = []
            for c in cg:
                sym = f"{c['symbol'].upper()}/USDT"
                last = float(c["current_price"] or 0)
                pct = float((c.get("price_change_percentage_24h") or 0))
                out.append({"symbol": sym, "last": last, "pct": pct, "src": "COINGECKO"})
            return out

# ========== Signal engine (very compact, skip on network fail) ==========
def compute_signal(symbol: str, futures=True, tf="5m"):
    try:
        df = fetch_klines(symbol, tf, limit=210, futures=futures)
        if len(df) < 50:
            return None
        c = df["close"].astype(float)
        r = float(rsi(c).iloc[-1])
        m_line, m_sig, _ = macd(c)
        ml, ms = float(m_line.iloc[-1]), float(m_sig.iloc[-1])
        trend = "Boğa" if ema(c,50).iloc[-1] > c.rolling(200).mean().iloc[-1] else "Ayı"
        score = 0
        direction, comment = "Bekle", "Teyit beklenmeli"
        if r < 30: score += 40; direction="Long"; comment="RSI düşük"
        elif r > 70: score += 40; direction="Short"; comment="RSI yüksek"
        elif ml > ms: score += 25; direction="Long"; comment="MACD ↑"
        elif ml < ms: score += 25; direction="Short"; comment="MACD ↓"
        elif trend == "Boğa": score += 20; direction="Long"; comment="Trend Boğa"
        else: score += 20; direction="Short"; comment="Trend Ayı"
        return {
            "sym": symbol,
            "price": float(c.iloc[-1]),
            "rsi": round(r,2),
            "macd": round(ml,4),
            "macd_sig": round(ms,4),
            "trend": trend,
            "score": int(score),
            "dir": direction,
            "comment": comment,
            "lev":"5x","sl":"3%","tp":"15%","tf":tf,"style":"Swing"
        }
    except Exception:
        return None

def compute_scalp(symbol: str):
    try:
        df = fetch_klines(symbol, "1m", limit=240, futures=True)
        if len(df) < 50:
            return None
        c = df["close"].astype(float)
        e9, e21 = ema(c,9), ema(c,21)
        direction, comment, score = "Bekle", "Sıkışma takip", 0
        if e9.iloc[-1] > e21.iloc[-1] and c.iloc[-1] > e9.iloc[-1]:
            direction, comment, score = "Long", "EMA9>21 + momentum", 35
        elif e9.iloc[-1] < e21.iloc[-1] and c.iloc[-1] < e9.iloc[-1]:
            direction, comment, score = "Short","EMA9<21 + momentum", 35
        return {
            "sym": symbol, "price": float(c.iloc[-1]),
            "rsi": round(float(rsi(c).iloc[-1]),2),
            "macd": 0.0, "macd_sig":0.0, "trend":"—",
            "score": score, "dir": direction, "comment": comment,
            "lev":"10x","sl":"0.7%","tp":"1.5%","tf":"1m","style":"Scalp"
        }
    except Exception:
        return None

# ========== TAB: Prices ==========
with tabs[2]:
    st.subheader("Anlık Fiyatlar")
    prices = read_prices()
    # Binance tarzı ok/renk
    dfp = pd.DataFrame(prices)
    dfp = dfp.sort_values("symbol").head(limit_rows)
    def fmt_row(row):
        arrow = "▲" if row["pct"] >= 0 else "▼"
        color = "#22c55e" if row["pct"] >= 0 else "#ef4444"
        return f"<b>{row['symbol']}</b>", f"{row['last']:.6f}", f"<span style='color:{color}'>{arrow} {row['pct']:.2f}%</span>", row["src"]
    if not dfp.empty:
        rows = [fmt_row(r) for _,r in dfp.iterrows()]
        st.markdown("""
        <style>
        table.prices td { padding:6px 10px; border-bottom:1px solid #222; }
        </style>
        """, unsafe_allow_html=True)
        html = "<table class='prices'><tr><th>Sembol</th><th>Fiyat</th><th>% 24h</th><th>Kaynak</th></tr>"
        for a,b,c,d in rows: html += f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("Fiyat bulunamadı.")

# ========== TAB: All coins – Signals ==========
def _render_signal_table(items, title):
    st.subheader(title)
    if not items:
        st.info("Sinyal bulunamadı (ağ kısıtı olabilir).")
        return
    cols = ["Yön","Tarz","TF","Sembol","Fiyat","Skor","RSI","MACD","Trend","Yorum","Lev","SL","TP"]
    rows = []
    for it in items:
        direction = it["dir"]
        arrow = "▲ LONG" if direction=="Long" else ("▼ SHORT" if direction=="Short" else "— BEKLE")
        rows.append([
            arrow, it["style"], it["tf"], it["sym"], f"{it['price']:.6f}", it["score"], it["rsi"],
            it["macd"], it["trend"], it["comment"], it["lev"], it["sl"], it["tp"]
        ])
    df = pd.DataFrame(rows, columns=cols)
    st.dataframe(df.head(limit_rows), use_container_width=True)

with tabs[0]:
    st.subheader("Tüm USDT pariteleri (futures dene → olmazsa skip)")
    # “büyük liste” yerine kafa coinleri + ilk 100 sembol (fiyatlardan)
    base = [p["symbol"] for p in prices if p["symbol"].endswith("/USDT")]
    base = list(dict.fromkeys(base))  # unique
    base = KAFA_COINLER + [s.split("/")[0] for s in base][:150]
    syms = [f"{s}/USDT" for s in base]
    items = []
    for sym in syms:
        it = compute_signal(sym, futures=True, tf="5m")
        if it: items.append(it)
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    _render_signal_table(items, "Tüm Coinler – Sinyal")

# ========== TAB: Kafa Coinler ==========
with tabs[1]:
    items = []
    for s in KAFA_COINLER:
        it = compute_signal(f"{s}/USDT", futures=True, tf="5m")
        if it: items.append(it)
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    _render_signal_table(items, "Kafa Coinler – (Binance futures)")

# ========== TAB: Scalp ==========
with tabs[3]:
    items = []
    for s in KAFA_COINLER[:60]:
        it = compute_scalp(f"{s}/USDT")
        if it: items.append(it)
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    _render_signal_table(items, "VUR-KAÇ (Scalp) – 1m")

# ========== TAB: Fast Futures ==========
with tabs[4]:
    items = []
    for s in KAFA_COINLER[:60]:
        it = compute_scalp(f"{s}/USDT")  # hızlı tetik için aynı kuralları kullandık
        if it: items.append(it)
    items = sorted(items, key=lambda x: x["score"], reverse=True)[:60]
    _render_signal_table(items, "Vadeli Hızlı – hızlı momentum tetikleri")

# ========== TAB: MEXC 0 Fee ==========
if enable_mexc_tab:
    with tabs[5]:
        st.subheader("MEXC 0 Fee (sade fiyat/sinyal görünümü – Binance engelinde fallback)")
        items = []
        for s in MEXC_ZERO:
            # MEXC verisini doğrudan REST’ten almak için hızlı yaklaşım: CoinGecko fallback
            # (MEXC public OHLC endpoint’leri bazı hostlarda CORS/koruma yapabiliyor)
            try:
                # önce Binance denenir (aynı sembol varsa)
                it = compute_signal(f"{s}/USDT", futures=True, tf="5m")
            except Exception:
                it = None
            if not it:
                # tamamen boş kalmasın diye fiyatı CG’den çekip minimal kart basalım
                cg = fetch_coingecko_markets(per_page=250, page=1)
                row = next((x for x in cg if x["symbol"].upper()==s), None)
                if row:
                    it = {"sym":f"{s}/USDT","price":float(row["current_price"] or 0),"rsi":0,"macd":0,"macd_sig":0,
                          "trend":"—","score":0,"dir":"Bekle","comment":"(fallback) fiyat","lev":"—","sl":"—","tp":"—",
                          "tf":"—","style":"—"}
            if it: items.append(it)
        _render_signal_table(items, "MEXC 0 Fee listesi")

# ========== TAB: Top Gainers / Losers ==========
with tabs[6 if enable_mexc_tab else 5]:
    st.subheader("Top Gainers / Losers (24h)")
    dfp = pd.DataFrame(prices)
    if not dfp.empty:
        dfp = dfp.sort_values("pct", ascending=False)
        left, right = st.columns(2)
        with left:
            st.write("**Gainers**")
            st.dataframe(dfp.head(10)[["symbol","last","pct","src"]], use_container_width=True)
        with right:
            st.write("**Losers**")
            st.dataframe(dfp.tail(10)[["symbol","last","pct","src"]], use_container_width=True)
    else:
        st.info("Veri yok.")

# ========== TAB: News (TR) ==========
@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def fetch_news_tr():
    feeds = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss',
        'https://decrypt.co/feed'
    ]
    out = []
    for url in feeds:
        try:
            xml = _try_get_text(url, timeout=6)
            soup = BeautifulSoup(xml, 'xml')
            for it in soup.select('item')[:20]:
                title = it.title.text if it.title else ''
                title_tr = translate_tr(title)
                link = it.link.text if it.link else ''
                pub = it.pubDate.text if it.pubDate else ''
                src = url.split('//')[1].split('/')[0]
                out.append({"Zaman":pub,"Kaynak":src,"Başlık":title_tr,"Link":link})
        except Exception:
            continue
    return pd.DataFrame(out)

with tabs[7 if enable_mexc_tab else 6]:
    st.subheader("Haberler (Türkçe)")
    dfnews = fetch_news_tr()
    if not dfnews.empty:
        st.dataframe(dfnews[["Zaman","Kaynak","Başlık"]], use_container_width=True)
        st.caption("Satır üzerine gelince tam başlığı görebilirsin. (Link sütunu gizli)")
    else:
        st.info("Haber bulunamadı.")

# ========== TAB: Kasa Yönetimi ==========
with tabs[8 if enable_mexc_tab else 7]:
    st.subheader("Kasa Yönetimi (öneri/kağıt-trade) – güvenlik için emir göndermez")
    bal = st.number_input("Mevcut kasa (USDT)", min_value=50.0, value=500.0, step=50.0)
    risk_pct = st.slider("İşlem başı risk (%)", 0.1, 3.0, 1.0, 0.1)
    take_n = st.slider("Aynı anda max pozisyon adedi", 1, 8, 3, 1)
    st.caption("Sistem, en yüksek skorlu sinyallerden başlayarak notional/margin hesaplar; emir YOLLAMAZ, sadece öneri üretir.")

    # kaynak sinyaller (önce hızlı olanlar)
    pool = []
    for s in KAFA_COINLER:
        it = compute_scalp(f"{s}/USDT")
        if it: pool.append(it)
    # swing ekle
    for s in KAFA_COINLER:
        it = compute_signal(f"{s}/USDT", futures=True, tf="5m")
        if it: pool.append(it)

    pool = sorted(pool, key=lambda x: x["score"], reverse=True)[:take_n]
    rows = []
    for it in pool:
        # basit notional: risk_pct * balance * kaldıraç
        lev = 10 if it["style"]=="Scalp" else 5
        notional = bal * (risk_pct/100.0) * lev
        price = it["price"]
        qty = (notional/lev) / max(price, 1e-9)
        rows.append({
            "Sembol": it["sym"],
            "Yön": it["dir"],
            "Tarz": it["style"],
            "TF": it["tf"],
            "Skor": it["score"],
            "Fiyat": f"{price:.6f}",
            "Risk%": f"{risk_pct:.2f}%",
            "SL": it["sl"],
            "TP": it["tp"],
            "Lev": lev,
            "Notional $": round(notional,2),
            "Margin $": round(notional/lev,2),
            "Adet": round(qty,6),
            "Not": it["comment"]
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Uygun sinyal bulunamadı (ağ/erişim kısıtı olabilir).")
