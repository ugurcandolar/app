# Web Panel (Streamlit) — Crypto Predator (BINANCE + MEXC)
# --------------------------------------------------------
# Çalıştırma:
#   pip install streamlit ccxt pandas numpy requests beautifulsoup4 matplotlib
#   streamlit run app.py
#
# Notlar:
# - Sadece public uçlar kullanılır (API key gerekmiyor).
# - Otomatik yenileme Streamlit Cloud uyumlu hale getirildi (st.autorefresh).
# - Vadeli semboller ccxt'de 'BTC/USDT:USDT' formatındadır.

import time, math, json, io
from datetime import datetime
from typing import List, Dict, Tuple

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import ccxt

# ------------------ Config ------------------
APP_TITLE = "Crypto Predator – Web"
REFRESH_MS = 15_000
SIGNAL_INTERVAL_SEC = 60
SCALP_INTERVAL_SEC = 20
FUTURES_FAST_SEC = 5
TOPN = 10

ZERO_FEE_FUTURES = [
    'BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','AVAX/USDT:USDT','SEI/USDT:USDT',
    'MAGIC/USDT:USDT','BNB/USDT:USDT','XRP/USDT:USDT','ADA/USDT:USDT','DOGE/USDT:USDT','SUI/USDT:USDT'
]
MEXC_0_FEE = [
    'SOL/USDT','SUI/USDT','ADA/USDT','PEPE/USDT','PUMP/USDT','PENGU/USDT','LTC/USDT',
    'ONDO/USDT','HYPE/USDT','LDO/USDT','AAVE/USDT','XLM/USDT','POPCAT/USDT','ETHFI/USDT',
    'APT/USDT','TONU/USDT','SEI/USDT','WLD/USDT','TAO/USDT','NEAR/USDT','SHIB/USDT'
]
RSS_FEEDS = [
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://cointelegraph.com/rss',
    'https://decrypt.co/feed'
]

DEFAULT_LEVERAGE = '10x'
DEFAULT_SL = '0.6%'
DEFAULT_TP = '1.2%'

# ------------------ Helpers ------------------
@st.cache_data(ttl=120)
def translate_tr(text: str) -> str:
    text = (text or '').strip()
    if not text:
        return text
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client":"gtx","sl":"auto","tl":"tr","dt":"t","q": text[:5000]}
        j = requests.get(url, params=params, timeout=4).json()
        return "".join(part[0] for part in j[0] if part and part[0])
    except Exception:
        return text

@st.cache_data(ttl=300)
def load_exchanges():
    spot = ccxt.binance({'enableRateLimit': True})
    fut = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    spot.load_markets(); fut.load_markets()
    mexc = ccxt.mexc({'enableRateLimit': True}); mexc.load_markets()
    return spot, fut, mexc

@st.cache_data(ttl=30)
def fetch_tickers(use_futures=True) -> Dict:
    spot, fut, _ = load_exchanges()
    ex = fut if use_futures else spot
    return ex.fetch_tickers()

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol: str, timeframe: str = '5m', limit: int = 210, use_futures: bool = False):
    spot, fut, _ = load_exchanges()
    ex = fut if use_futures else spot
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

@st.cache_data(ttl=60)
def fetch_ohlcv_mexc(symbol: str, timeframe: str = '5m', limit: int = 210):
    _, _, mexc = load_exchanges()
    return mexc.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# TA

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(window).mean()
    down = -d.clip(upper=0).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s: pd.Series):
    e12 = ema(s, 12); e26 = ema(s, 26)
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Signal engine

def compute_swing(symbol: str, use_futures: bool) -> dict | None:
    try:
        o = fetch_ohlcv(symbol, '5m', 210, use_futures)
        if not o or len(o) < 60:
            return None
        df = pd.DataFrame(o, columns=['ts','o','h','l','c','v']).astype(float)
        c = df['c']; h = df['h']; l = df['l']
        r = rsi(c).iloc[-1]
        m_line, m_sig, _ = macd(c)
        ema50 = ema(c, 50).iloc[-1]
        ma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else c.mean()
        trend = 'Boğa' if ema50 > ma200 else ('Ayı' if ema50 < ma200 else 'Nötr')
        price = float(c.iloc[-1])
        # heuristic score
        score = 0
        direction, comment = 'Bekle', 'Teyit beklenmeli'
        if r < 30: score+=40; direction='Long'; comment='RSI düşük'
        elif r > 70: score+=40; direction='Short'; comment='RSI yüksek'
        elif m_line.iloc[-1] > m_sig.iloc[-1]: score+=25; direction='Long'; comment='MACD yukarı'
        elif m_line.iloc[-1] < m_sig.iloc[-1]: score+=25; direction='Short'; comment='MACD aşağı'
        elif trend == 'Boğa': score+=20; direction='Long'; comment='Trend boğa'
        elif trend == 'Ayı': score+=20; direction='Short'; comment='Trend ayı'
        else: score+=10; direction='Bekle'; comment='Yönsüz'
        return {
            'Yön': '▲ LONG' if direction=='Long' else ('▼ SHORT' if direction=='Short' else '— BEKLE'),
            'Tarz': 'Swing', 'TF':'5m', 'Sembol':symbol, 'Fiyat': price, 'Skor': int(score),
            'RSI': round(float(r),2), 'MACD': round(float(m_line.iloc[-1]),4), 'Trend': trend,
            'Yorum': comment, 'Lev': DEFAULT_LEVERAGE, 'SL': '3%', 'TP': '15%', 'Not':''
        }
    except Exception:
        return None

def compute_scalp(symbol: str) -> dict | None:
    try:
        o = fetch_ohlcv(symbol, '1m', 200, True)
        if not o or len(o) < 60:
            return None
        df = pd.DataFrame(o, columns=['ts','o','h','l','c','v']).astype(float)
        c = df['c']; h = df['h']; l = df['l']
        e9 = ema(c,9).iloc[-1]; e21 = ema(c,21).iloc[-1]
        m_line, m_sig, _ = macd(c)
        r = rsi(c).iloc[-1]
        a = atr(h,l,c,14); atrp = (a/c).iloc[-1]*100
        direction, comment, score = 'Bekle','Sıkışma',0
        if e9>e21 and m_line.iloc[-1]>m_sig.iloc[-1] and r>52 and atrp>0.08:
            direction='Long'; comment='EMA9>21 + momentum'; score=55
        elif e9<e21 and m_line.iloc[-1]<m_sig.iloc[-1] and r<48 and atrp>0.08:
            direction='Short'; comment='EMA9<21 + momentum'; score=55
        else:
            return None
        return {
            'Yön': '▲ LONG' if direction=='Long' else '▼ SHORT', 'Tarz':'Scalp','TF':'1m','Sembol':symbol,
            'Fiyat': float(c.iloc[-1]), 'Skor': score, 'RSI': round(float(r),2), 'MACD': round(float(m_line.iloc[-1]),4),
            'Trend':'—','Yorum':comment,'Lev':'10x','SL':'0.6%','TP':'1.2%','Not':''
        }
    except Exception:
        return None

# Styling helpers

def style_direction(val: str):
    if isinstance(val, str) and 'LONG' in val:
        return 'color: #22c55e; font-weight:600'
    if isinstance(val, str) and 'SHORT' in val:
        return 'color: #ef4444; font-weight:600'
    return 'color: #94a3b8'

# -------------- UI --------------
st.set_page_config(page_title=APP_TITLE, layout='wide')
st.title(APP_TITLE)

# Auto-refresh (Streamlit Cloud uyumlu)
try:
    st.autorefresh(interval=REFRESH_MS, key='auto')
except Exception:
    pass

# Sidebar controls
with st.sidebar:
    st.subheader("Ayarlar")
    show_mexc = st.checkbox("MEXC 0 Fee sekmesi", value=True)
    max_rows = st.slider("Tablo satır limiti", 20, 400, 100, 10)
    if st.button("🔄 Yenile"):
        try:
            st.rerun()
        except Exception:
            pass
    st.caption("Veriler ccxt ile çekilir. Hızlı yenilemede borsa rate‑limit uygulayabilir.")

# Tabs
T1, T2, T3, T4, T5, T6, T7, T8, T9 = st.tabs([
    '📊 Tüm Coinler – Sinyal', '🧠 Kafa Coinler', '💹 Anlık Fiyatlar',
    '⚡ VUR-KAÇ (Scalp)', '🚀 Vadeli Hızlı', '🟧 MEXC 0 Fee',
    '📈 Top Gainers/Losers', '📰 Haberler', '💼 Kasa Yönetimi (öneri)'
])

# --- Live Prices ---
with T3:
    st.subheader('Anlık Fiyatlar (Futures)')
    t = fetch_tickers(True)
    rows = []
    for sym, info in t.items():
        try:
            if not (sym.endswith('/USDT') or sym.endswith(':USDT')): continue
            price = info.get('last') or info.get('close')
            pct = info.get('percentage')
            rows.append([sym, price, pct, 'BINANCE FUTURES'])
        except: pass
    dfp = pd.DataFrame(rows, columns=['Sembol','Fiyat','%24h','Kaynak']).sort_values('Sembol')
    def _fmt_pct(v):
        try:
            v = float(v); arrow = '▲' if v>=0 else '▼'; return f"{arrow} {v:.2f}%"
        except:
            return v
    dfp['%24h'] = dfp['%24h'].apply(_fmt_pct)
    st.dataframe(dfp.head(max_rows), use_container_width=True)

# --- All Coins Signals ---
with T1:
    st.subheader('Tüm Coinler – Sinyal (Spot USDT)')
    spot_syms = [s for s in load_exchanges()[0].symbols if s.endswith('/USDT')][:300]
    out = []
    for s in spot_syms:
        r = compute_swing(s, use_futures=False)
        if r: out.append(r)
    dfa = pd.DataFrame(out).sort_values('Skor', ascending=False).head(max_rows)
    if not dfa.empty:
        st.dataframe(dfa.style.applymap(style_direction, subset=['Yön']), use_container_width=True)
    else:
        st.info('Veri bulunamadı veya rate-limit. Tekrar deneyin.')

# --- Kafa Coinler (Futures) ---
with T2:
    st.subheader('Kafa Coinler – Futures 5m Swing')
    out = []
    for s in ZERO_FEE_FUTURES:
        r = compute_swing(s, use_futures=True)
        if r: out.append(r)
    dfz = pd.DataFrame(out).sort_values('Skor', ascending=False).head(max_rows)
    if not dfz.empty:
        st.dataframe(dfz.style.applymap(style_direction, subset=['Yön']), use_container_width=True)
    else:
        st.warning('Şu anda sinyal yok.')

# --- Scalp (1m Futures) ---
with T4:
    st.subheader('Vur-Kaç (Scalp) – 1m Futures')
    fut_syms = [s for s in load_exchanges()[1].symbols if s.endswith(':USDT')][:200]
    out = []
    for s in fut_syms:
        r = compute_scalp(s)
        if r: out.append(r)
    dfs = pd.DataFrame(out).sort_values('Skor', ascending=False).head(max_rows)
    if not dfs.empty:
        st.dataframe(dfs.style.applymap(style_direction, subset=['Yön']), use_container_width=True)
    else:
        st.info('Şu an scalp tetik yok.')

# --- Futures Quick (fast heuristics) ---
with T5:
    st.subheader('Vadeli Hızlı – 1m Momentum')
    out = []
    for s in ZERO_FEE_FUTURES:
        r = compute_scalp(s)
        if r: out.append(r)
    dfq = pd.DataFrame(out).sort_values('Skor', ascending=False).head(min(30, max_rows))
    if not dfq.empty:
        st.dataframe(dfq.style.applymap(style_direction, subset=['Yön']), use_container_width=True)
    else:
        st.info('Hızlı tetik yok.')

# --- MEXC 0 Fee (Spot) ---
with T6:
    if show_mexc:
        st.subheader('MEXC 0 Fee – Spot 5m Swing')
        out = []
        for s in MEXC_0_FEE:
            try:
                o = fetch_ohlcv_mexc(s, '5m', 210)
                if not o or len(o) < 60: 
                    continue
                df = pd.DataFrame(o, columns=['ts','o','h','l','c','v']).astype(float)
                c=df['c']; h=df['h']; l=df['l']
                r = rsi(c).iloc[-1]
                m_line, m_sig, _ = macd(c)
                ema50 = ema(c,50).iloc[-1]
                ma200 = c.rolling(200).mean().iloc[-1] if len(c)>=200 else c.mean()
                trend = 'Boğa' if ema50>ma200 else ('Ayı' if ema50<ma200 else 'Nötr')
                price = float(c.iloc[-1])
                score=0; direction='Bekle'; comment='Teyit beklenmeli'
                if r<30: score+=40; direction='Long'; comment='RSI düşük'
                elif r>70: score+=40; direction='Short'; comment='RSI yüksek'
                elif m_line.iloc[-1]>m_sig.iloc[-1]: score+=25; direction='Long'; comment='MACD yukarı'
                elif m_line.iloc[-1]<m_sig.iloc[-1]: score+=25; direction='Short'; comment='MACD aşağı'
                elif trend=='Boğa': score+=20; direction='Long'; comment='Trend boğa'
                elif trend=='Ayı': score+=20; direction='Short'; comment='Trend ayı'
                else: score+=10
                out.append({'Yön':'▲ LONG' if direction=='Long' else ('▼ SHORT' if direction=='Short' else '— BEKLE'),
                            'Tarz':'Swing','TF':'5m','Sembol':s,'Fiyat':price,'Skor':score,'RSI':round(float(r),2),
                            'MACD':round(float(m_line.iloc[-1]),4),'Trend':trend,'Yorum':comment,'Lev':DEFAULT_LEVERAGE,
                            'SL':'3%','TP':'15%','Not':''})
            except Exception:
                continue
        dfm = pd.DataFrame(out).sort_values('Skor', ascending=False).head(max_rows)
        if not dfm.empty:
            st.dataframe(dfm.style.applymap(style_direction, subset=['Yön']), use_container_width=True)
        else:
            st.info('MEXC verisi şu an boş veya rate-limit.')
    else:
        st.info('Sol menüden MEXC sekmesini açabilirsin.')

# --- Top gainers / losers + mini sparklines ---
with T7:
    st.subheader('Top Gainers / Losers (Futures)')
    t = fetch_tickers(True)
    arr = []
    for sym, info in t.items():
        try:
            if not (sym.endswith('/USDT') or sym.endswith(':USDT')): continue
            pct = info.get('percentage'); last = info.get('last') or info.get('close')
            if pct is None or last is None: continue
            arr.append((sym, float(pct), float(last)))
        except: pass
    arr.sort(key=lambda x:x[1], reverse=True)
    gain = arr[:TOPN]
    arr.sort(key=lambda x:x[1])
    loss = arr[:TOPN]

    def plot_spark(symbol: str) -> io.BytesIO:
        try:
            o = fetch_ohlcv(symbol, '5m', 60, True)
            c = [x[4] for x in o[-60:]] if o else []
        except Exception:
            c = []
        fig, ax = plt.subplots(figsize=(3,0.7))
        ax.plot(c)
        ax.set_axis_off()
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0); plt.close(fig)
        buf.seek(0); return buf

    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown('**Gainers**')
        for s, p, last in gain:
            colA, colB, colC = st.columns([3,1,2])
            colA.write(f"**{s}**  ")
            colB.write(f"**{p:.2f}%**")
            colC.image(plot_spark(s))
    with cg2:
        st.markdown('**Losers**')
        for s, p, last in loss:
            colA, colB, colC = st.columns([3,1,2])
            colA.write(f"**{s}**  ")
            colB.write(f"**{p:.2f}%**")
            colC.image(plot_spark(s))

# --- News (TR) ---
with T8:
    st.subheader('Haberler (Türkçe başlık)')
    items = []
    for url in RSS_FEEDS:
        try:
            xml = requests.get(url, timeout=6, headers={'User-Agent':'Mozilla/5.0'}).text
            soup = BeautifulSoup(xml, 'xml')
            for it in soup.select('item')[:15]:
                title = (it.title.text if it.title else '').strip()
                title_tr = translate_tr(title)
                link = (it.link.text if it.link else '').strip()
                pub  = (it.pubDate.text if it.pubDate else '')
                src  = url.split('//')[1].split('/')[0]
                items.append({'Zaman':pub,'Kaynak':src,'Başlık':title_tr,'Link':link})
        except Exception:
            continue
    dfn = pd.DataFrame(items)[:40]
    st.dataframe(dfn, use_container_width=True)
    st.caption('Satıra tıklayıp sağdaki Link hücresini kopyalayarak açabilirsin.')

# --- Kasa Yönetimi (öneri) basit sürüm ---
with T9:
    st.subheader('Kasa Yönetimi – Öneri Dağıtımı (Basit)')
    colA, colB, colC, colD = st.columns(4)
    cap = colA.number_input('Sermaye ($)', min_value=10.0, value=500.0, step=10.0)
    risk_pct = colB.number_input('Risk / İşlem (%)', min_value=0.1, value=1.0, step=0.1)
    lev = colC.number_input('Kaldıraç', min_value=1, max_value=125, value=10, step=1)
    max_pos = colD.number_input('Max Pozisyon', min_value=1, max_value=10, value=3, step=1)

    st.caption('Adaylar: Kafa Coinler + Scalp üstleri. SL/TP sabit varsayılmıştır (0.6% / 1.2%).')
    cand = []
    for s in ZERO_FEE_FUTURES:
        r = compute_scalp(s)
        if r: cand.append(r)
    cand = sorted(cand, key=lambda x:x['Skor'], reverse=True)[:max_pos]

    rows = []
    risk_dollars = cap * (risk_pct/100.0)
    remain = cap
    for c in cand:
        slp = 0.6
        notional = risk_dollars / (slp/100.0)
        margin = notional / max(1, lev)
        if margin > remain:
            notional = remain * lev
            margin = remain
        if margin <= 0: continue
        qty = notional / max(1e-9, float(c['Fiyat']))
        rows.append({
            'Sembol': c['Sembol'], 'Yön': c['Yön'], 'Tarz': c['Tarz'], 'TF': c['TF'], 'Skor': c['Skor'],
            'Fiyat': c['Fiyat'], 'Risk%': risk_pct, 'SL': '0.6%', 'TP': '1.2%', 'Lev': lev,
            'Notional $': notional, 'Margin $': margin, 'Adet': qty, 'Not': c['Yorum']
        })
        remain -= margin
        if remain <= 0: break
    dfl = pd.DataFrame(rows)
    if not dfl.empty:
        st.dataframe(dfl.style.applymap(style_direction, subset=['Yön']), use_container_width=True)
    else:
        st.info('Aday sinyal yok. Üst sekmelerde tetik oluşunca burada belirecek.')

st.caption(f"Son güncelleme: {datetime.utcnow().strftime('%H:%M:%S')} UTC | Otomatik yenileme: {REFRESH_MS/1000:.0f}s")
