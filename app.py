# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from datetime import datetime

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="BIST % Getiri Tahmini (T+1)", layout="wide")
st.title("BIST Hisse T+1 % Getiri Tahmini (Regresyon)")

st.markdown(
    """
Bu uygulama, seçtiğiniz hisse için **yarınki yüzde getiriyi** (T+1) tahmin etmeye çalışır.  
Veri: yfinance (günlük OHLCV, BIST100, USD/TRY).  
Model: LightGBM (regresyon), **walk-forward** yıllık değerlendirme.

> ⚠️ Bu bir teknik demodur; yatırım tavsiyesi değildir.
"""
)

DEFAULT_TICKERS = [
    "THYAO", "ASELS", "BIMAS", "EREGL", "KCHOL",
    "SAHOL", "SISE", "TCELL", "TUPRS", "TOASO",
    "YKBNK", "GARAN", "AKBNK", "ISCTR", "PETKM",
]

col1, col2, col3 = st.columns([2,1,1])
with col1:
    ticker = st.selectbox("Hisse seç (BIST):", options=DEFAULT_TICKERS, index=0)
with col2:
    start_year = st.number_input("Başlangıç yılı", min_value=2005, max_value=datetime.now().year-1, value=2015, step=1)
with col3:
    train_until_year = st.number_input("Walk-forward için minimum eğitim bitiş yılı", min_value=start_year+3, max_value=datetime.now().year-2, value=2020, step=1)

st.divider()

# ------------------------------
# Yardımcı Fonksiyonlar
# ------------------------------
@st.cache_data(show_spinner=True)
def load_data(tick: str, start: str = "2015-01-01"):
    """
    Seçilen hisse için OHLCV + BIST100 + USDTRY verilerini çeker ve tek DataFrame döner.
    """
    sym_stock = f"{tick}.IS"
    bench = "^XU100"
    usdtry = "USDTRY=X"

    px = yf.download([sym_stock, bench, usdtry], start=start, auto_adjust=True, progress=False)
    if px.empty:
        return pd.DataFrame()

    # Çok seviyeli kolonlar: ('Adj Close', 'THYAO.IS'), ('Volume','THYAO.IS')... gibi
    # Basitleştirelim
    def pick(colname):
        if colname in px.columns.get_level_values(0):
            return px[colname].rename_axis(index="Date")
        return pd.DataFrame()

    close = px["Close"].copy()
    if isinstance(close, pd.Series):  # Eğer tek enstrüman döndüyse
        close = close.to_frame(sym_stock)

    # Hisse Close + Benchmark + USDTRY
    df = pd.DataFrame(index=close.index)
    df["close"]    = close[sym_stock]
    df["bist"]     = close[bench]
    df["usdtry"]   = close[usdtry]

    # Volume (sadece hisse)
    if ("Volume" in px.columns.get_level_values(0)) and (sym_stock in px["Volume"].columns):
        df["volume"] = px["Volume"][sym_stock]
    else:
        df["volume"] = np.nan

    # Ek olarak OHLC (hisse)
    for f in ["Open","High","Low"]:
        if (f in px.columns.get_level_values(0)) and (sym_stock in px[f].columns):
            df[f.lower()] = px[f][sym_stock]
        else:
            df[f.lower()] = np.nan

    df = df.dropna(subset=["close", "bist", "usdtry"])
    return df

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd_diff(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def zscore(s: pd.Series, win: int = 60) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - m) / (sd + 1e-12)

def make_features(df0: pd.DataFrame) -> pd.DataFrame:
    """
    Özellik mühendisliği ve T+1 label oluşturma:
      - Feature'lar: yalnızca t gününe kadar olan bilgiler
      - Label: (t+1) yüzdesel getiri
    """
    df = df0.copy()
    df = df.sort_index()

    # Getiriler (yüzde)
    df["ret_1"]   = df["close"].pct_change(1) * 100
    df["ret_5"]   = df["close"].pct_change(5) * 100
    df["ret_10"]  = df["close"].pct_change(10) * 100
    df["ret_20"]  = df["close"].pct_change(20) * 100

    # Piyasa ve kur etkisi
    df["mkt_ret_1"]  = df["bist"].pct_change(1) * 100
    df["mkt_ret_5"]  = df["bist"].pct_change(5) * 100
    df["usd_ret_1"]  = df["usdtry"].pct_change(1) * 100
    df["usd_ret_5"]  = df["usdtry"].pct_change(5) * 100

    # Volatilite (20 günlük)
    df["vol_20"] = df["close"].pct_change().rolling(20).std() * 100

    # Trend / momentum
    df["ema_20"] = ema(df["close"], 20)
    df["ema_50"] = ema(df["close"], 50)
    df["ema_gap"] = (df["close"] / (df["ema_20"] + 1e-12) - 1.0) * 100
    df["rsi_14"] = rsi(df["close"], 14)
    df["macd_diff"] = macd_diff(df["close"])

    # Hacim z-skoru (log ile stabilize)
    if df["volume"].notna().sum() > 30:
        df["log_vol"] = np.log(df["volume"].replace(0, np.nan))
        df["vol_z"] = zscore(df["log_vol"].fillna(method="ffill"), 60)
    else:
        df["vol_z"] = np.nan

    # ATR benzeri volatilite (Close ile kaba yaklaşım, OHLC eksikse düşebilir)
    if df[["high","low","close"]].notna().all(axis=None):
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean() / (df["close"] + 1e-12) * 100
    else:
        df["atr_14"] = np.nan

    # Label: T+1 yüzdesel getiri
    df["target_ret_pct_t1"] = df["close"].pct_change(1).shift(-1) * 100

    # Temizlik
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[
        "ret_1","ret_5","ret_10","ret_20",
        "mkt_ret_1","mkt_ret_5","usd_ret_1","usd_ret_5",
        "vol_20","ema_20","ema_50","ema_gap","rsi_14","macd_diff",
        "target_ret_pct_t1"
    ])

    # Özellik listesi
    feature_cols = [
        "ret_1","ret_5","ret_10","ret_20",
        "mkt_ret_1","mkt_ret_5","usd_ret_1","usd_ret_5",
        "vol_20","ema_gap","rsi_14","macd_diff","vol_z","atr_14"
    ]

    # Nümerik doldurma (sadece özellik tarafı)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df, feature_cols

def walk_forward_train(df_feat: pd.DataFrame, feature_cols, min_train_end_year: int):
    """
    Yıllık walk-forward:
      - start_year..Y-1: train, Y: test
      - min_train_end_year: ilk test yılı için eğitim bitiş eşiği
    Döndürür: sonuçlar tablosu, yıllık test tahminleri (seri)
    """
    years = sorted(df_feat.index.year.unique())
    results = []
    preds_all = pd.Series(index=df_feat.index, dtype=float)

    for test_year in years:
        if test_year <= min_train_end_year:
            continue
        train_idx = df_feat.index.year < test_year
        test_idx  = df_feat.index.year == test_year

        if train_idx.sum() < 200 or test_idx.sum() < 20:
            continue

        X_tr = df_feat.loc[train_idx, feature_cols]
        y_tr = df_feat.loc[train_idx, "target_ret_pct_t1"]

        X_te = df_feat.loc[test_idx, feature_cols]
        y_te = df_feat.loc[test_idx, "target_ret_pct_t1"]

        model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        model.fit(X_tr, y_tr)

        y_hat = model.predict(X_te)
        preds_all.loc[X_te.index] = y_hat

        mae = mean_absolute_error(y_te, y_hat)
        r2  = r2_score(y_te, y_hat)
        # Yön isabeti (işaret)
        hit = (np.sign(y_te) == np.sign(y_hat)).mean()

        results.append({
            "test_year": int(test_year),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "MAE(%)": round(mae, 3),
            "R2": round(r2, 3),
            "Direction_Hit": round(hit, 3)
        })

    res_df = pd.DataFrame(results)
    return res_df, preds_all

# ------------------------------
# Veri Yükleme & Özellikler
# ------------------------------
df_raw = load_data(ticker, start=f"{start_year}-01-01")
if df_raw.empty:
    st.error("Veri çekilemedi. Ticker/başlangıç yılını değiştirip tekrar deneyin.")
    st.stop()

st.subheader("Ham Veri (örnek)")
st.dataframe(df_raw.tail(10))

df_feat, feature_cols = make_features(df_raw)

st.subheader("Özellikli Veri & Label (örnek)")
st.dataframe(df_feat[feature_cols + ["target_ret_pct_t1"]].tail(10))

# ------------------------------
# Walk-Forward Eğitim & Değerlendirme
# ------------------------------
st.subheader("Walk-Forward Sonuçları (Yıllık Test)")
res_df, preds_all = walk_forward_train(df_feat, feature_cols, min_train_end_year=train_until_year)

if res_df.empty:
    st.warning("Walk-forward sonuç üretilemedi. (Veri aralığı kısa olabilir.)")
else:
    st.dataframe(res_df)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Ortalama MAE (%)", f"{res_df['MAE(%)'].mean():.3f}")
    with colB:
        st.metric("Ortalama R²", f"{res_df['R2'].mean():.3f}")
    with colC:
        st.metric("Ort. Yön İsabeti", f"{res_df['Direction_Hit'].mean():.3f}")

# ------------------------------
# Son 30 Günde Out-of-Sample Karşılaştırma
# ------------------------------
st.subheader("Son 30 Günde (Walk-Forward Testte) Tahmin vs Gerçek")
cmp = pd.DataFrame(index=df_feat.index)
cmp["y_true_%"] = df_feat["target_ret_pct_t1"]
cmp["y_hat_%"]  = preds_all
cmp = cmp.dropna().tail(30)

if cmp.empty:
    st.info("Son 30 gün karşılaştırması için yeterli OOS tahmin bulunamadı.")
else:
    st.line_chart(cmp[["y_true_%","y_hat_%"]])
    st.dataframe(cmp)

# ------------------------------
# Bugün → Yarın Tahmini (En Güncel Modelle)
# ------------------------------
st.subheader("Bugün → Yarın % Getiri Tahmini")

# En güncel veriye kadar eğit, son günü tahmin için kullan
last_date = df_feat.index.max()
X_all = df_feat.loc[df_feat.index < last_date, feature_cols]
y_all = df_feat.loc[df_feat.index < last_date, "target_ret_pct_t1"]

if len(X_all) >= 200:
    final_model = LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    final_model.fit(X_all, y_all)

    X_today = df_feat.loc[[last_date], feature_cols]
    pred_next_pct = float(final_model.predict(X_today)[0])

    # Basit güven aralığı benzeri: geçmiş hataya göre ±MAE
    hist_mae = mean_absolute_error(y_all, final_model.predict(X_all))
    st.metric(label=f"{ticker}.IS için T+1 Tahmini (%):", value=f"{pred_next_pct:.2f} %")
    st.caption(f"Basit hata bandı (yaklaşık MAE): ±{hist_mae:.2f} puan")

    st.markdown(
        """
        **Notlar**
        - Tahmin, yalnızca **günlük kapanış sonrası** verilerle anlamlıdır.
        - Duyuru saatleri (KAP), temettü/split gibi olaylarda **look-ahead** hatalarından kaçınmak için veri damgası önemlidir.
        - Canlıda komisyon ve kaymayı hesaba kat.
        """
    )
else:
    st.info("Güvenilir bir final model için yeterli geçmiş veri yok (>=200 örnek önerilir).")

st.divider()
st.caption("© Demo – bilimsel/teknik amaçlıdır; yatırım tavsiyesi değildir.")
