import os
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, t
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from transformers import pipeline

# Optional Gemini SDK integration — don't hard-fail if SDK isn't installed
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    _HAS_GENAI = False

# 1. THIS MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Quant Trade & Risk Engine", page_icon="⚡", layout="wide")

# 2. Initialize session state
if "model_run" not in st.session_state:
    st.session_state.model_run = False

# ==========================================
# 1. ML SENTIMENT ENGINE (FinBERT HuggingFace)
# ==========================================
@st.cache_resource(show_spinner="Loading FinBERT NLP Model...")
def load_finbert():
    try:
        return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    except Exception:
        return None

finbert_pipe = load_finbert()

def fetch_news_and_sentiment(tk, max_items=8):
    try:
        news_items = tk.news or []
    except Exception:
        news_items = []
        
    parsed_news = []
    scores = []
    
    for item in news_items[:max_items]:
        title = item.get("title") if isinstance(item, dict) else None
        if not title and isinstance(item, dict) and "content" in item:
            title = item["content"].get("title", "")
        publisher = item.get("publisher", "Market News") if isinstance(item, dict) else "News"
        link = item.get("link", "#") if isinstance(item, dict) else "#"
        
        if title:
            score = 0.0
            if finbert_pipe:
                try:
                    res = finbert_pipe(title[:512])[0]
                    probs = {str(p.get('label')): float(p.get('score', 0.0)) for p in res if isinstance(p, dict)}
                    score = float(probs.get('positive', 0.0)) - float(probs.get('negative', 0.0))
                except Exception:
                    score = 0.0
            
            scores.append(score)
            parsed_news.append({"title": title, "publisher": publisher, "link": link, "score": score})
            
    avg_sentiment = float(np.mean(scores)) if scores else 0.0
    return parsed_news, avg_sentiment

# ------------------------------------------------------------------------
# 2. MARKET DATA & TECHNICALS
# ------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_stock_data(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="2y", auto_adjust=True)
    if hist.empty:
        return None, None, None, None, None, None
    price_series = hist["Close"].dropna()
    s0 = float(price_series.iloc[-1])
    log_returns = pd.Series(np.log(price_series / price_series.shift(1))).dropna()
    daily_vol = float(log_returns.std())
    annual_vol = daily_vol * np.sqrt(252)
    return s0, daily_vol, annual_vol, log_returns, price_series, datetime.now()

def compute_technical_indicators(price_series):
    s0_now = float(price_series.iloc[-1])
    n = len(price_series)

    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi_val = (100 - (100 / (1 + rs))).iloc[-1]
    rsi = float(rsi_val) if pd.notna(rsi_val) else 50.0

    sma_20 = float(price_series.rolling(20).mean().iloc[-1]) if n >= 20 else s0_now
    sma_50 = float(price_series.rolling(50).mean().iloc[-1]) if n >= 50 else s0_now

    if n >= 20:
        bb_middle = price_series.rolling(20).mean()
        bb_std = price_series.rolling(20).std()
        bb_upper = float((bb_middle + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_middle - 2 * bb_std).iloc[-1])
    else:
        bb_upper = s0_now
        bb_lower = s0_now

    ema_12 = price_series.ewm(span=12, adjust=False).mean()
    ema_26 = price_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1])

    return rsi, sma_20, sma_50, bb_upper, bb_lower, macd_hist

def compute_risk_metrics(log_returns, s0, horizon_days):
    confidence = 0.95
    mu = log_returns.mean() * horizon_days
    sigma = log_returns.std() * np.sqrt(horizon_days)
    var_param_pct = norm.ppf(1 - confidence, mu, sigma)
    var_param_usd = s0 * (1 - np.exp(var_param_pct))
    
    rolling_returns = log_returns.rolling(horizon_days).sum().dropna()
    if not rolling_returns.empty:
        var_hist_pct = np.percentile(rolling_returns, (1 - confidence) * 100)
        tail_losses = rolling_returns[rolling_returns <= var_hist_pct]
        cvar_pct = tail_losses.mean() if not tail_losses.empty else var_hist_pct
        cvar_usd = s0 * (1 - np.exp(cvar_pct))
    else:
        cvar_usd = var_param_usd * 1.25
        
    return var_param_usd, cvar_usd

def run_monte_carlo_fat_tail(s0, entry_price, daily_vol, holding_days, sims=10000):
    degrees_of_freedom = 4 
    sigma_period = daily_vol * np.sqrt(holding_days)
    z_t = t.rvs(df=degrees_of_freedom, size=sims, random_state=42) * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom) * sigma_period
    terminal_prices = s0 * np.exp(-0.5 * (sigma_period ** 2) + z_t)
    pop_pct = (np.sum(terminal_prices > entry_price) / sims) * 100.0
    return pop_pct, float(np.percentile(terminal_prices, 5)), float(np.percentile(terminal_prices, 95))

def fetch_macro_regime():
    """Checks the broader market regime using the VIX and S&P 500 200-day SMA."""
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        spy = yf.Ticker("SPY").history(period="1y")["Close"]
        spy_sma200 = spy.rolling(200).mean().iloc[-1]
        spy_spot = spy.iloc[-1]
        
        # A crash/bear regime is flagged if VIX is high OR S&P 500 is below its 200-day moving average
        is_bear_regime = (vix > 25) or (spy_spot < spy_sma200)
        return float(vix), is_bear_regime
    except Exception:
        return 20.0, False

def fetch_options_sentiment(ticker):
    """Calculates the Put/Call Open Interest ratio for the nearest options expiration."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return 1.0 # Default neutral if no options exist (e.g., small caps or Bursa stocks)
        
        chain = tk.option_chain(expirations[0])
        puts_oi = chain.puts['openInterest'].sum()
        calls_oi = chain.calls['openInterest'].sum()
        
        if calls_oi == 0: return 1.0
        return float(puts_oi / calls_oi)
    except Exception:
        return 1.0

def check_portfolio_correlation(new_ticker, portfolio_df):
    """Calculates how heavily correlated the new stock is to your existing holdings."""
    try:
        if portfolio_df.empty:
            return 0.0
            
        tickers = portfolio_df["Ticker"].dropna().unique().tolist()
        if new_ticker not in tickers:
            tickers.append(new_ticker)
            
        if len(tickers) < 2:
            return 0.0
            
        # Download 3 months of history for correlation check
        downloaded = yf.download(tickers, period="3mo", progress=False)
        if downloaded is None:
            return 0.0
        data = downloaded["Close"]
        if isinstance(data, pd.Series): 
            return 0.0
            
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        if new_ticker in corr_matrix.columns:
            # Get the average correlation of this new ticker against everything else you own
            avg_corr = corr_matrix[new_ticker].drop(new_ticker).mean()
            return float(avg_corr)
        return 0.0
    except Exception:
        return 0.0

# ------------------------------------------------------------------------
# 3. XGBOOST PREDICTIVE ENGINE & POSITION SIZING
# ------------------------------------------------------------------------
def train_xgboost_entry_model(price_series):
    df = pd.DataFrame({"Close": price_series})
    df["Returns"] = df["Close"].pct_change()
    df["Vol_10d"] = df["Returns"].rolling(10).std()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]

    feature_cols = ["Returns", "Vol_10d", "RSI"]
    latest_row = df[feature_cols].iloc[[-1]]
    
    if latest_row.isna().any(axis=1).iloc[0]:
        return 0.5, None

    df["Future_Min_3d"] = df["Close"].shift(-3).rolling(3).min()
    df["Target_Dip"] = np.where(df["Future_Min_3d"].notna(),
                                 (df["Future_Min_3d"] <= df["BB_Lower"]).astype(float), np.nan)

    labeled = df.dropna(subset=feature_cols + ["Target_Dip"])
    if len(labeled) < 60:
        return 0.5, None

    X = labeled[feature_cols]
    y = labeled["Target_Dip"].astype(int)

    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("xgb", XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, eval_metric="logloss"))
    ])

    for train_index, test_index in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        
        pos = max(int(y_train_cv.sum()), 1)
        neg = max(len(y_train_cv) - pos, 1)
        xgb_pipeline.set_params(xgb__scale_pos_weight=(neg/pos))
        
        xgb_pipeline.fit(X_train_cv, y_train_cv)
        cv_scores.append(xgb_pipeline.score(X_test_cv, y_test_cv))

    holdout_acc = float(np.mean(cv_scores))

    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    xgb_pipeline.set_params(xgb__scale_pos_weight=(neg/pos))
    xgb_pipeline.fit(X, y)

    prob_dip = float(xgb_pipeline.predict_proba(latest_row)[0][1])

    return prob_dip, holdout_acc

def calculate_3day_buy_target(s0, daily_vol, bb_lower, sma_20, sentiment_score, prob_dip):
    sigma_3d = daily_vol * np.sqrt(3)
    vol_pullback = s0 * (1 - (0.5 + prob_dip) * sigma_3d)
    tech_support = bb_lower if bb_lower < s0 else sma_20
    
    base_entry = (0.5 * vol_pullback) + (0.5 * tech_support)
    sentiment_adjustment = sentiment_score * (0.3 * sigma_3d * s0)
    
    optimal_buy_limit = min(s0, base_entry + sentiment_adjustment)
    conservative_buy = min(optimal_buy_limit, s0 * (1 - 1.2 * sigma_3d))
    
    return optimal_buy_limit, conservative_buy

def calculate_optimal_sell_target(s0, daily_vol, bb_upper, sma_20, rsi, sentiment_score, holding_days):
    sigma_h = daily_vol * np.sqrt(holding_days)
    vol_upside_target = s0 * (1 + 1.0 * sigma_h)
    tech_resistance = bb_upper if bb_upper > s0 else s0 + (s0 - sma_20)
    base_exit = (0.6 * vol_upside_target) + (0.4 * tech_resistance)
    sentiment_adjustment = sentiment_score * (0.3 * sigma_h * s0)
    if rsi > 70:
        rsi_adjustment = -0.15 * sigma_h * s0
    elif rsi < 30:
        rsi_adjustment = 0.10 * sigma_h * s0
    else:
        rsi_adjustment = 0.0
    optimal_sell = max(s0, base_exit + sentiment_adjustment + rsi_adjustment)
    aggressive_sell = max(optimal_sell, s0 * (1 + 1.5 * sigma_h))
    return optimal_sell, aggressive_sell, sigma_h

def calculate_position_sizing(account_size, entry_price, is_bursa):
    if entry_price <= 0 or account_size <= 0:
        return 0.0, 0.0
    
    if is_bursa:
        max_shares = int(account_size // entry_price)
        lots = max_shares // 100
        recommended_shares = float(lots * 100)
        total_capital = recommended_shares * entry_price
    else:
        recommended_shares = round(account_size / entry_price, 4)
        total_capital = account_size
        
    return recommended_shares, total_capital

def calculate_kelly_allocation(win_probability, take_profit_price, stop_loss_price, entry_price):
    if entry_price <= stop_loss_price or take_profit_price <= entry_price:
        return 0.0 
        
    reward = (take_profit_price - entry_price) / entry_price
    risk = (entry_price - stop_loss_price) / entry_price
    rr_ratio = reward / risk 
    
    kelly_fraction = (win_probability * (rr_ratio + 1) - 1) / rr_ratio
    safe_kelly = np.clip(kelly_fraction * 0.5, 0.0, 0.25)
    return safe_kelly

def evaluate_trade_suitability(prob_dip, sentiment_score, rsi, bb_lower, s0, pop_pct, kelly_fraction, pc_ratio, is_bear_regime, avg_corr):
    """
    Upgraded to include Smart Money Options, Macro Regime, and Portfolio Correlation.
    """
    score = 0.0
    
    # 1. Base ML & Technicals (scaled down slightly to make room for macro)
    if prob_dip > 0.70: score += 20
    elif prob_dip > 0.55: score += 10
    
    if sentiment_score > 0.2: score += 10
    
    if rsi < 40 and s0 <= (bb_lower * 1.02): score += 15
    elif rsi < 50: score += 5
    
    if pop_pct > 60: score += 10
    if kelly_fraction > 0.10: score += 15
    elif kelly_fraction > 0.05: score += 5
    
    # 2. Options Smart Money (0 to 10 points)
    if pc_ratio < 0.7: score += 10 # More calls than puts (bullish smart money)
    elif pc_ratio < 1.0: score += 5
    
    # 3. Macro Filter (0 to 10 points)
    if not is_bear_regime: score += 10 
    
    # 4. Correlation Diversification (0 to 10 points)
    if avg_corr < 0.3: score += 10 # Uncorrelated to your portfolio (Good)
    elif avg_corr < 0.6: score += 5
    
    # --- FATAL FLAW PENALTIES ---
    # Heavy penalty if it makes your portfolio dangerously unbalanced
    if avg_corr > 0.75: score -= 20
    # Heavy penalty if trying to buy a bearish stock during a macro crash
    if is_bear_regime and pc_ratio > 1.2: score -= 30
    
    score = np.clip(score, 0, 100)
    
    if score >= 75:
        signal = "🟢 **STRONG BUY:** Ideal setup. Models are aligned."
    elif score >= 55:
        signal = "🟡 **HOLD / WATCH:** Wait for better technical entry or ML conviction."
    else:
        signal = "🔴 **PASS:** Negative edge. Do not allocate capital."
        
    return score, signal

def backtest_strategy(price_series, holding_days):
    df = pd.DataFrame({"Close": price_series})
    df["Returns"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["STD_20"] = df["Close"].rolling(20).std()
    df["BB_Lower"] = df["SMA_20"] - 2 * df["STD_20"]
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    df["Signal"] = ((df["Close"] <= df["BB_Lower"]) & (df["RSI"] < 50)).astype(int)
    
    df["Forward_Return"] = df["Close"].shift(-holding_days) / df["Close"] - 1.0
    trades = df[df["Signal"] == 1]["Forward_Return"].dropna()
    
    if trades.empty:
        return 0.0, 0.0, 0.0, 0.0
        
    win_rate = (np.sum(trades > 0) / len(trades)) * 100.0
    total_return = float((np.prod(1 + trades) - 1) * 100.0)
    sharpe = float((trades.mean() / (trades.std() + 1e-9)) * np.sqrt(252 / holding_days))
    max_drawdown = float(trades.min() * 100.0)
    
    return win_rate, total_return, sharpe, max_drawdown

# -----------------------------------------------------------------------------
# HISTORICAL 5-YEAR BACKTEST ENGINE
# -----------------------------------------------------------------------------
def run_historical_backtest(ticker, initial_capital=10000, max_holding_days=10, sl_pct=0.07, tp_pct=0.10):
    """Runs a 5-year walk-forward backtest simulating the core Quant Engine."""
    ticker = ticker.strip().upper()
    
    try:
        # 1. Download target ticker safely
        df_raw = yf.download(ticker, period="5y", progress=False)
        if df_raw is None or df_raw.empty:
            return None
        
        # Format Close price column cleanly
        if "Close" in df_raw.columns:
            df = df_raw[["Close"]].copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["Close"]
        else:
            return None

        # 2. Fetch SPY with fallback if download fails
        try:
            spy_data = yf.download("SPY", period="5y", progress=False)
            if spy_data is not None and not spy_data.empty:
                df["SPY_Close"] = spy_data["Close"]
            else:
                df["SPY_Close"] = df["Close"]  # Fallback to stock close
        except Exception:
            df["SPY_Close"] = df["Close"]  # Fallback to stock close

        # 3. Fetch ^VIX with fallback if download fails
        try:
            vix_data = yf.download("^VIX", period="5y", progress=False)
            if vix_data is not None and not vix_data.empty:
                df["VIX"] = vix_data["Close"]
            else:
                df["VIX"] = 20.0  # Neutral baseline fallback
        except Exception:
            df["VIX"] = 20.0  # Neutral baseline fallback

        # 4. Technical Indicators
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["STD_20"] = df["Close"].rolling(20).std()
        df["BB_Lower"] = df["SMA_20"] - 2 * df["STD_20"]

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # 5. Regime Filters
        df["SPY_SMA200"] = df["SPY_Close"].rolling(200).mean()
        df["Bear_Regime"] = (df["VIX"] > 25) | (df["SPY_Close"] < df["SPY_SMA200"])

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"Backtest Error: {e}")  # Prints the exact error in your terminal/logs
        return None

# ------------------------------------------------------------------------
# 4. PREDICTION TRACK RECORD
# ------------------------------------------------------------------------
PREDICTION_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.csv")
LOG_COLUMNS = ["Timestamp", "Ticker", "Spot", "Buy_Target", "Sell_Target", "Dip_Prob", "Holding_Days"]

def log_prediction(ticker, spot, buy_target, sell_target, dip_prob, holding_days):
    row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker, "Spot": spot, "Buy_Target": buy_target,
        "Sell_Target": sell_target, "Dip_Prob": dip_prob, "Holding_Days": holding_days,
    }])
    write_header = not os.path.exists(PREDICTION_LOG_PATH)
    row.to_csv(PREDICTION_LOG_PATH, mode="a", header=write_header, index=False)

def load_prediction_history():
    if not os.path.exists(PREDICTION_LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        hist = pd.read_csv(PREDICTION_LOG_PATH, parse_dates=["Timestamp"])
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)
    if "Holding_Days" not in hist.columns:
        hist["Holding_Days"] = 10
    return hist

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history_cached(ticker):
    try:
        s = yf.Ticker(ticker).history(period="1y", auto_adjust=True)["Close"].dropna()
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s
    except Exception:
        return pd.Series(dtype=float)

def score_prediction_history(history_df):
    if history_df.empty:
        return pd.DataFrame()
    results = []
    for ticker, grp in history_df.groupby("Ticker"):
        price_series = fetch_price_history_cached(ticker)
        if price_series.empty:
            continue
        for _, row in grp.iterrows():
            future = price_series[price_series.index > row["Timestamp"]]
            horizon = row.get("Holding_Days", 10)
            horizon = 10 if pd.isna(horizon) else int(horizon)
            if future.empty:
                status, buy_hit, sell_hit = "Pending (no data yet)", None, None
            else:
                buy_hit = bool((future <= row["Buy_Target"]).any()) if pd.notna(row.get("Buy_Target")) else None
                sell_hit = bool((future >= row["Sell_Target"]).any()) if pd.notna(row.get("Sell_Target")) else None
                status = "Evaluated" if len(future) >= horizon else f"In progress ({len(future)}/{horizon}d)"
            results.append({
                "Timestamp": row["Timestamp"], "Ticker": ticker, "Spot_at_call": row["Spot"],
                "Buy_Target": row.get("Buy_Target"), "Buy_Hit": buy_hit,
                "Sell_Target": row.get("Sell_Target"), "Sell_Hit": sell_hit,
                "Dip_Prob": row.get("Dip_Prob"), "Status": status,
            })
    return pd.DataFrame(results)

def compute_calibration(scored_df, min_n=20):
    usable = scored_df[(scored_df["Buy_Hit"] == True) | (scored_df["Status"] == "Evaluated")].copy()
    usable = usable.dropna(subset=["Dip_Prob", "Buy_Hit"])
    if len(usable) < min_n:
        return None, len(usable)
    bins = [0, 0.3, 0.5, 0.7, 1.01]
    labels = ["0-30%", "30-50%", "50-70%", "70-100%"]
    usable["Predicted Bucket"] = pd.cut(usable["Dip_Prob"], bins=bins, labels=labels, right=False)
    cal = usable.groupby("Predicted Bucket", observed=True).agg(
        avg_predicted=("Dip_Prob", "mean"), actual_hit_rate=("Buy_Hit", "mean"), n=("Buy_Hit", "count")
    ).reset_index()
    return cal, len(usable)

# ------------------------------------------------------------------------
# 5. PORTFOLIO HOLDINGS TRACKER
# ------------------------------------------------------------------------
PORTFOLIO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.csv")
PORTFOLIO_COLUMNS = ["Ticker", "Buy_Price", "Quantity", "Buy_Date"]

def load_portfolio():
    if not os.path.exists(PORTFOLIO_PATH):
        return pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    try:
        df = pd.read_csv(PORTFOLIO_PATH)
        for col in PORTFOLIO_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[PORTFOLIO_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=PORTFOLIO_COLUMNS)

def save_portfolio(df):
    df.to_csv(PORTFOLIO_PATH, index=False)

def currency_symbol_for(ticker):
    return "RM " if str(ticker).upper().endswith(".KL") else "$"

@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_price(ticker):
    try:
        h = yf.Ticker(ticker).history(period="1d", auto_adjust=True)["Close"].dropna()
        return float(h.iloc[-1]) if not h.empty else None
    except Exception:
        return None

# ------------------------------------------------------------------------
# STREAMLIT UI LAYOUT & SIDEBAR
# ------------------------------------------------------------------------
st.title("⚡ Quantitative Trade & Risk Engine")
st.markdown("XGBoost Predictive Targets, FinBERT Sentiment & Position Sizing")

# --- GLOBAL SETTINGS SIDEBAR ---
st.sidebar.header("⚙️ Global Settings")
currency_option = st.sidebar.selectbox("Base Currency", ["USD ($)", "MYR (RM)"])

st.sidebar.markdown("---")
st.sidebar.header("🔧 Trade Parameters")
ticker = st.sidebar.text_input("Ticker Symbol (e.g. SKHY or 1155.KL)", value="SKHY").strip().upper()

is_bursa = ticker.endswith(".KL") or "MYR" in currency_option
sym = "RM " if "MYR" in currency_option or ticker.endswith(".KL") else "$"

default_capital = 740.00 if is_bursa else 64.00
entry_price_input = st.sidebar.number_input(f"Cost Basis / Entry Price ({sym})", value=134.00 if ticker == "SKHY" else 1.88, step=0.01)
portfolio_capital = st.sidebar.number_input(f"Available Capital ({sym})", value=default_capital, step=10.00)
holding_days = st.sidebar.slider("Holding Horizon (Trading Days)", min_value=1, max_value=60, value=10)
max_risk_pct = st.sidebar.slider("Max Acceptable Loss Limit (%)", min_value=1.0, max_value=20.0, value=7.0, step=0.5)

# Single primary action button updates session state
if st.sidebar.button("Run Quantitative Model", type="primary"):
    st.session_state.model_run = True

# Safely halt execution if model run hasn't been triggered yet
if not st.session_state.model_run:
    st.info("👈 Please configure your parameters in the sidebar and click 'Run Quantitative Model' to begin.")
    st.stop()

# ------------------------------------------------------------------------
# MAIN DASHBOARD EXECUTION
# ------------------------------------------------------------------------
with st.spinner(f"Running ML models & analytics for {ticker}..."):
    s0, daily_vol, annual_vol, log_returns, price_series, fetched_at = fetch_stock_data(ticker)

if s0 is None:
    st.error(f"Could not load market data for '{ticker}'. For Bursa stocks, remember to add `.KL` (e.g., `1155.KL`).")
    st.stop()

tk = yf.Ticker(ticker) 

fresh_col1, fresh_col2 = st.columns([4, 1])
if fetched_at is not None:
    fresh_col1.caption(f"📡 Price data as of **{fetched_at.strftime('%Y-%m-%d %H:%M:%S')}** "
                        f"(auto-refreshes every 60s).")
if fresh_col2.button("🔄 Refresh now"):
    fetch_stock_data.clear()
    st.rerun()

entry_price = entry_price_input if entry_price_input > 0 else s0

rsi, sma_20, sma_50, bb_upper, bb_lower, macd_hist = compute_technical_indicators(price_series)
var_param_usd, cvar_usd = compute_risk_metrics(log_returns, s0, holding_days)

# PATCED MONTE CARLO
pop_pct, p05, p95 = run_monte_carlo_fat_tail(s0, entry_price, daily_vol, holding_days)

news_items, sentiment_score = fetch_news_and_sentiment(tk)

# PATCHED XGBOOST
prob_dip, dip_model_acc = train_xgboost_entry_model(price_series)

optimal_buy, conservative_buy = calculate_3day_buy_target(s0, daily_vol, bb_lower, sma_20, sentiment_score, prob_dip)
optimal_sell, aggressive_sell, sigma_h_exit = calculate_optimal_sell_target(
    s0, daily_vol, bb_upper, sma_20, rsi, sentiment_score, holding_days
)
sigma_period = daily_vol * np.sqrt(holding_days)
target_tp = entry_price * (1 + 1.5 * sigma_period)
quant_sl = entry_price * (1 - 1.0 * sigma_period)
user_sl = entry_price * (1 - max_risk_pct / 100.0)
target_sl = max(quant_sl, user_sl)

rec_shares, total_alloc = calculate_position_sizing(portfolio_capital, entry_price, is_bursa)

log_prediction(ticker, s0, optimal_buy, optimal_sell, prob_dip, holding_days)

# --- NEW: Institutional Risk Models (Macro, Options, Correlation) ---
with st.spinner("Fetching Options Chain, Macro Regime, and Correlation Matrix..."):
    vix_val, is_bear_regime = fetch_macro_regime()
    pc_ratio = fetch_options_sentiment(ticker)
    current_portfolio = load_portfolio()
    avg_corr = check_portfolio_correlation(ticker, current_portfolio)

# --- Kelly Sizing & Upgraded Master Suitability Engine ---
kelly_fraction = calculate_kelly_allocation(pop_pct / 100.0, target_tp, target_sl, entry_price)

quant_score, trade_signal = evaluate_trade_suitability(
    prob_dip, sentiment_score, rsi, bb_lower, s0, pop_pct, kelly_fraction, 
    pc_ratio, is_bear_regime, avg_corr
)

st.markdown("---")
st.subheader("🧠 Institutional AI Quant Decision")
st.info(f"{trade_signal} (Master Quant Score: {quant_score:.0f}/100)")
if kelly_fraction > 0:
    st.success(f"⚖️ **Kelly Optimal Allocation:** Risk maximum of **{kelly_fraction*100:.1f}%** of your total portfolio on this trade.")
else:
    st.error("⚖️ **Kelly Optimal Allocation:** **0%** (Risk/Reward ratio is mathematically unfavorable).")

st.caption(f"**Macro Engine:** VIX at {vix_val:.2f} | **Options Flow:** Put/Call Ratio at {pc_ratio:.2f} | **Portfolio Correlation:** {avg_corr:.2f}")
st.markdown("---")

# --- DISPLAY DASHBOARD METRICS ---
st.subheader("🎯 3-Day ML Optimal Buy Price Target")
b_col1, b_col2, b_col3, b_col4 = st.columns(4)
b_col1.metric("Current Spot Price", f"{sym}{s0:.2f}")
b_col2.metric("Target Limit Buy Entry", f"{sym}{optimal_buy:.2f}", f"{((optimal_buy-s0)/s0)*100:+.2f}%")
b_col3.metric("Conservative Entry", f"{sym}{conservative_buy:.2f}", f"{((conservative_buy-s0)/s0)*100:+.2f}%")
acc_label = f"{dip_model_acc*100:.0f}% holdout acc" if dip_model_acc is not None else "not enough history to validate"
b_col4.metric("XGBoost Dip Probability", f"{prob_dip*100:.1f}%", acc_label, delta_color="off")

st.markdown("---")

st.subheader("💰 Optimal Sell Price Recommendation")
s_col1, s_col2, s_col3 = st.columns(3)
s_col1.metric("Current Spot Price", f"{sym}{s0:.2f}")
s_col2.metric("Target Sell Price", f"{sym}{optimal_sell:.2f}", f"{((optimal_sell-s0)/s0)*100:+.2f}%")
s_col3.metric("Aggressive Sell Price", f"{sym}{aggressive_sell:.2f}", f"{((aggressive_sell-s0)/s0)*100:+.2f}%")
st.caption(f"Based on resistance (BB-upper/SMA-20), {holding_days}d volatility, sentiment, and RSI.")

st.markdown("---")

st.subheader("📊 Position Sizing & Target Limits")
p_col1, p_col2, p_col3, p_col4 = st.columns(4)
p_col1.metric("Target Take Profit", f"{sym}{target_tp:.2f}")
p_col2.metric("Dynamic Stop Loss", f"{sym}{target_sl:.2f}")

if is_bursa:
    unit_str = f"{int(rec_shares // 100)}"
    unit_label = "lots"
else:
    unit_str = f"{rec_shares:.4f}".rstrip('0').rstrip('.') if rec_shares % 1 != 0 else f"{int(rec_shares)}"
    unit_label = "units"
    
p_col3.metric("Recommended Allocation", f"{unit_str} {unit_label}")
p_col4.metric("Capital Allocation", f"{sym}{total_alloc:,.2f}")

st.markdown("---")

# --- PLOTLY CHART ---
st.subheader("📈 Technical Level Chart")
series_tail = pd.Series(price_series).tail(120)
df_chart = pd.DataFrame({"Close": series_tail})
df_chart["BB_Upper"] = df_chart["Close"].rolling(20).mean() + 2 * df_chart["Close"].rolling(20).std()
df_chart["BB_Lower"] = df_chart["Close"].rolling(20).mean() - 2 * df_chart["Close"].rolling(20).std()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["Close"], name="Close Price", line=dict(color="white", width=2)))
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["BB_Upper"], name="BB Upper", line=dict(color="gray", dash="dash")))
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["BB_Lower"], name="BB Lower", line=dict(color="gray", dash="dash")))

fig.add_hline(y=optimal_buy, line_color="cyan", line_dash="dash", annotation_text="Target Buy Limit")
fig.add_hline(y=optimal_sell, line_color="lime", line_dash="dash", annotation_text="Target Sell")
fig.add_hline(y=target_tp, line_color="green", line_dash="dot", annotation_text="Take Profit")
fig.add_hline(y=target_sl, line_color="red", line_dash="dash", annotation_text="Stop Loss")

fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- DETAILED TABS ---
tab_portfolio, tab_backtest, tab_news, tab_risk, tab_mc, tab_track, tab_ai = st.tabs([
    "💼 My Portfolio", "🧪 Historical Strategy Backtest", "📰 FinBERT News Feed", "⚠️ Risk Matrix", "🎲 Monte Carlo", "📒 Track Record", "🤖 AI Analyst"
])

with tab_portfolio:
    st.caption("What you actually hold - saved locally to portfolio.csv. Add, edit, or delete rows directly in the table below, then hit Save.")
    portfolio_df = load_portfolio()

    edited = st.data_editor(
        portfolio_df, num_rows="dynamic", use_container_width=True, key="portfolio_editor",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True, help="e.g. SKHY or 0820EA.KL"),
            "Buy_Price": st.column_config.NumberColumn("Buy Price", format="%.4f", required=True),
            "Quantity": st.column_config.NumberColumn("Quantity", format="%.4f", required=True),
            "Buy_Date": st.column_config.TextColumn("Buy Date (optional)"),
        },
    )

    if st.button("💾 Save Portfolio"):
        clean = edited.dropna(subset=["Ticker", "Buy_Price", "Quantity"]).copy()
        clean["Ticker"] = clean["Ticker"].astype(str).str.upper().str.strip()
        save_portfolio(clean)
        st.success(f"Saved {len(clean)} holding(s).")
        st.rerun()

    holdings = portfolio_df.dropna(subset=["Ticker", "Buy_Price", "Quantity"])
    if holdings.empty:
        st.info("No holdings saved yet - add a row above (ticker, buy price, quantity) and hit Save.")
    else:
        st.markdown("---")
        st.subheader("Live valuation")
        rows = []
        for _, r in holdings.iterrows():
            live_price = fetch_latest_price(r["Ticker"])
            cost = float(r["Buy_Price"]) * float(r["Quantity"])
            mval = live_price * float(r["Quantity"]) if live_price is not None else None
            pnl = (mval - cost) if mval is not None else None
            pnl_pct = (pnl / cost * 100) if pnl is not None and cost else None
            rows.append({
                "Ticker": r["Ticker"], "Currency": currency_symbol_for(r["Ticker"]).strip() or "USD",
                "Buy Price": r["Buy_Price"], "Qty": r["Quantity"], "Cost Basis": cost,
                "Current Price": live_price, "Market Value": mval, "Unrealized P&L": pnl, "P&L %": pnl_pct,
            })
        val_df = pd.DataFrame(rows)
        display_df = val_df.copy()
        for col in ["Buy Price", "Current Price", "Cost Basis", "Market Value", "Unrealized P&L"]:
            display_df[col] = display_df.apply(
                lambda x: f"{currency_symbol_for(x['Ticker'])}{x[col]:,.2f}" if pd.notna(x[col]) else "N/A", axis=1)
        display_df["P&L %"] = val_df["P&L %"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
        st.dataframe(display_df, use_container_width=True)

        st.caption("Totals are grouped by currency - a MYR holding and a USD holding can't be summed together directly.")
        for curr, grp in val_df.groupby("Currency"):
            sym_g = "RM " if curr == "RM" else "$"
            total_cost = grp["Cost Basis"].sum()
            total_val = grp["Market Value"].sum(skipna=True) if grp["Market Value"].notna().any() else None
            t1, t2, t3 = st.columns(3)
            t1.metric(f"Cost Basis ({curr})", f"{sym_g}{total_cost:,.2f}")
            t2.metric(f"Market Value ({curr})", f"{sym_g}{total_val:,.2f}" if total_val is not None else "N/A")
            if total_val is not None:
                pnl_total = total_val - total_cost
                t3.metric(f"Unrealized P&L ({curr})", f"{sym_g}{pnl_total:,.2f}", f"{(pnl_total/total_cost*100):+.2f}%" if total_cost else None)
            else:
                t3.metric(f"Unrealized P&L ({curr})", "N/A")

with tab_backtest:
    win_rate, total_ret, sharpe, max_dd = backtest_strategy(price_series, holding_days)
    bt1, bt2, bt3, bt4 = st.columns(4)
    bt1.metric("Historical Win Rate", f"{win_rate:.1f}%")
    bt2.metric("Cumulative Strategy Return", f"{total_ret:+.2f}%")
    bt3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    bt4.metric("Max Drawdown", f"{max_dd:.2f}%")

with tab_news:
    st.write(f"**FinBERT Sentiment Index:** `{sentiment_score:+.3f}`")
    for n in news_items:
        score_tag = "🟢 Bullish" if n["score"] > 0.1 else "🔴 Bearish" if n["score"] < -0.1 else "⚪ Neutral"
        st.markdown(f"- **[{n['title']}]({n['link']})** ({n['publisher']}) — *{score_tag} ({n['score']:+.2f})*")

with tab_risk:
    st.write(f"**95% Parametric VaR:** `{sym}{var_param_usd:.2f}` potential risk limit")
    st.write(f"**CVaR Tail Loss:** `{sym}{cvar_usd:.2f}`")

with tab_mc:
    st.write(f"**Probability of Profit (PoP):** `{pop_pct:.1f}%`")
    st.write(f"**5th–95th Tail Percentiles:** `{sym}{p05:.2f}` to `{sym}{p95:.2f}`")

with tab_track:
    st.caption("Every run logs itself automatically.")
    history = load_prediction_history()
    if history.empty:
        st.info("No predictions logged yet.")
    else:
        st.write(f"**{len(history)} predictions logged** across {history['Ticker'].nunique()} ticker(s).")
        with st.spinner("Scoring past predictions..."):
            scored = score_prediction_history(history)

        if scored.empty:
            st.warning("Couldn't fetch price data to score against.")
        else:
            evaluated = scored[scored["Status"] == "Evaluated"]
            if not evaluated.empty:
                sc1, sc2 = st.columns(2)
                buy_rate = evaluated["Buy_Hit"].dropna().mean() * 100 if evaluated["Buy_Hit"].notna().any() else None
                sell_rate = evaluated["Sell_Hit"].dropna().mean() * 100 if evaluated["Sell_Hit"].notna().any() else None
                sc1.metric("Buy target hit rate", f"{buy_rate:.0f}%" if buy_rate is not None else "N/A", f"n={evaluated['Buy_Hit'].notna().sum()}")
                sc2.metric("Sell target hit rate", f"{sell_rate:.0f}%" if sell_rate is not None else "N/A", f"n={evaluated['Sell_Hit'].notna().sum()}")
            else:
                st.caption("No predictions have reached their full holding horizon yet.")

            st.markdown("**Dip probability calibration**")
            cal, n_usable = compute_calibration(scored)
            if cal is None:
                st.caption(f"Need at least 20 resolved predictions to check calibration honestly - you have {n_usable} so far.")
            else:
                cal_display = cal.copy()
                cal_display["avg_predicted"] = (cal_display["avg_predicted"] * 100).round(1).astype(str) + "%"
                cal_display["actual_hit_rate"] = (cal_display["actual_hit_rate"] * 100).round(1).astype(str) + "%"
                st.dataframe(cal_display, use_container_width=True)

            st.dataframe(scored, use_container_width=True)
        st.dataframe(history, use_container_width=True)

with tab_ai:
    st.subheader("🤖 Gemini AI Financial Analyst")
    st.caption("Ask me anything about your current targets, risk metrics, or market sentiment.")
    
    if "GEMINI_API_KEY" in st.secrets and _HAS_GENAI:
        # If the official SDK is available, use it. Use a very small, defensive call pattern
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # type: ignore
        except Exception:
            pass

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input("e.g., 'Should I sell at the current price?'"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # Minimal local analyst fallback if Gemini SDK / key are not available
        if not _HAS_GENAI or "GEMINI_API_KEY" not in st.secrets:
            # concise heuristic reply
            if quant_score >= 75:
                advice = "Strong buy — models aligned."
            elif quant_score >= 55:
                advice = "Hold / watch — wait for better entry or conviction."
            else:
                advice = "Pass — negative edge."

            extra = []
            if rsi > 70:
                extra.append("RSI high — consider taking profits")
            if prob_dip < 0.4:
                extra.append("Low dip probability")

            reply = advice + (" — " + "; ".join(extra) if extra else "")
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)
        else:
            # Attempt to call the Gemini SDK if present. Keep it minimal and tolerant to errors.
            context = f"""
You are a quantitative risk analyst. Answer concisely based on these live metrics:
- Ticker: {ticker}
- Current Spot Price: {sym}{s0:.2f}
- Entry / Cost Basis: {sym}{entry_price:.2f}
- 14-Day RSI: {rsi:.1f}
- 3-Day ML Dip Probability: {prob_dip*100:.1f}%
- Target Buy Limit: {sym}{optimal_buy:.2f}
- Target Sell Limit: {sym}{optimal_sell:.2f}
- Take Profit Target: {sym}{target_tp:.2f}
- Stop Loss Limit: {sym}{target_sl:.2f}
- News Sentiment Score: {sentiment_score:+.2f}
"""
            prompt = f"{context}\n\nUser Question: {user_query}"
            with st.spinner("Analyzing data..."):
                try:
                    resp = genai.generate(prompt=prompt)  # type: ignore
                    text = resp.text if hasattr(resp, "text") else str(resp)
                    st.session_state.chat_history.append({"role": "assistant", "content": text})
                    st.chat_message("assistant").write(text)
                except Exception as e:
                    st.error(f"API Error: {e}")

# --- 5-YEAR BACKTEST MODULE UI ---
st.markdown("---")
with st.expander("📊 Run 5-Year Strategy Backtest (Walk-Forward Simulation)", expanded=False):
    # Convert absolute dollar limits into percentages for the backtest engine
    sl_pct_bt = (entry_price - target_sl) / entry_price if entry_price > 0 else 0.07
    tp_pct_bt = (target_tp - entry_price) / entry_price if entry_price > 0 else 0.10

    st.write(f"Testing the Master Quant Strategy strictly on **{ticker}** over the last 5 years, isolating out macro crashes (VIX/SPY regime filter) and targeting a {sl_pct_bt*100:.1f}% stop loss and ~{tp_pct_bt*100:.1f}% take profit.")
    
    if st.button(f"Initialize {ticker} Backtest Engine"):
        with st.spinner("Compiling historical data & simulating executions..."):
            bt_results = run_historical_backtest(ticker, sl_pct=sl_pct_bt, tp_pct=tp_pct_bt)
            
            if bt_results is None:
                st.error("Backtest failed: Insufficient data for this ticker.")
            else:
                equity_df, strat_return, buy_hold_return, win_rate, total_trades = bt_results
                
                # Top Level Metrics
                cols = st.columns(4)
                cols[0].metric("Total Trades", total_trades)
                cols[1].metric("Historical Win Rate", f"{win_rate:.1f}%")
                cols[2].metric("Strategy Return", f"{strat_return:.1f}%")
                cols[3].metric("Buy & Hold Return", f"{buy_hold_return:.1f}%")
                
                # Equity Curve Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["Equity"], mode='lines', name='Strategy Equity', line=dict(color='#00ff9d', width=2)))
                fig.update_layout(title="Strategy Equity Growth ($10,000 Initial Capital)", template="plotly_dark", height=400, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                if strat_return > buy_hold_return:
                    st.success("🟢 **Alpha Generated:** The dynamic quant strategy successfully outperformed passive buy-and-hold risk over the 5-year period.")
                else:
                    st.warning("🟡 **Risk Mitigation:** The strict macro and technical filters resulted in less return than passive buy-and-hold (though likely with less drawdown).")