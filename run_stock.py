#!/usr/bin/env python3
"""
================================================================================
 INSTITUTIONAL MULTI-MODEL QUANTITATIVE TRADE ENGINE
================================================================================
"""

import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is not installed. Run: pip install yfinance")
    sys.exit(1)

# ------------------------------------------------------------------------
# 1. MODEL: NLP SENTIMENT ENGINE
# ------------------------------------------------------------------------
VADER_AVAILABLE = False
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

_HEURISTIC_LEXICON = {
    "beat": 1, "beats": 1, "surge": 1.2, "soar": 1.3, "rally": 1, "upgrade": 1,
    "outperform": 1, "growth": 0.7, "profit": 0.7, "bullish": 1.3, "record": 0.8,
    "miss": -1, "misses": -1, "plunge": -1.3, "crash": -1.5, "downgrade": -1,
    "underperform": -1, "loss": -0.7, "layoffs": -1, "bearish": -1.3, "weak": -0.6
}

def analyze_headline_heuristic(headline: str) -> float:
    words = headline.lower().replace(",", " ").replace(".", " ").split()
    score, hits = 0.0, 0
    for w in words:
        if w in _HEURISTIC_LEXICON:
            score += _HEURISTIC_LEXICON[w]
            hits += 1
    return float(np.clip(score / hits, -1.0, 1.0)) if hits > 0 else 0.0

def get_news_sentiment(ticker_obj: yf.Ticker, max_headlines: int = 20) -> tuple:
    try:
        news_items = ticker_obj.news or []
    except Exception:
        news_items = []

    headlines = []
    for item in news_items[:max_headlines]:
        title = item.get("title") if isinstance(item, dict) else None
        if not title and isinstance(item, dict) and "content" in item:
            title = item["content"].get("title")
        if title:
            headlines.append(title)

    if not headlines:
        return 0.0, 0

    scores = [
        _analyzer.polarity_scores(h)["compound"] if VADER_AVAILABLE else analyze_headline_heuristic(h)
        for h in headlines
    ]
    return float(np.clip(np.mean(scores), -1.0, 1.0)), len(headlines)

# ------------------------------------------------------------------------
# 2. MODELS: TECHNICAL ANALYSIS (RSI, MACD, BOLLINGER BANDS)
# ------------------------------------------------------------------------
def compute_technical_indicators(price_series: pd.Series) -> dict:
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    latest_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    s0 = float(price_series.iloc[-1])
    n = len(price_series)
    sma_20 = float(price_series.rolling(20).mean().iloc[-1]) if n >= 20 else s0
    sma_50 = float(price_series.rolling(50).mean().iloc[-1]) if n >= 50 else s0

    if n >= 20:
        bb_middle = price_series.rolling(20).mean()
        bb_std = price_series.rolling(20).std()
        bb_upper = float((bb_middle + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_middle - 2 * bb_std).iloc[-1])
    else:
        # short listing history (e.g. a newly-issued warrant) - collapse band to spot
        bb_upper = s0
        bb_lower = s0
    pct_b = (s0 - bb_lower) / (bb_upper - bb_lower + 1e-9)

    ema_12 = price_series.ewm(span=12, adjust=False).mean()
    ema_26 = price_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return {
        "rsi_14": latest_rsi,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "pct_b": pct_b,
        "macd_line": float(macd_line.iloc[-1]),
        "signal_line": float(signal_line.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
    }

# ------------------------------------------------------------------------
# 3. MODELS: VALUE-AT-RISK (VaR) & EXPECTED SHORTFALL (CVaR)
# ------------------------------------------------------------------------
def compute_advanced_risk_metrics(log_returns: pd.Series, s0: float, horizon_days: int) -> dict:
    confidence = 0.95
    mu = log_returns.mean() * horizon_days
    sigma = log_returns.std() * np.sqrt(horizon_days)

    var_param_pct = norm.ppf(1 - confidence, mu, sigma)
    var_param_usd = s0 * (1 - np.exp(var_param_pct))

    rolling_returns = log_returns.rolling(horizon_days).sum().dropna()
    if not rolling_returns.empty:
        var_hist_pct = np.percentile(rolling_returns, (1 - confidence) * 100)
        var_hist_usd = s0 * (1 - np.exp(var_hist_pct))
        
        tail_losses = rolling_returns[rolling_returns <= var_hist_pct]
        cvar_pct = tail_losses.mean() if not tail_losses.empty else var_hist_pct
        cvar_usd = s0 * (1 - np.exp(cvar_pct))
    else:
        var_hist_usd = var_param_usd
        cvar_usd = var_param_usd * 1.25

    return {
        "var_param_usd": var_param_usd,
        "var_hist_usd": var_hist_usd,
        "cvar_usd": cvar_usd,
    }

# ------------------------------------------------------------------------
# 4. MODEL: BLACK-SCHOLES OPTIONS ENGINE
# ------------------------------------------------------------------------
def black_scholes_price(s0: float, k: float, t: float, r: float, sigma: float, option_type: str = "call") -> float:
    if t <= 0 or sigma <= 0:
        return max(0.0, (s0 - k) if option_type == "call" else (k - s0))
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)) if option_type == "call" else float(k * np.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1))

# ------------------------------------------------------------------------
# 5. MODEL: SHORT-HORIZON MONTE CARLO & PROBABILITY OF PROFIT (PoP)
# ------------------------------------------------------------------------
def run_short_horizon_mc(s0: float, entry_price: float, daily_vol: float, holding_days: int, sims: int = 10000) -> dict:
    rng = np.random.default_rng(42)
    sigma_period = daily_vol * np.sqrt(holding_days)
    
    z = rng.standard_normal(sims)
    terminal_prices = s0 * np.exp(-0.5 * (sigma_period ** 2) + sigma_period * z)

    profitable_paths = np.sum(terminal_prices > entry_price)
    pop_pct = (profitable_paths / sims) * 100.0

    return {
        "expected_mean": float(np.mean(terminal_prices)),
        "p05": float(np.percentile(terminal_prices, 5)),
        "p25": float(np.percentile(terminal_prices, 25)),
        "p50": float(np.median(terminal_prices)),
        "p75": float(np.percentile(terminal_prices, 75)),
        "p95": float(np.percentile(terminal_prices, 95)),
        "pop_pct": pop_pct,
        "period_vol_pct": sigma_period * 100.0,
    }

# ------------------------------------------------------------------------
# 5b. MODEL: OPTIMAL BUY & SELL PRICE ENGINE
# ------------------------------------------------------------------------
def calculate_3day_buy_target(s0: float, daily_vol: float, bb_lower: float, sma_20: float, sentiment_score: float) -> dict:
    sigma_3d = daily_vol * np.sqrt(3)
    vol_pullback_target = s0 * (1 - 0.5 * sigma_3d)
    tech_support = bb_lower if bb_lower < s0 else sma_20
    base_entry = (0.6 * vol_pullback_target) + (0.4 * tech_support)
    sentiment_adjustment = sentiment_score * (0.3 * sigma_3d * s0)
    optimal_buy_limit = min(s0, base_entry + sentiment_adjustment)
    conservative_buy = min(optimal_buy_limit, s0 * (1 - 1.0 * sigma_3d))
    return {"optimal_buy": optimal_buy_limit, "conservative_buy": conservative_buy, "sigma_3d": sigma_3d}

def calculate_optimal_sell_target(s0: float, daily_vol: float, bb_upper: float, sma_20: float, rsi: float,
                                   sentiment_score: float, holding_days: int) -> dict:
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
    return {"optimal_sell": optimal_sell, "aggressive_sell": aggressive_sell, "sigma_h": sigma_h}

# ------------------------------------------------------------------------
# 6. MODEL: FUNDAMENTAL VALUATION
# ------------------------------------------------------------------------
def fetch_fundamentals(ticker_obj: yf.Ticker) -> dict:
    try:
        info = ticker_obj.info or {}
        return {
            "forward_pe": info.get("forwardPE", "N/A"),
            "peg_ratio": info.get("pegRatio", "N/A"),
            "beta": info.get("beta", "N/A"),
            "profit_margins": f"{info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') else "N/A",
        }
    except Exception:
        return {"forward_pe": "N/A", "peg_ratio": "N/A", "beta": "N/A", "profit_margins": "N/A"}

# ------------------------------------------------------------------------
# 7. DATA INGESTION & REPORTING
# ------------------------------------------------------------------------
def fetch_stock_data(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"Could not retrieve price history for '{ticker}'.")
        
    price_series = hist["Close"].dropna()
    s0 = float(price_series.iloc[-1])
    log_returns = pd.Series(np.log(price_series / price_series.shift(1))).dropna()
    daily_vol = log_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    
    return s0, daily_vol, annual_vol, log_returns, price_series, tk

def print_comprehensive_report(ticker: str, s0: float, entry_price: float, holding_days: int, max_risk_pct: float,
                               daily_vol: float, annual_vol: float, tech: dict, risk: dict, bs_call: float, 
                               bs_put: float, bs_skew: float, sentiment: float, hl_count: int, fund: dict, mc: dict,
                               buy_target: dict, sell_target: dict):
    W = 76
    line = "=" * W
    
    sigma_period = daily_vol * np.sqrt(holding_days)
    target_tp = entry_price * (1 + 1.5 * sigma_period)
    quant_sl = entry_price * (1 - 1.0 * sigma_period)
    user_sl = entry_price * (1 - max_risk_pct / 100.0)
    target_sl = max(quant_sl, user_sl)
    
    pnl_tp = ((target_tp - entry_price) / entry_price) * 100.0
    pnl_sl = ((target_sl - entry_price) / entry_price) * 100.0
    rr_ratio = (target_tp - entry_price) / (entry_price - target_sl) if (entry_price - target_sl) > 0 else 0.0
    unrealized = ((s0 - entry_price) / entry_price) * 100.0

    print("\n" + line)
    print(f" COMPREHENSIVE MULTI-MODEL QUANT REPORT -- {ticker.upper()}")
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Horizon: {holding_days} Trading Days")
    print(line)

    print("\n[1] POSITION & BASELINE METRICS")
    print("-" * W)
    print(f" Current Spot Price:        ${s0:,.2f}")
    print(f" Cost Basis (Entry):        ${entry_price:,.2f}  [Unrealized P&L: {unrealized:+.2f}%]")
    print(f" Annualized Volatility:     {annual_vol*100:.2f}%  (Period Volatility: ±{mc['period_vol_pct']:.2f}%)")
    print(f" Beta / Profit Margin:      {fund['beta']} / {fund['profit_margins']}")

    print("\n[2] TECHNICAL INDICATOR MATRIX")
    print("-" * W)
    print(f" RSI (14-Day):              {tech['rsi_14']:.2f} " + ("(Oversold)" if tech['rsi_14'] < 30 else "(Overbought)" if tech['rsi_14'] > 70 else "(Neutral)"))
    print(f" Moving Averages:           SMA-20: ${tech['sma_20']:.2f} | SMA-50: ${tech['sma_50']:.2f}")
    print(f" Bollinger Bands (20, 2):   Lower: ${tech['bb_lower']:.2f} | Upper: ${tech['bb_upper']:.2f} [%B: {tech['pct_b']:.2f}]")
    print(f" MACD (12, 26, 9):          MACD: {tech['macd_line']:.2f} | Signal: {tech['signal_line']:.2f} | Hist: {tech['macd_hist']:+.2f}")

    print("\n[3] NLP SENTIMENT & FUNDAMENTAL VALUATION")
    print("-" * W)
    print(f" Sentiment Score:           {sentiment:+.3f} (Analyzed {hl_count} news headlines)")
    print(f" Valuation Ratios:          Forward P/E: {fund['forward_pe']} | PEG Ratio: {fund['peg_ratio']}")

    print("\n[4] OPTIONS SKEW & DOWNSIDE RISK (VaR / CVaR)")
    print("-" * W)
    print(f" ATM Call / Put ({holding_days}d):    Call: ${bs_call:.2f} | Put: ${bs_put:.2f}")
    print(f" Options Premium Skew:      {bs_skew*100:+.2f}% of spot")
    print(f" 95% Parametric VaR:        ${risk['var_param_usd']:.2f} max loss expected over {holding_days} days")
    print(f" 95% Historical VaR:        ${risk['var_hist_usd']:.2f} max loss expected over {holding_days} days")
    print(f" Expected Shortfall (CVaR): ${risk['cvar_usd']:.2f} tail risk loss if VaR is breached")

    print("\n[5] MONTE CARLO & PROBABILITY OF PROFIT (PoP)")
    print("-" * W)
    print(f" Probability of Profit:     {mc['pop_pct']:.1f}% (Paths ending above ${entry_price:,.2f})")
    print(f" Projected Median Price:    ${mc['p50']:.2f} (Mean: ${mc['expected_mean']:.2f})")
    print(f" 5th / 95th Percentile:     ${mc['p05']:.2f}  to  ${mc['p95']:.2f}")

    print("\n[6] OPTIMAL BUY / SELL PRICE ENGINE")
    print("-" * W)
    print(f" 3-Day Optimal Buy Entry:   ${buy_target['optimal_buy']:.2f}  (Conservative: ${buy_target['conservative_buy']:.2f})")
    print(f" Optimal Sell Price:        ${sell_target['optimal_sell']:.2f}  (Aggressive: ${sell_target['aggressive_sell']:.2f})")
    print(f"   Basis: resistance (BB-upper/SMA-20) + {holding_days}d volatility + sentiment + RSI tilt")

    print("\n[7] QUANTITATIVE TRADE EXECUTION PLAN")
    print("=" * W)
    print(f"   >>> TARGET TAKE PROFIT:     ${target_tp:,.2f} ({pnl_tp:+.2f}%)")
    print(f"   >>> DYNAMIC STOP LOSS:      ${target_sl:,.2f} ({pnl_sl:+.2f}%)")
    print(f"   >>> Risk-to-Reward Ratio:   1 : {rr_ratio:.2f}")
    print("=" * W + "\n")

# ------------------------------------------------------------------------
# MAIN INTERACTIVE DRIVER
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== MULTI-MODEL QUANTITATIVE TRADE PLANNER ===")
    
    ticker = input("1. Enter Ticker Symbol [e.g. SKHY]: ").strip().upper() or "SKHY"
    
    try:
        s0, daily_vol, annual_vol, log_returns, price_series, tk = fetch_stock_data(ticker)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        sys.exit(1)

    entry_str = input(f"2. Enter your buy price (Press Enter for spot ${s0:,.2f}): ").strip()
    entry_price = float(entry_str) if entry_str else s0

    days_str = input("3. Enter target holding period in trading days [Default: 10 (~2 weeks)]: ").strip()
    holding_days = int(days_str) if days_str else 10

    risk_str = input("4. Enter max acceptable loss % for stop-loss [Default: 7.0%]: ").strip()
    max_risk_pct = float(risk_str) if risk_str else 7.0

    print(f"\nProcessing multi-model analytics for {ticker}...")
    
    tech = compute_technical_indicators(price_series)
    risk = compute_advanced_risk_metrics(log_returns, s0, holding_days)
    sentiment, hl_count = get_news_sentiment(tk)
    fund = fetch_fundamentals(tk)
    
    t_years = holding_days / 365.0
    r = 0.037
    bs_call = black_scholes_price(s0, s0, t_years, r, annual_vol, "call")
    bs_put = black_scholes_price(s0, s0, t_years, r, annual_vol, "put")
    bs_skew = (bs_call - bs_put) / s0

    mc = run_short_horizon_mc(s0, entry_price, daily_vol, holding_days)

    buy_target = calculate_3day_buy_target(s0, daily_vol, tech["bb_lower"], tech["sma_20"], sentiment)
    sell_target = calculate_optimal_sell_target(s0, daily_vol, tech["bb_upper"], tech["sma_20"], tech["rsi_14"], sentiment, holding_days)

    print_comprehensive_report(
        ticker, s0, entry_price, holding_days, max_risk_pct,
        daily_vol, annual_vol, tech, risk, bs_call, bs_put, bs_skew,
        sentiment, hl_count, fund, mc, buy_target, sell_target
    )