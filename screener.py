#!/usr/bin/env python3
"""
Taiwan Stock Screener — core logic module.

Provides screen() for use by both the CLI (stock_picker.py) and the
Flask web server (app.py).
"""

import json
import os
import threading
import time
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from typing import Callable

import stock_cache
import t86_cache

import numpy as np
import pandas as pd
import yfinance as yf

TW_TZ = timezone(timedelta(hours=8))

warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    # Scoring weights (must sum to 1.0)
    "weight_technical": 0.35,
    "weight_value":     0.35,
    "weight_financial": 0.30,

    # Value thresholds
    "pe_excellent": 12,           # P/E below this = excellent
    "pe_good":      20,           # P/E below this = good
    "pb_excellent": 1.5,          # P/B below this = excellent
    "pb_good":      3.0,          # P/B below this = good
    "div_yield_good":      0.03,  # 3% dividend yield = good
    "div_yield_excellent": 0.05,  # 5% dividend yield = excellent

    # Financial health thresholds
    "rev_growth_good":      0.05,  # 5%
    "rev_growth_excellent": 0.15,  # 15%
    "eps_growth_good":      0.05,  # 5%
    "eps_growth_excellent": 0.20,  # 20%

    # Output
    "min_score": 50,   # Minimum composite score to appear in top picks
    "top_n":     20,   # Number of top picks to display
    "api_delay": 0.5,  # Seconds between API calls to avoid rate limiting
}


# ─── Taiwan Stock Universe ─────────────────────────────────────────────────────

_UNIVERSE_CACHE = os.path.join(os.path.dirname(__file__), "universe_cache.json")
_TWSE_ALL_URL   = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"


def _load_universe() -> list[str]:
    """
    Return all TWSE-listed common stocks as 'XXXX.TW' symbols.

    Fetches from the TWSE Open API and caches to universe_cache.json for
    the rest of the calendar day.  Falls back to the hardcoded list on
    any network or parse failure.
    """
    today = datetime.now(TW_TZ).date().isoformat()

    # ── Cache hit ──────────────────────────────────────────────────────────────
    try:
        with open(_UNIVERSE_CACHE) as f:
            cached = json.load(f)
        if cached.get("date") == today and cached.get("symbols"):
            return cached["symbols"]
    except Exception:
        pass

    # ── Fetch from TWSE Open API ───────────────────────────────────────────────
    try:
        req = urllib.request.Request(
            _TWSE_ALL_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        symbols = [
            f"{row['Code']}.TW"
            for row in data
            if row.get("Code", "").isdigit()
            and len(row["Code"]) == 4
            and not row["Code"].startswith("0")
        ]

        if symbols:
            with open(_UNIVERSE_CACHE, "w") as f:
                json.dump({"date": today, "symbols": symbols}, f)
            return symbols
    except Exception:
        pass

    # ── Fallback: hardcoded list ───────────────────────────────────────────────
    return _FALLBACK_UNIVERSE


_FALLBACK_UNIVERSE = [
    # ── Semiconductors ────────────────────────────────────────────────────────
    "2330.TW",  # TSMC
    "2303.TW",  # UMC
    "2454.TW",  # MediaTek
    "2379.TW",  # Realtek
    "3034.TW",  # Novatek
    "2344.TW",  # Winbond
    "2337.TW",  # Macronix
    "3711.TW",  # ASMedia Technology
    "6415.TW",  # Silergy Corp

    # ── Electronics & Manufacturing ────────────────────────────────────────────
    "2317.TW",  # Hon Hai (Foxconn)
    "2382.TW",  # Quanta Computer
    "2357.TW",  # ASUS
    "2353.TW",  # Acer
    "2356.TW",  # Inventec
    "2301.TW",  # Lite-On
    "2385.TW",  # Chicony
    "2308.TW",  # Delta Electronics
    "3008.TW",  # LARGAN Precision
    "6669.TW",  # Wiwynn
    "4938.TW",  # Pegatron

    # ── Financials ─────────────────────────────────────────────────────────────
    "2881.TW",  # Fubon Financial
    "2882.TW",  # Cathay Financial
    "2883.TW",  # KGI Financial
    "2884.TW",  # E.SUN Financial
    "2885.TW",  # Yuanta Financial
    "2886.TW",  # Mega Financial
    "2887.TW",  # Taishin Financial
    "2890.TW",  # Sinopac Financial
    "2891.TW",  # CTBC Financial
    "2892.TW",  # First Financial
    "5871.TW",  # Chailease Holding

    # ── Telecom ────────────────────────────────────────────────────────────────
    "2412.TW",  # Chunghwa Telecom
    "4904.TW",  # Far EasTone
    "3045.TW",  # Taiwan Mobile

    # ── Petrochemicals & Materials ─────────────────────────────────────────────
    "1301.TW",  # Formosa Plastics
    "1303.TW",  # Nan Ya Plastics
    "1326.TW",  # Formosa Chemicals
    "6505.TW",  # Formosa Petrochemical
    "2002.TW",  # China Steel
    "2006.TW",  # Tung Ho Steel

    # ── Shipping ───────────────────────────────────────────────────────────────
    "2603.TW",  # Evergreen Marine
    "2609.TW",  # Yang Ming Marine
    "2615.TW",  # Wan Hai Lines

    # ── Consumer & Retail ──────────────────────────────────────────────────────
    "2912.TW",  # President Chain Store (7-Eleven TW)
    "1216.TW",  # Uni-President Enterprises
    "2105.TW",  # Cheng Shin Rubber
]

STOCK_UNIVERSE = _load_universe()


# ─── Technical Indicators ─────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    df["MA5"]        = close.rolling(5).mean()
    df["MA20"]       = close.rolling(20).mean()
    df["MA60"]       = close.rolling(60).mean()
    df["MA120"]      = close.rolling(120).mean()
    df["RSI"]        = _rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(close)
    return df


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_technical(df: pd.DataFrame) -> tuple[int, dict]:
    """Score 0-100 based on MA, RSI, and MACD."""
    if df is None or len(df) < 61:
        return 0, {}

    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    score   = 0
    details = {}

    close = float(latest["Close"])

    # BIAS ratio vs MA5 — (close - MA5) / MA5 * 100
    ma5 = latest.get("MA5", float("nan"))
    if not np.isnan(ma5) and ma5 != 0:
        details["Bias5"] = round((close - float(ma5)) / float(ma5) * 100, 2)

    # Moving averages (40 pts)
    ma20 = latest.get("MA20", float("nan"))
    ma60 = latest.get("MA60", float("nan"))
    if not np.isnan(ma20):
        above20 = close > ma20
        details["above_MA20"] = above20
        score += 20 if above20 else 0
    if not np.isnan(ma60):
        above60 = close > ma60
        details["above_MA60"] = above60
        score += 20 if above60 else 0

    # RSI (30 pts) — reward healthy momentum, penalise extremes
    rsi = latest.get("RSI", float("nan"))
    if not np.isnan(rsi):
        details["RSI"] = round(float(rsi), 1)
        if 40 <= rsi <= 65:
            score += 30
        elif (30 <= rsi < 40) or (65 < rsi <= 75):
            score += 15

    # MACD (30 pts)
    macd  = latest.get("MACD",        float("nan"))
    msig  = latest.get("MACD_Signal", float("nan"))
    mhist = latest.get("MACD_Hist",   float("nan"))
    phist = prev.get("MACD_Hist",     float("nan"))
    if not (np.isnan(macd) or np.isnan(msig)):
        bullish = macd > msig
        details["MACD_bullish"] = bullish
        score += 20 if bullish else 0
        if not (np.isnan(mhist) or np.isnan(phist)) and mhist > phist:
            score += 10

    return min(score, 100), details


def score_value(info: dict) -> tuple[int, dict]:
    """Score 0-100 based on P/E, P/B, and dividend yield."""
    score   = 0
    details = {}

    # P/E ratio (40 pts)
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe and pe > 0:
        details["PE"] = round(float(pe), 1)
        if pe < CONFIG["pe_excellent"]:
            score += 40
        elif pe < CONFIG["pe_good"]:
            score += 25
        elif pe < 30:
            score += 10

    # P/B ratio (30 pts)
    pb = info.get("priceToBook")
    if pb and pb > 0:
        details["PB"] = round(float(pb), 2)
        if pb < CONFIG["pb_excellent"]:
            score += 30
        elif pb < CONFIG["pb_good"]:
            score += 15
        elif pb < 5:
            score += 5

    # Dividend yield (30 pts)
    dy = info.get("dividendYield")
    if dy:
        if dy > 1:
            dy = dy / 100
        details["DivYield"] = round(dy * 100, 2)  # store as % float e.g. 3.5
        if dy >= CONFIG["div_yield_excellent"]:
            score += 30
        elif dy >= CONFIG["div_yield_good"]:
            score += 20
        elif dy >= 0.01:
            score += 10

    return min(score, 100), details


def score_financial(info: dict) -> tuple[int, dict]:
    """Score 0-100 based on revenue growth, earnings growth, and profit margin."""
    score   = 0
    details = {}

    # Revenue growth YoY (40 pts)
    rg = info.get("revenueGrowth")
    if rg is not None:
        details["RevGrowth"] = round(float(rg) * 100, 1)  # store as % float e.g. 20.5
        if rg >= CONFIG["rev_growth_excellent"]:
            score += 40
        elif rg >= CONFIG["rev_growth_good"]:
            score += 25
        elif rg >= 0:
            score += 10

    # Earnings growth (40 pts)
    eg = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
    if eg is not None:
        details["EPSGrowth"] = round(float(eg) * 100, 1)  # store as % float
        if eg >= CONFIG["eps_growth_excellent"]:
            score += 40
        elif eg >= CONFIG["eps_growth_good"]:
            score += 25
        elif eg >= 0:
            score += 10

    # Profit margin (20 pts)
    pm = info.get("profitMargins")
    if pm is not None:
        details["ProfitMargin"] = round(float(pm) * 100, 1)  # store as % float
        if pm >= 0.15:
            score += 20
        elif pm >= 0.05:
            score += 10
        elif pm >= 0:
            score += 5

    return min(score, 100), details


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch(symbol: str, retries: int = 3, timeout: int = 20) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Fetch 1-year price history and fundamental info via yfinance.

    Checks the local SQLite cache first — if today's data is already
    stored, returns it immediately without hitting the network.
    Otherwise fetches from yfinance (daemon thread with hard timeout),
    saves to cache on success, and retries with exponential backoff on
    clean failures.
    """
    today = datetime.now(TW_TZ).date().isoformat()

    # ── Cache hit ──────────────────────────────────────────────────────────────
    df, info = stock_cache.load(symbol, today)
    if df is not None:
        return df, info

    # ── Fetch from yfinance ────────────────────────────────────────────────────
    for attempt in range(retries):
        container: list = []

        def _do_fetch():
            try:
                ticker = yf.Ticker(symbol)
                df     = ticker.history(period="1y")
                if not df.empty:
                    container.append((df, ticker.info))
            except Exception:
                pass

        t = threading.Thread(target=_do_fetch, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if container:
            df, info = container[0]
            stock_cache.save(symbol, today, df, info)
            return df, info

        if t.is_alive():
            # Hung socket — skip entirely, no retry
            return None, None

        # Clean failure — retry with backoff
        if attempt < retries - 1:
            time.sleep(2 ** attempt)     # 1 s → 2 s → 4 s

    return None, None


def _nan_to_none(val):
    """Convert NaN/inf floats to None for JSON safety."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
    except Exception:
        pass
    return val


def _parse_shares(s) -> int | None:
    """Parse a comma-formatted share count string like '13,345,675' or '-148,250' to int."""
    try:
        return int(str(s).replace(",", "").replace(" ", ""))
    except (ValueError, AttributeError):
        return None


# ─── TWSE T86 — Institutional Flow ────────────────────────────────────────────

def fetch_t86(max_lookback: int = 14) -> tuple[dict[str, dict], str | None]:
    """
    Fetch TWSE T86 daily institutional investor net buy/sell data (share counts).

    Column mapping from T86 JSON response:
        [0]  Security Code
        [3]  Foreign Investors net  (excl. foreign dealers)
        [9]  Securities Investment Trust net  (domestic mutual funds)
        [10] Dealers net  (combined proprietary + hedge)
        [17] Three-major-institutions total net

    Returns:
        (data_dict, date_iso) — data_dict keyed by 4-digit stock code.
        On failure returns ({}, None).
    """
    today = datetime.now(TW_TZ).date()
    for delta in range(max_lookback):
        d        = today - timedelta(days=delta)
        date_iso = d.isoformat()

        # ── Cache hit ──────────────────────────────────────────────────────────
        cached = t86_cache.load_day(date_iso)
        if cached is not None:
            return cached, date_iso

        # ── Fetch from API ─────────────────────────────────────────────────────
        date_str = d.strftime("%Y%m%d")
        url      = (
            "https://www.twse.com.tw/en/fund/T86"
            f"?response=json&date={date_str}&selectType=ALL"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
            if payload.get("stat") != "OK":
                continue
            result = {}
            for row in payload.get("data", []):
                code = row[0].strip()
                result[code] = {
                    "foreign_net":     _parse_shares(row[3]),
                    "trust_net":       _parse_shares(row[9]),
                    "dealer_net":      _parse_shares(row[10]),
                    "three_major_net": _parse_shares(row[17]),
                }
            t86_cache.save_day(date_iso, result)
            return result, date_iso
        except Exception:
            continue
    return {}, None


def fetch_t86_multi(n_days: int = 7, max_lookback: int = 25) -> tuple[list[dict], list[str]]:
    """
    Fetch up to n_days of TWSE T86 institutional flow data, most-recent first.

    Returns (days_data, dates):
      days_data — list of per-day dicts (same structure as fetch_t86() returns),
                  ordered most-recent first
      dates     — matching ISO date strings
    Stops after collecting n_days successful trading days within max_lookback
    calendar days. Adds 0.3s sleep between requests to avoid rate-limiting.
    """
    today     = datetime.now(TW_TZ).date()
    days_data = []
    dates     = []

    for delta in range(max_lookback):
        if len(days_data) >= n_days:
            break
        d        = today - timedelta(days=delta)
        date_iso = d.isoformat()

        # ── Cache hit (no sleep needed) ────────────────────────────────────────
        cached = t86_cache.load_day(date_iso)
        if cached is not None:
            days_data.append(cached)
            dates.append(date_iso)
            continue

        # ── Fetch from API ─────────────────────────────────────────────────────
        date_str = d.strftime("%Y%m%d")
        url      = (
            "https://www.twse.com.tw/en/fund/T86"
            f"?response=json&date={date_str}&selectType=ALL"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
            if payload.get("stat") != "OK":
                if delta > 0:
                    time.sleep(0.3)
                continue
            result = {}
            for row in payload.get("data", []):
                code = row[0].strip()
                result[code] = {
                    "foreign_net":     _parse_shares(row[3]),
                    "trust_net":       _parse_shares(row[9]),
                    "dealer_net":      _parse_shares(row[10]),
                    "three_major_net": _parse_shares(row[17]),
                }
            t86_cache.save_day(date_iso, result)
            days_data.append(result)
            dates.append(date_iso)
        except Exception:
            pass
        if delta > 0:
            time.sleep(0.3)

    return days_data, dates


def _consecutive_buys(days_data: list[dict], code: str, field: str) -> int:
    """Count consecutive days from most recent where days_data[i][code][field] > 0."""
    count = 0
    for day in days_data:          # already ordered most-recent first
        val = day.get(code, {}).get(field)
        if val is not None and val > 0:
            count += 1
        else:
            break
    return count


# ─── Screener ─────────────────────────────────────────────────────────────────

def screen(
    symbols: list[str],
    on_progress: Callable[[int, int, str, bool, dict], None] | None = None,
) -> list[dict]:
    """
    Screen a list of symbols and return result dicts with numeric values.

    on_progress(current, total, symbol, ok, result_dict) is called after
    each symbol is processed. Pass None to skip the callback.
    """
    total   = len(symbols)
    results = []

    # Fetch institutional flow for up to 7 trading days — covers all TWSE stocks
    days_data, dates = fetch_t86_multi(7)
    t86_map  = days_data[0] if days_data else {}
    t86_date = dates[0]     if dates     else None

    for i, sym in enumerate(symbols, 1):
        try:
            df, info = fetch(sym)
            if df is None:
                if on_progress:
                    on_progress(i, total, sym, False, {})
                time.sleep(CONFIG["api_delay"])
                continue

            df = add_indicators(df)

            t_score, t_det = score_technical(df)
            v_score, v_det = score_value(info)
            f_score, f_det = score_financial(info)

            composite = (
                t_score * CONFIG["weight_technical"]
                + v_score * CONFIG["weight_value"]
                + f_score * CONFIG["weight_financial"]
            )

            price = float(df["Close"].iloc[-1])
            name  = (info.get("shortName") or sym)[:28]

            # Look up institutional flow by the bare 4-digit code
            code = sym.split(".")[0]
            t86  = t86_map.get(code, {})

            result = {
                "symbol":          sym,
                "name":            name,
                "price":           round(price, 2),
                "composite":       round(composite, 1),
                "technical":       t_score,
                "value":           v_score,
                "financial":       f_score,
                "pe":              _nan_to_none(v_det.get("PE")),
                "pb":              _nan_to_none(v_det.get("PB")),
                "div_yield":       _nan_to_none(v_det.get("DivYield")),
                "rsi":             _nan_to_none(t_det.get("RSI")),
                "bias5":           _nan_to_none(t_det.get("Bias5")),
                "macd_bull":       bool(t_det.get("MACD_bullish", False)),
                "rev_growth":      _nan_to_none(f_det.get("RevGrowth")),
                "eps_growth":      _nan_to_none(f_det.get("EPSGrowth")),
                "profit_margin":   _nan_to_none(f_det.get("ProfitMargin")),
                # ── TWSE T86 institutional flow (shares, positive = net buy) ──
                "foreign_net":     t86.get("foreign_net"),
                "trust_net":       t86.get("trust_net"),
                "dealer_net":      t86.get("dealer_net"),
                "three_major_net": t86.get("three_major_net"),
                "t86_date":        t86_date,
                # ── Consecutive buy-day streaks ───────────────────────────────
                "foreign_consecutive": _consecutive_buys(days_data, code, "foreign_net"),
                "sitc_consecutive":    _consecutive_buys(days_data, code, "trust_net"),
                "dealer_consecutive":  _consecutive_buys(days_data, code, "dealer_net"),
            }

            results.append(result)

            if on_progress:
                on_progress(i, total, sym, True, result)

        except Exception:
            if on_progress:
                on_progress(i, total, sym, False, {})

        time.sleep(CONFIG["api_delay"])

    return results
