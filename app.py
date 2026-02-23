#!/usr/bin/env python3
"""
Taiwan Stock Screener — Flask web server.

Endpoints:
    GET  /        → serve index.html
    POST /screen  → start background screening thread
    GET  /status  → poll progress
    GET  /results → fetch completed results as JSON
"""

import threading

import yfinance as yf
from flask import Flask, jsonify, render_template, request

import screener
import t86_cache

app = Flask(__name__)

# ─── Global State ─────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_state = {
    "status":   "idle",   # idle | running | done | error
    "current":  0,
    "total":    0,
    "log":      [],       # [{symbol, ok, composite, technical, value, financial}]
    "results":  [],
    "t86_date": None,     # ISO date string of the T86 data used, e.g. "2025-01-17"
}


def _reset_state(total: int) -> None:
    with _lock:
        _state["status"]   = "running"
        _state["current"]  = 0
        _state["total"]    = total
        _state["log"]      = []
        _state["results"]  = []
        _state["t86_date"] = None


def _on_progress(current: int, total: int, symbol: str, ok: bool, result: dict) -> None:
    entry = {
        "symbol":    symbol,
        "ok":        ok,
        "composite": result.get("composite"),
        "t":         result.get("technical"),
        "v":         result.get("value"),
        "f":         result.get("financial"),
    }
    with _lock:
        _state["current"] = current
        _state["log"].append(entry)
        if ok:
            _state["results"].append(result)
            # Capture T86 date from the first successful result
            if _state["t86_date"] is None:
                _state["t86_date"] = result.get("t86_date")


def _run_screen(symbols: list[str]) -> None:
    try:
        screener.screen(symbols, on_progress=_on_progress)
        with _lock:
            _state["status"] = "done"
    except Exception as exc:
        with _lock:
            _state["status"] = f"error: {exc}"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", universe_count=len(screener.STOCK_UNIVERSE))


@app.route("/screen", methods=["POST"])
def start_screen():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Already running"}), 400

    body    = request.get_json(silent=True) or {}
    symbols = body.get("symbols")

    if symbols:
        # normalise: strip whitespace, uppercase, add .TW if no dot
        symbols = [
            s.strip().upper() if "." in s.strip().upper() else f"{s.strip().upper()}.TW"
            for s in symbols
        ]
    else:
        symbols = screener.STOCK_UNIVERSE

    _reset_state(len(symbols))
    t = threading.Thread(target=_run_screen, args=(symbols,), daemon=True)
    t.start()

    return jsonify({"started": True, "total": len(symbols)})


@app.route("/status")
def get_status():
    with _lock:
        return jsonify({
            "status":   _state["status"],
            "current":  _state["current"],
            "total":    _state["total"],
            "log":      list(_state["log"]),
            "t86_date": _state["t86_date"],
        })


@app.route("/results")
def get_results():
    with _lock:
        return jsonify(list(_state["results"]))


@app.route("/chart/<path:symbol>")
def get_chart(symbol):
    """Return 6 months of daily OHLCV + cached institutional flow for *symbol*."""
    try:
        hist = yf.Ticker(symbol).history(period="6mo", interval="1d")
        if hist.empty:
            return jsonify({"ohlcv": [], "flow": []})
        ohlcv = [
            {
                "time":   ts.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),   2),
                "high":   round(float(row["High"]),   2),
                "low":    round(float(row["Low"]),    2),
                "close":  round(float(row["Close"]),  2),
                "volume": int(row["Volume"]),
            }
            for ts, row in hist.iterrows()
        ]
        code = symbol.split(".")[0]
        flow = t86_cache.load_stock(code)
        return jsonify({"ohlcv": ohlcv, "flow": flow})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Port 5001 avoids conflict with macOS AirPlay Receiver on port 5000
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
