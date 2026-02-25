"""
SQLite cache for yfinance price history and fundamental info.

Cache is keyed by (symbol, date) where date is today's ISO date — data
is considered fresh for the rest of the calendar day and re-fetched the
next trading day.

DB file: stock_cache.db  (excluded from git)
"""

import json
import os
import sqlite3

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "stock_cache.db")

_CREATE = """
CREATE TABLE IF NOT EXISTS price_cache (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    data   TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS info_cache (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    data   TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(_CREATE)
    return con


def load(symbol: str, date_iso: str) -> tuple:
    """
    Return (df, info) if both are cached for *symbol* on *date_iso*,
    otherwise (None, None).
    """
    with _connect() as con:
        prow = con.execute(
            "SELECT data FROM price_cache WHERE symbol=? AND date=?",
            (symbol, date_iso),
        ).fetchone()
        irow = con.execute(
            "SELECT data FROM info_cache WHERE symbol=? AND date=?",
            (symbol, date_iso),
        ).fetchone()

    if not prow or not irow:
        return None, None

    df   = pd.read_json(prow[0], orient="split", convert_dates=True)
    info = json.loads(irow[0])
    return df, info


def save(symbol: str, date_iso: str, df: pd.DataFrame, info: dict) -> None:
    """Persist price history and fundamentals for *symbol* on *date_iso*."""
    df_json   = df.to_json(orient="split", date_format="iso")
    info_json = json.dumps(
        {k: v for k, v in info.items() if v is not None},
        default=str,   # handle any non-serialisable types safely
    )
    with _connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO price_cache (symbol, date, data) VALUES (?,?,?)",
            (symbol, date_iso, df_json),
        )
        con.execute(
            "INSERT OR REPLACE INTO info_cache (symbol, date, data) VALUES (?,?,?)",
            (symbol, date_iso, info_json),
        )
        con.commit()
