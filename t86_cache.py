"""
SQLite cache for TWSE T86 institutional flow data.

Schema (t86_cache.db):
    date            TEXT  — ISO date, e.g. "2025-02-20"
    code            TEXT  — 4-digit stock code, e.g. "2330"
    foreign_net     INTEGER
    trust_net       INTEGER
    dealer_net      INTEGER
    three_major_net INTEGER
    PRIMARY KEY (date, code)
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "t86_cache.db")

_CREATE = """
CREATE TABLE IF NOT EXISTS t86_cache (
    date            TEXT    NOT NULL,
    code            TEXT    NOT NULL,
    foreign_net     INTEGER,
    trust_net       INTEGER,
    dealer_net      INTEGER,
    three_major_net INTEGER,
    PRIMARY KEY (date, code)
);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute(_CREATE)
    con.commit()
    return con


def load_day(date_iso: str) -> dict[str, dict] | None:
    """
    Return cached T86 data for *date_iso* as a dict keyed by stock code,
    or None if that date is not in the cache.
    """
    with _connect() as con:
        rows = con.execute(
            "SELECT code, foreign_net, trust_net, dealer_net, three_major_net "
            "FROM t86_cache WHERE date = ?",
            (date_iso,),
        ).fetchall()

    if not rows:
        return None

    return {
        code: {
            "foreign_net":     foreign_net,
            "trust_net":       trust_net,
            "dealer_net":      dealer_net,
            "three_major_net": three_major_net,
        }
        for code, foreign_net, trust_net, dealer_net, three_major_net in rows
    }


def load_stock(code: str) -> list[dict]:
    """
    Return all cached rows for *code*, sorted ascending by date.
    Each row: {time, foreign_net, trust_net, dealer_net, three_major_net}.
    Returns [] if nothing is cached for this stock.
    """
    with _connect() as con:
        rows = con.execute(
            "SELECT date, foreign_net, trust_net, dealer_net, three_major_net "
            "FROM t86_cache WHERE code = ? ORDER BY date ASC",
            (code,),
        ).fetchall()
    return [
        {
            "time":           date,
            "foreign_net":    foreign_net,
            "trust_net":      trust_net,
            "dealer_net":     dealer_net,
            "three_major_net": three_major_net,
        }
        for date, foreign_net, trust_net, dealer_net, three_major_net in rows
    ]


def save_day(date_iso: str, data: dict[str, dict]) -> None:
    """
    Persist a full day of T86 data.  Existing rows for *date_iso* are
    replaced so re-running with fresh data is safe.
    """
    rows = [
        (
            date_iso,
            code,
            v.get("foreign_net"),
            v.get("trust_net"),
            v.get("dealer_net"),
            v.get("three_major_net"),
        )
        for code, v in data.items()
    ]
    with _connect() as con:
        con.executemany(
            "INSERT OR REPLACE INTO t86_cache "
            "(date, code, foreign_net, trust_net, dealer_net, three_major_net) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
