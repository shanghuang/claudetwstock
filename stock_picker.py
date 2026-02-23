#!/usr/bin/env python3
"""
Taiwan Stock Market Screener — CLI wrapper
Screens stocks using value investing, technical analysis, and financial health criteria.

Usage:
    python stock_picker.py                    # Screen default list of major TW stocks
    python stock_picker.py 2330 2454 2317     # Screen specific stocks (auto-adds .TW suffix)
    python stock_picker.py 2330.TW 2454.TW    # Same with explicit suffix
"""

import sys
from datetime import datetime

import pandas as pd

from screener import CONFIG, STOCK_UNIVERSE, screen


# ─── Output ───────────────────────────────────────────────────────────────────

def _fmt(val, suffix="") -> str:
    """Format a numeric value or return '-' if None."""
    if val is None:
        return "-"
    return f"{val}{suffix}"


def _fmt_shares(val) -> str:
    """Format share count with K/M suffix and +/- sign."""
    if val is None:
        return "-"
    sign = "+" if val > 0 else ""
    abs_v = abs(val)
    if abs_v >= 1_000_000:
        return f"{sign}{val / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{sign}{val / 1_000:.0f}K"
    return f"{sign}{val}"


def display(results: list[dict]) -> None:
    if not results:
        print("\nNo results — check your stock list or network connection.")
        return

    results.sort(key=lambda r: r["composite"], reverse=True)
    qualified = [r for r in results if r["composite"] >= CONFIG["min_score"]]
    top       = qualified[: CONFIG["top_n"]] or results[: CONFIG["top_n"]]

    t86_date = results[0].get("t86_date") if results else None
    t86_note = f"  Institutional flow: {t86_date}" if t86_date else "  Institutional flow: N/A"

    SEP = "─" * 130
    print(f"\n{'═' * 130}")
    print(
        f"  Taiwan Stock Picks — "
        f"{len(results)} screened · {len(qualified)} passed ≥{CONFIG['min_score']} score"
        f"{t86_note}"
    )
    print(f"{'═' * 130}")
    print(
        f"{'#':<4} {'Symbol':<12} {'Name':<29} {'Price':>8} "
        f"{'Score':>6} {'Tech':>5} {'Val':>5} {'Fin':>5} "
        f"{'PE':>6} {'PB':>5} {'Div':>6} {'RSI':>5} {'BIAS5':>7} {'MACD':>5} "
        f"{'RevG':>7} {'EPSG':>7} {'Foreign':>9} {'3-Major':>9}"
    )
    print(SEP)

    for rank, r in enumerate(top, 1):
        div_str = _fmt(r["div_yield"], "%") if r["div_yield"] is not None else "-"
        rg_str  = _fmt(r["rev_growth"], "%") if r["rev_growth"] is not None else "-"
        eg_str  = _fmt(r["eps_growth"], "%") if r["eps_growth"] is not None else "-"
        macd_arrow = "↑" if r["macd_bull"] else "↓"
        print(
            f"{rank:<4} {r['symbol']:<12} {r['name']:<29} {r['price']:>8.2f} "
            f"{r['composite']:>6.1f} {r['technical']:>5} {r['value']:>5} {r['financial']:>5} "
            f"{str(_fmt(r['pe'])):>6} {str(_fmt(r['pb'])):>5} {div_str:>6} "
            f"{str(_fmt(r['rsi'])):>5} "
            f"{(_fmt(r['bias5'], '') + '%') if r.get('bias5') is not None else '-':>7} "
            f"{macd_arrow:>5} "
            f"{rg_str:>7} {eg_str:>7} "
            f"{_fmt_shares(r.get('foreign_net')):>9} {_fmt_shares(r.get('three_major_net')):>9}"
        )

    print(SEP)
    print(
        "Score = Technical×35% + Value×35% + Financial×30%\n"
        "Tech: MA20/60 + RSI (40–65) + MACD  │  "
        "Val: P/E + P/B + Dividend  │  "
        "Fin: Revenue & EPS growth + Margin  │  "
        "Foreign/3-Major: TWSE T86 net shares (positive = net buy)"
    )

    # Save CSV
    rows = []
    for r in results:
        rows.append({
            "Symbol":         r["symbol"],
            "Name":           r["name"],
            "Price(TWD)":     r["price"],
            "Composite":      r["composite"],
            "Technical":      r["technical"],
            "Value":          r["value"],
            "Financial":      r["financial"],
            "PE":             r["pe"] if r["pe"] is not None else "-",
            "PB":             r["pb"] if r["pb"] is not None else "-",
            "DivYield":       f"{r['div_yield']}%" if r["div_yield"] is not None else "-",
            "RSI":            r["rsi"] if r["rsi"] is not None else "-",
            "Bias5":          f"{r['bias5']}%" if r.get("bias5") is not None else "-",
            "MACD_Bull":      "↑" if r["macd_bull"] else "↓",
            "RevGrowth":      f"{r['rev_growth']}%" if r["rev_growth"] is not None else "-",
            "EPSGrowth":      f"{r['eps_growth']}%" if r["eps_growth"] is not None else "-",
            "ProfitMargin":   f"{r['profit_margin']}%" if r["profit_margin"] is not None else "-",
            "ForeignNet":     r.get("foreign_net", ""),
            "SITC_Net":       r.get("trust_net", ""),
            "DealerNet":      r.get("dealer_net", ""),
            "ThreeMajorNet":  r.get("three_major_net", ""),
            "T86_Date":       r.get("t86_date", ""),
        })
    out = f"tw_picks_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    pd.DataFrame(rows).sort_values("Composite", ascending=False).to_csv(out, index=False)
    print(f"\nFull results saved → {out}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("═" * 60)
    print("  Taiwan Stock Market Screener")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 60)
    print(
        f"\nWeights → Technical: {CONFIG['weight_technical']*100:.0f}%  "
        f"Value: {CONFIG['weight_value']*100:.0f}%  "
        f"Financial: {CONFIG['weight_financial']*100:.0f}%"
    )

    if len(sys.argv) > 1:
        raw     = [s.strip().upper() for s in sys.argv[1:]]
        symbols = [s if "." in s else f"{s}.TW" for s in raw]
        print(f"\nCustom list: {symbols}")
    else:
        symbols = STOCK_UNIVERSE
        print(f"\nDefault universe: {len(symbols)} major Taiwan stocks")

    def _cli_progress(current, total, symbol, ok, result):
        status = (
            f"composite={result['composite']:5.1f}  "
            f"(T={result['technical']:3d} V={result['value']:3d} F={result['financial']:3d})"
            if ok else "✗  no data"
        )
        print(f"[{current:2d}/{total}] {symbol:<12} {status}")

    print(f"\nScreening {len(symbols)} stocks ...\n")
    results = screen(symbols, on_progress=_cli_progress)
    display(results)


if __name__ == "__main__":
    main()
