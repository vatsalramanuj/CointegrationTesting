#!/usr/bin/env python3
"""
Download an ETF universe from yfinance in the SAME csv layout as your stock file
(3 header rows: Ticker / Price / Date) and write the matching groups file.

Why these ETFs: pairs are only tested INSIDE a group, and every group is a set of funds that
are economically linked (same sector across providers, physical gold vs gold miners, oil vs
energy equities, neighbouring-maturity Treasuries, regional country funds, ...). Funds that
decay mechanically (leveraged / inverse / VIX products, natural gas) are left out.

Usage:
    python download_etf_prices.py                      # -> etf_prices.csv + groups_etf.csv
    python download_etf_prices.py --start 2010-01-01
    python download_etf_prices.py --groups-only        # just (re)write groups_etf.csv

Then:
    python backtest_rolling_rescreen.py etf_prices.csv --groups-csv groups_etf.csv \
        --min-formation-trades 0 --min-edge-ratio 2 --candidates 50 --select 15
"""
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

GROUPS = {
    "precious_metals": "GLD IAU SGOL SLV SIVR GDX GDXJ SIL",
    "us_large_cap_index": "SPY IVV VOO VTI QQQ DIA RSP",
    "us_small_mid": "IWM IJR IJH MDY VB",
    "us_style": "IWD IWF VTV VUG IVE IVW SPYG SPYV",
    "financials": "XLF VFH KBE KRE IYF",
    "energy_equity_and_oil": "XLE VDE XOP OIH IYE USO BNO",
    "technology": "XLK VGT IYW SMH SOXX",
    "healthcare": "XLV VHT IYH IBB XBI",
    "utilities": "XLU VPU IDU",
    "real_estate": "VNQ IYR SCHH RWR",
    "consumer_staples": "XLP VDC KXI",
    "consumer_discretionary": "XLY VCR XRT",
    "industrials": "XLI VIS ITA",
    "materials": "XLB VAW",
    "europe": "EWG EWQ EWI EWP EWN EWU EWL EWD VGK",
    "asia_pacific_developed": "EWJ EWA EWS EWH",
    "emerging_asia": "EWY EWT INDA FXI MCHI EWM THD",
    "latin_america": "EWZ EWW ECH",
    "world_ex_us": "EFA VEA SCHF VXUS ACWX",
    "emerging_broad": "EEM VWO SCHE",
    "treasuries": "SHY IEI IEF TLH TLT VGSH VGIT VGLT GOVT",
    "investment_grade_credit": "LQD VCIT VCSH IGIB",
    "high_yield": "HYG JNK",
    "aggregate_bonds": "AGG BND SCHZ",
    "tips": "TIP SCHP STIP",
    "broad_commodities": "DBC GSG",
    "agriculture": "DBA CORN WEAT SOYB",
    "base_metals": "CPER DBB",
    "currencies": "FXE FXY FXB FXA FXC FXF UUP",
}
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def groups_table():
    rows = [(t, g) for g, ts in GROUPS.items() for t in ts.split()]
    df = pd.DataFrame(rows, columns=["ticker", "group"])
    dup = df[df["ticker"].duplicated(keep=False)]
    assert dup.empty, f"ticker listed in more than one group: {sorted(set(dup['ticker']))}"
    return df


def clean_and_save(raw, out_path, max_missing=0.05):
    """raw: yfinance frame with (Ticker, Price) columns. Drops tickers with no data, reports the ones the
    backtester will drop for missing history (>5%), and writes the 3-header-row csv."""
    if raw.columns.names[0] != "Ticker":
        raw = raw.swaplevel(axis=1)
    raw.columns.names = ["Ticker", "Price"]
    raw.index.name = "Date"
    raw = raw.sort_index(axis=1).dropna(how="all")
    tickers = sorted(set(raw.columns.get_level_values(0)))
    keep, none, short = [], [], []
    for t in tickers:
        if t not in raw.columns.get_level_values(0) or "Close" not in raw[t].columns:
            none.append(t); continue
        miss = raw[t]["Close"].isna().mean()
        if raw[t]["Close"].notna().sum() == 0:
            none.append(t)
        else:
            keep.append(t)
            if miss > max_missing:
                short.append((t, miss, raw[t]["Close"].first_valid_index().date()))
    cols = pd.MultiIndex.from_tuples([(t, f) for t in keep for f in FIELDS if f in raw[t].columns],
                                     names=["Ticker", "Price"])
    out = raw.reindex(columns=cols)
    out.to_csv(out_path)
    return out, none, short


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="etf_prices.csv")
    ap.add_argument("--groups-out", default="groups_etf.csv")
    ap.add_argument("--start", default="2012-01-01",
                    help="earlier = more history, but funds launched later than ~5%% of the span after the "
                         "start are dropped by the backtester's missing-data rule")
    ap.add_argument("--end", default=None)
    ap.add_argument("--groups-only", action="store_true")
    ap.add_argument("--batch", type=int, default=40, help="tickers per yfinance request")
    args = ap.parse_args()

    g = groups_table()
    g.to_csv(args.groups_out, index=False)
    n_pairs = sum(len(list(combinations(ts.split(), 2))) for ts in GROUPS.values())
    print(f"{len(g)} tickers in {len(GROUPS)} groups -> {n_pairs} within-group pairs "
          f"(vs {len(g) * (len(g) - 1) // 2} if every pair were tested). Groups -> {args.groups_out}")
    if args.groups_only:
        return

    import yfinance as yf                                   # imported lazily so --groups-only needs no network
    tickers = sorted(g["ticker"])
    parts = []
    for i in range(0, len(tickers), args.batch):
        chunk = tickers[i:i + args.batch]
        print(f"Downloading {len(chunk)} tickers ({i + 1}-{i + len(chunk)} of {len(tickers)}) ...")
        d = yf.download(chunk, start=args.start, end=args.end, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)
        if len(chunk) == 1:                                  # single ticker comes back flat
            d.columns = pd.MultiIndex.from_product([chunk, d.columns])
        parts.append(d)
    raw = pd.concat(parts, axis=1)
    out, none, short = clean_and_save(raw, args.out)

    n_t = len(set(out.columns.get_level_values(0)))
    print(f"\nSaved {args.out}: {n_t} tickers, {len(out)} days, {out.index[0].date()} -> {out.index[-1].date()}")
    if none:
        print(f"No data returned for {len(none)} tickers (renamed/delisted/rate-limited?): {none}")
    if short:
        print("These tickers have >5% missing history (launched later) and will be DROPPED by the backtester; "
              "use a later --start or accept the loss:")
        for t, m, first in short:
            print(f"   {t:<6s} first price {first}  ({100 * m:.0f}% missing)")
    print("\nNext:\n  python backtest_rolling_rescreen.py " + args.out + " --groups-csv " + args.groups_out +
          " --min-formation-trades 0 --min-edge-ratio 2 --candidates 50 --select 15")


if __name__ == "__main__":
    main()
