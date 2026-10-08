#!/usr/bin/env python3
"""
Daily-close walk-forward backtest for the frozen pairs strategy.

Pairs:   CBOE/IHG, CBOE/HLT, LLY/PCAR, ED/LMT
Rules:   Entry |z| > 2.0 (only while |z| <= stop), exit on directional
         zero-crossing, stop |z| > 4.0 (with re-entry lockout), time stop.

Timing (no look-ahead):
    * Signal is computed with data up to and including the close of day t.
    * The trade is filled at the close of day t+1.
    * P&L accrues from the t+1 close onward.

Method:
    * Hedge ratio: OLS on LOG prices over a trailing window, re-estimated every
      `refit_every` days using only data up to the signal date. Beta is only
      refit while flat, and is frozen for the life of a trade.
    * z-score: current spread vs. mean/std of the previous `z_window` spreads
      (the current bar is excluded from its own mean/std).
    * Optional Engle-Granger cointegration gate evaluated at each refit; new
      entries are blocked when the trailing-window p-value is above the cutoff.
    * Sizing: capital / sleeves per pair (default 7 sleeves, 4 used, 3/7 cash).
      Each trade is dollar-hedged: $A = sleeve/(1+|beta|), $B = \vert{}beta\vert{} *$A,
      converted to fixed share counts at the fill price.
    * Costs: `cost_bps` on the traded notional of BOTH legs at entry and exit,
      plus optional annualised borrow cost on the short leg(s).

Results are reported for the specified trading period (default: trailing 500 days)
and split at the research cutoff for in-sample / out-of-sample comparison.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

PAIRS = [("CBOE", "IHG"), ("CBOE", "HLT"), ("LLY", "PCAR"), ("ED", "LMT")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-data", default="raw_data.csv")
    p.add_argument("--days", type=int, default=500,
                   help="Number of trailing trading days to backtest (default: 500). Set to 0 to test all data.")
    p.add_argument("--start", default=None, help="First date on which signals may be generated.")
    p.add_argument("--end", default=None, help="Last date used (inclusive).")
    p.add_argument("--research-cutoff", default="2026-04-13",
                   help="Used only to split in-sample / out-of-sample reporting.")
    p.add_argument("--beta-window", type=int, default=315, help="Trailing days for hedge-ratio fit.")
    p.add_argument("--refit-every", type=int, default=42, help="Days between refits (only while flat).")
    p.add_argument("--z-window", type=int, default=90)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.0)
    p.add_argument("--stop-z", type=float, default=4.0)
    p.add_argument("--max-hold", type=int, default=60, help="Time stop in trading days; 0 disables.")
    p.add_argument("--coint-p", type=float, default=1.0,
                   help="Block new entries if Engle-Granger p-value exceeds this; 1.0 disables.")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Per side, per leg.")
    p.add_argument("--borrow-bps", type=float, default=0.0, help="Annual borrow cost on short leg, bps.")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--sleeves", type=int, default=4)
    p.add_argument("--out-prefix", default="pairs_research/frozen_strategy/frozen_strategy")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_daily_close(path: str) -> pd.DataFrame:
    """Parse the multi-row yfinance CSV (ticker row, field row, blank/date row, data)."""
    raw = pd.read_csv(path, header=None, low_memory=False)
    if raw.shape[0] < 4:
        raise ValueError(f"{path} is not the expected 4-row+ yfinance CSV format.")
    ticker_row = raw.iloc[0].astype(str).tolist()
    field_row = raw.iloc[1].astype(str).tolist()
    date = pd.to_datetime(raw.iloc[3:, 0], errors="coerce")
    out = pd.DataFrame(index=date)
    out.index.name = "Date"
    active = None
    for j in range(1, raw.shape[1]):
        t = ticker_row[j].strip()
        f = field_row[j].strip().lower()
        if t and t != "nan":
            active = t
        if active and f == "close":
            out[active] = pd.to_numeric(raw.iloc[3:, j], errors="coerce").to_numpy()
    out = out.loc[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def fit_beta(la: np.ndarray, lb: np.ndarray) -> float:
    X = sm.add_constant(lb)
    return float(sm.OLS(la, X).fit().params[1])


def coint_pvalue(la: np.ndarray, lb: np.ndarray) -> float:
    try:
        return float(coint(la, lb)[1])
    except Exception:
        return 1.0


def zscore(la: np.ndarray, lb: np.ndarray, i: int, beta: float, window: int) -> float:
    """z at bar i; mean/std use the `window` bars BEFORE i (causal)."""
    if i < window:
        return np.nan
    s = la[i - window:i + 1] - beta * lb[i - window:i + 1]
    hist = s[:-1]
    sd = hist.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float((s[-1] - hist.mean()) / sd)


# ----------------------------------------------------------------------------
# Single-pair simulation
# ----------------------------------------------------------------------------
def backtest_pair(px: pd.DataFrame, a: str, b: str, args, trade_from: pd.Timestamp | None):
    df = px[[a, b]].dropna()
    idx = df.index
    n = len(df)
    A = df[a].to_numpy(float)
    B = df[b].to_numpy(float)
    la, lb = np.log(A), np.log(B)
    sleeve = args.initial_capital / args.sleeves
    cost_rate = args.cost_bps / 1e4
    borrow_daily = args.borrow_bps / 1e4 / 252.0

    gross = np.zeros(n)
    cost = np.zeros(n)
    borrow = np.zeros(n)
    posn = np.zeros(n, dtype=int)
    zs = np.full(n, np.nan)

    pos = 0
    legA = legB = 0.0           # signed shares
    pending = None              # target position to execute at this bar's close
    pending_reason = ""
    locked = False
    beta = None
    gate_ok = True
    last_fit = -10**9
    trade = None
    trades = []

    for i in range(n):
        # 1) P&L earned over (i-1 -> i) by the position held after the previous close
        if i > 0 and pos != 0:
            g = legA * (A[i] - A[i - 1]) + legB * (B[i] - B[i - 1])
            gross[i] = g
            short_notional = max(-legA, 0.0) * A[i - 1] + max(-legB, 0.0) * B[i - 1]
            borrow[i] = short_notional * borrow_daily
            trade["gross"] += g
            trade["cost"] += borrow[i]

        # 2) Execute the order generated at the previous close, at THIS close
        if pending is not None and pending != pos:
            if pos != 0:  # close existing trade
                c = cost_rate * (abs(legA) * A[i] + abs(legB) * B[i])
                cost[i] += c
                trade["cost"] += c
                trade.update(exit_date=idx[i], exit_reason=pending_reason, closed=True)
                trade["net"] = trade["gross"] - trade["cost"]
                trades.append(trade)
                trade = None
                legA = legB = 0.0
            if pending != 0:  # open new trade
                dA = sleeve / (1.0 + abs(beta))
                dB = abs(beta) * dA
                nA, nB = dA / A[i], dB / B[i]
                legA = pending * nA
                legB = -pending * np.sign(beta) * nB
                c = cost_rate * (abs(legA) * A[i] + abs(legB) * B[i])
                cost[i] += c
                trade = dict(pair=f"{a}/{b}", entry_date=idx[i], exit_date=None,
                             direction="Long spread" if pending > 0 else "Short spread",
                             beta=beta, entry_idx=i, gross=0.0, cost=c, net=np.nan,
                             exit_reason="", closed=False)
            pos = pending
        pending = None
        posn[i] = pos

        # 3) Generate a signal from data up to and including close i
        if trade_from is not None and idx[i] < trade_from:
            continue

        if pos == 0 and i >= args.beta_window and (beta is None or i - last_fit >= args.refit_every):
            lo = i - args.beta_window + 1
            beta = fit_beta(la[lo:i + 1], lb[lo:i + 1])
            gate_ok = coint_pvalue(la[lo:i + 1], lb[lo:i + 1]) <= args.coint_p
            last_fit = i
        if beta is None:
            continue

        z = zscore(la, lb, i, beta, args.z_window)
        zs[i] = z
        if not np.isfinite(z):
            continue

        if locked and abs(z) < args.entry_z:
            locked = False

        target, reason = pos, ""
        if pos == 0:
            if abs(z) > args.stop_z:
                locked = True
            elif not locked and gate_ok and abs(z) > args.entry_z:
                target = -1 if z > 0 else 1
        else:
            if (pos > 0 and z >= 0) or (pos < 0 and z <= 0):
                target, reason = 0, "zero-cross"
            elif abs(z) > args.stop_z:
                target, reason, locked = 0, "stop", True
            elif args.max_hold and (i - trade["entry_idx"]) >= args.max_hold:
                target, reason = 0, "time-stop"

        if target != pos and i + 1 < n:
            pending, pending_reason = target, reason

    # Mark any still-open trade (no exit cost charged; flagged as open)
    if trade is not None:
        trade["net"] = trade["gross"] - trade["cost"]
        trade["exit_date"] = idx[-1]
        trade["exit_reason"] = "open at end"
        trades.append(trade)

    net = gross - cost - borrow
    detail = pd.DataFrame({"a": A, "b": B, "z": zs, "position": posn,
                           "gross_pnl": gross, "cost": cost + borrow, "net_pnl": net}, index=idx)
    if trade_from is not None:
        detail = detail.loc[trade_from:]

    tl = pd.DataFrame(trades)
    if not tl.empty:
        tl["days"] = [(d1 - d0).days for d0, d1 in zip(tl["entry_date"], tl["exit_date"])]
        tl = tl.drop(columns=["entry_idx"])
    return detail, tl


# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------
def perf(net: pd.Series, capital: float) -> dict:
    if len(net) < 2:
        return {}
    eq = capital + net.cumsum()
    r = net / capital
    sd = r.std()
    years = max((net.index[-1] - net.index[0]).days / 365.25, 1e-9)
    return {
        "start": net.index[0].date(), "end": net.index[-1].date(), "days": len(net),
        "final_equity": eq.iloc[-1],
        "total_return": eq.iloc[-1] / capital - 1,
        "ann_return": (eq.iloc[-1] / capital) ** (1 / years) - 1 if years >= 0.25 else np.nan,
        "sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "net_pnl": net.sum(),
    }


def print_perf(title: str, d: dict):
    print(f"\n--- {title} ---")
    if not d:
        print("  (not enough data)")
        return
    print(f"  Period:       {d['start']} -> {d['end']} ({d['days']} days)")
    print(f"  Net P&L:      ${d['net_pnl']:,.2f}  (final equity${d['final_equity']:,.2f})")
    print(f"  Total return: {d['total_return']:.2%}")
    ar = d["ann_return"]
    print(f"  Ann. return:  {ar:.2%}" if pd.notna(ar) else "  Ann. return:  n/a (<3 months)")
    print(f"  Sharpe:       {d['sharpe']:.2f}")
    print(f"  Max drawdown: {d['max_drawdown']:.2%}")


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.sleeves < len(PAIRS):
        raise ValueError("sleeves must be >= number of pairs (4).")

    px = load_daily_close(args.raw_data)
    if args.end:
        px = px.loc[:pd.Timestamp(args.end)]
    need = sorted({t for p in PAIRS for t in p})
    missing = [t for t in need if t not in px.columns]
    if missing:
        raise ValueError(f"Missing tickers in {args.raw_data}: {missing}")
    px = px[need].dropna(how="all")

    if args.days and args.days > 0:
        if len(px) > args.days:
            # Retain trailing `days` for backtesting, plus `beta_window` for lookback estimation warmup
            total_needed = args.days + args.beta_window
            px = px.iloc[-total_needed:]
            trade_from = px.index[-args.days]
        else:
            trade_from = pd.Timestamp(args.start) if args.start else None
    else:
        trade_from = pd.Timestamp(args.start) if args.start else None

    cutoff = pd.Timestamp(args.research_cutoff)
    print(f"Loaded daily data. Backtest estimation window start: {px.index.min().date()} -> end: {px.index.max().date()}")
    if trade_from:
        print(f"Active trading backtest period: {trade_from.date()} -> {px.index[-1].date()} ({args.days} trading days)")

    details, trade_logs, rows = {}, [], []
    for a, b in PAIRS:
        d, tl = backtest_pair(px, a, b, args, trade_from)
        key = f"{a}/{b}"
        details[key] = d
        if not tl.empty:
            trade_logs.append(tl)
            closed = tl[tl["closed"]]
            wr = (closed["net"] > 0).mean() if len(closed) else np.nan
            ah = closed["days"].mean() if len(closed) else np.nan
            stops = int((tl["exit_reason"] == "stop").sum())
        else:
            closed, wr, ah, stops = tl, np.nan, np.nan, 0
        rows.append({"pair": key, "net_pnl": d["net_pnl"].sum(), "gross_pnl": d["gross_pnl"].sum(),
                     "cost": d["cost"].sum(), "n_trades": len(tl), "n_closed": len(closed),
                     "win_rate": wr, "avg_hold_days": ah, "stops": stops})
        print(f"  {key}: net ${d['net_pnl'].sum():+,.2f} | trades {len(tl)} | stops {stops}")

    net = pd.concat([d["net_pnl"].rename(k) for k, d in details.items()], axis=1).fillna(0.0)
    total = net.sum(axis=1)
    gross = pd.concat([d["gross_pnl"] for d in details.values()], axis=1).fillna(0.0).sum(axis=1)
    cost = pd.concat([d["cost"] for d in details.values()], axis=1).fillna(0.0).sum(axis=1)
    equity = args.initial_capital + total.cumsum()
    dd = equity / equity.cummax() - 1

    print_perf("FULL SAMPLE", perf(total, args.initial_capital))
    print_perf(f"IN-SAMPLE (<= {cutoff.date()})", perf(total.loc[:cutoff], args.initial_capital))
    print_perf(f"OUT-OF-SAMPLE (> {cutoff.date()})", perf(total.loc[cutoff + pd.Timedelta(days=1):], args.initial_capital))
    print(f"\nGross P&L ${gross.sum():,.2f} Costs${cost.sum():,.2f}")   

    all_trades = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()
    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_pair_summary.csv", index=False)
    all_trades.to_csv(f"{args.out_prefix}_trades.csv", index=False)
    pd.DataFrame({"equity": equity, "drawdown": dd, "gross_pnl": gross, "cost": cost,
                  "net_pnl": total}).to_csv(f"{args.out_prefix}_daily.csv")
    net.to_csv(f"{args.out_prefix}_pair_pnl.csv")
    for k, d in details.items():
        d.to_csv(f"{args.out_prefix}_{k.replace('/', '_')}_detail.csv")

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(equity.index, equity.values)
    ax[0].axvline(cutoff, color="red", ls="--", alpha=0.6, label="research cutoff")
    ax[0].set_title("Pairs strategy - daily equity (next-day-close fills)")
    ax[0].legend()
    ax[0].grid(alpha=0.25)
    ax[1].plot(dd.index, dd.values)
    ax[1].set_title("Drawdown")
    ax[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_equity_curve.png", dpi=150)
    plt.close()
    print(f"\nSaved outputs with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
