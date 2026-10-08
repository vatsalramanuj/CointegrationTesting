#!/usr/bin/env python3
"""
LOCKED OUT-OF-SAMPLE TEST
=========================

Evaluates the already-frozen revised pairs strategy on a separate unseen CSV.

Frozen strategy used by default:
    Pairs: CBOE/IHG, LLY/PCAR, CBOE/HLT, ED/LMT
    QCOM pairs: excluded
    Entry: z = 3.0 for every pair
    Exit: directional zero-crossing (long exits at z >= 0; short at z <= 0)
    Stop: |z| > 4.0
    Costs: 5 bps per leg turnover, charged on entry/exit
    Allocation: fixed 7-sleeve allocation; only 4 sleeves are deployed,
                 so 42.86% of account capital remains in cash.

IMPORTANT METHODOLOGY
---------------------
The historical CSV may itself contain later dates (e.g. through 2026-09-15).
For the holdout beginning 2026-04-14, ALL historical observations after
2026-04-13 are discarded. They are never used in feature calculation,
position reconstruction, or P&L.

The separate unseen CSV is then appended only for causal continuation:
rolling OLS estimates at date t use observations strictly before t.
No pair selection or parameter tuning is performed on the unseen data.

This script intentionally does NOT use backtest_summary.csv to select pairs,
because the supplied summary was produced using data extending into the
would-be holdout period. Using it for selection would contaminate the test.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_HISTORICAL = "raw_data_old.csv"
DEFAULT_UNSEEN = "raw_data_2022.csv"
DEFAULT_OUTPUT_PREFIX = "pairs_research/frozen_strategy/frozen_strategy_oos"
DEFAULT_START_DATE = "2015-01-01"

INITIAL_CAPITAL = 100_000.0
SLEEVE_COUNT = 4
ENTRY_Z = 3.0
STOP_Z = 4.0
COST_BPS = 5.0
FIT_DAYS = 1250

DEFAULT_PAIRS = [
    ("CBOE", "IHG"),
    ("LLY", "PCAR"),
    ("CBOE", "HLT"),
    ("ED", "LMT"),
]

# DEFAULT_PAIRS = [('CZR', 'DE'), ('DUK', 'WM'), ('AEP', 'MRK'), ('LEN', 'UPS')]

def _clean_columns_from_yf(multi: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(multi.columns, pd.MultiIndex):
        raise ValueError("Not a yfinance MultiIndex CSV")

    l0 = [str(x).strip() for x in multi.columns.get_level_values(0)]
    l1 = [str(x).strip() for x in multi.columns.get_level_values(1)]

    field_level = None
    mask = None
    for field in ("Adj Close", "Close"):
        m0 = np.array([x.lower() == field.lower() for x in l0])
        m1 = np.array([x.lower() == field.lower() for x in l1])
        if m0.any():
            field_level, mask = 0, m0
            break
        if m1.any():
            field_level, mask = 1, m1
            break

    if field_level is None:
        raise ValueError("Could not find Close or Adj Close in yfinance CSV")

    selected = multi.loc[:, mask].copy()
    tickers = [
        (l1[i] if field_level == 0 else l0[i])
        for i, keep in enumerate(mask) if keep
    ]
    selected.columns = tickers
    selected = selected.apply(pd.to_numeric, errors="coerce")
    selected = selected.loc[:, ~selected.columns.duplicated(keep="first")]

    # The supplied files have three logical header rows:
    # row 1 = ticker, row 2 = field, row 3 = Date, then actual dates.
    date_col = multi.iloc[:, 0].astype(str).str.strip()
    parsed = pd.to_datetime(date_col, errors="coerce", format="mixed")

    if parsed.notna().mean() > 0.90:
        valid = parsed.notna().to_numpy()
        selected = selected.loc[valid].copy()
        selected.index = pd.DatetimeIndex(parsed.loc[valid])
    else:
        selected.index = pd.bdate_range(DEFAULT_START_DATE, periods=len(selected))

    selected = selected[~selected.index.duplicated(keep="last")]
    return selected.sort_index().dropna(axis=1, how="all")


def load_prices(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    try:
        multi = pd.read_csv(path, header=[0, 1], low_memory=False)
        if isinstance(multi.columns, pd.MultiIndex):
            prices = _clean_columns_from_yf(multi)
            if len(prices) and pd.api.types.is_datetime64_any_dtype(prices.index):
                return prices
    except Exception:
        pass

    df = pd.read_csv(path, low_memory=False)
    date_col = next(
        (c for c in ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"] if c in df.columns),
        None,
    )
    if date_col is not None:
        idx = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        if idx.notna().mean() < 0.90:
            raise ValueError(f"Could not reliably parse {date_col} in {path}")
        prices = df.drop(columns=[date_col]).copy()
        prices.index = pd.DatetimeIndex(idx)
    else:
        idx = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="mixed")
        if idx.notna().mean() >= 0.90:
            prices = df.iloc[:, 1:].copy()
            prices.index = pd.DatetimeIndex(idx)
        else:
            prices = df.copy()
            prices.index = pd.bdate_range(DEFAULT_START_DATE, periods=len(df))

    prices.columns = [str(c).strip() for c in prices.columns]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.sort_index().dropna(axis=1, how="all")


def parse_pairs(value: str) -> List[Tuple[str, str]]:
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "/" not in item:
            raise ValueError(f"Invalid pair '{item}'. Use A/B format.")
        a, b = [x.strip().upper() for x in item.split("/", 1)]
        if not a or not b or a == b:
            raise ValueError(f"Invalid pair '{item}'.")
        pairs.append((a, b))
    if not pairs:
        raise ValueError("No pairs supplied.")
    return pairs


def prepare_features(a: pd.Series, b: pd.Series, fit_days: int) -> dict:
    data = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    a = data["A"].astype(float)
    b = data["B"].astype(float)
    w = fit_days
    if len(data) <= w + 5:
        raise ValueError(f"Only {len(data)} aligned observations; need > {w + 5}.")

    # Strictly causal: parameters used at date t are estimated from t-w ... t-1.
    sa = a.rolling(w).sum().shift(1)
    sb = b.rolling(w).sum().shift(1)
    sab = (a * b).rolling(w).sum().shift(1)
    sb2 = (b * b).rolling(w).sum().shift(1)
    sa2 = (a * a).rolling(w).sum().shift(1)
    n = float(w)

    denom = n * sb2 - sb * sb
    beta = (n * sab - sa * sb) / denom
    intercept = sa / n - beta * sb / n

    sse = (
        sa2 - 2 * beta * sab - 2 * intercept * sa
        + beta * beta * sb2 + 2 * beta * intercept * sb
        + n * intercept * intercept
    )
    std = np.sqrt(np.maximum(sse / (w - 1), 0.0))
    spread = a - beta * b - intercept
    z = spread / std

    valid = (
        beta.notna() & intercept.notna() & std.notna()
        & np.isfinite(beta) & np.isfinite(intercept) & np.isfinite(std)
        & (std > 0)
    )
    return {
        "a": a,
        "b": b,
        "beta": beta.where(valid),
        "intercept": intercept.where(valid),
        "spread": spread.where(valid),
        "z": z.where(valid),
    }


def summarize_trade_log(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_closed_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "avg_holding_days": np.nan,
        }
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    return {
        "n_closed_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_win": wins["net_pnl"].mean() if len(wins) else np.nan,
        "avg_loss": losses["net_pnl"].mean() if len(losses) else np.nan,
        "avg_holding_days": (
            pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
        ).dt.days.mean(),
    }


def run_pair(
    feat: dict,
    unseen_start: pd.Timestamp,
    unseen_end: pd.Timestamp,
    sleeve_notional: float,
    entry_z: float,
    stop_z: float,
    cost_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    a = feat["a"]
    b = feat["b"]
    beta = feat["beta"]
    z = feat["z"]
    idx = a.index

    valid_positions = np.flatnonzero(beta.notna().to_numpy())
    if len(valid_positions) == 0:
        raise ValueError("No valid hedge-ratio observations.")
    first_valid = int(valid_positions[0])

    position = 0
    pending_position = 0
    pending_beta = np.nan
    pending_signal_date = None
    pending_signal_z = np.nan
    open_trade = None

    daily = []
    trades = []
    boundary = None

    for t in range(first_valid, len(idx)):
        date = idx[t]
        px_a = float(a.iloc[t])
        px_b = float(b.iloc[t])
        gross_today = 0.0
        cost_today = 0.0

        # P&L from yesterday's executed position to today's close.
        if position != 0 and open_trade is not None:
            gross_today = (
                open_trade["units_a"] * (px_a - open_trade["prev_a"])
                + open_trade["units_b"] * (px_b - open_trade["prev_b"])
            )

        # Execute yesterday's signal at today's close.
        if pending_position != position:
            if position != 0 and open_trade is not None:
                turnover = abs(open_trade["units_a"]) * px_a + abs(open_trade["units_b"]) * px_b
                exit_cost = turnover * cost_bps / 10_000.0
                cost_today += exit_cost
                trades.append({
                    "entry_signal_date": open_trade["entry_signal_date"],
                    "entry_date": open_trade["entry_date"],
                    "exit_signal_date": pending_signal_date,
                    "exit_date": date,
                    "direction": "LONG_SPREAD" if position > 0 else "SHORT_SPREAD",
                    "beta": open_trade["beta"],
                    "entry_z": open_trade["entry_z"],
                    "exit_signal_z": pending_signal_z,
                    "entry_a": open_trade["entry_a"],
                    "entry_b": open_trade["entry_b"],
                    "exit_a": px_a,
                    "exit_b": px_b,
                    "units_a": open_trade["units_a"],
                    "units_b": open_trade["units_b"],
                    "entry_cost": open_trade["entry_cost"],
                    "exit_cost": exit_cost,
                    "gross_pnl": open_trade["gross_pnl_accum"] + gross_today,
                    "net_pnl": open_trade["gross_pnl_accum"] + gross_today - open_trade["entry_cost"] - exit_cost,
                    "held_days": (date - open_trade["entry_date"]).days,
                    "started_before_unseen": bool(open_trade["entry_date"] < unseen_start),
                })
                open_trade = None
                position = 0

            if pending_position != 0:
                trade_beta = float(pending_beta)
                denom = px_a + abs(trade_beta) * px_b
                if np.isfinite(denom) and denom > 0:
                    units_a_abs = sleeve_notional / denom
                    units_a = pending_position * units_a_abs
                    units_b = -pending_position * trade_beta * units_a_abs
                    entry_turnover = abs(units_a) * px_a + abs(units_b) * px_b
                    entry_cost = entry_turnover * cost_bps / 10_000.0
                    cost_today += entry_cost
                    open_trade = {
                        "entry_signal_date": pending_signal_date,
                        "entry_date": date,
                        "beta": trade_beta,
                        "entry_z": pending_signal_z,
                        "entry_a": px_a,
                        "entry_b": px_b,
                        "units_a": float(units_a),
                        "units_b": float(units_b),
                        "entry_cost": entry_cost,
                        "prev_a": px_a,
                        "prev_b": px_b,
                        "gross_pnl_accum": 0.0,
                    }
                    position = int(pending_position)

        z_t = float(z.iloc[t]) if pd.notna(z.iloc[t]) else np.nan
        desired = position
        if np.isfinite(z_t):
            if position == 0:
                if z_t > entry_z:
                    desired = -1
                elif z_t < -entry_z:
                    desired = 1
                else:
                    desired = 0
            else:
                # Directional exit: close on the appropriate zero crossing.
                crossed_mean = (position > 0 and z_t >= 0) or (position < 0 and z_t <= 0)
                stopped = abs(z_t) > stop_z
                if crossed_mean or stopped:
                    desired = 0

        if desired != position:
            pending_position = int(desired)
            pending_beta = float(beta.iloc[t]) if pd.notna(beta.iloc[t]) else np.nan
            pending_signal_date = date
            pending_signal_z = z_t
        else:
            pending_position = position
            pending_beta = np.nan
            pending_signal_date = None
            pending_signal_z = np.nan

        if open_trade is not None:
            open_trade["gross_pnl_accum"] += gross_today
            open_trade["prev_a"] = px_a
            open_trade["prev_b"] = px_b

        if date >= unseen_start:
            if boundary is None:
                boundary = {
                    "date": date,
                    "position": int(position),
                    "zscore": z_t,
                    "beta": float(beta.iloc[t]) if pd.notna(beta.iloc[t]) else np.nan,
                    "pending_position": int(pending_position),
                }
            daily.append({
                "date": date,
                "gross_pnl": gross_today,
                "cost": cost_today,
                "net_pnl": gross_today - cost_today,
                "position": int(position),
                "zscore": z_t,
                "beta": float(beta.iloc[t]) if pd.notna(beta.iloc[t]) else np.nan,
            })

    daily_df = pd.DataFrame(daily).set_index("date")

    open_state = None
    if open_trade is not None:
        final_date = idx[-1]
        final_a = float(a.iloc[-1])
        final_b = float(b.iloc[-1])
        unrealized = (
            open_trade["units_a"] * (final_a - open_trade["entry_a"])
            + open_trade["units_b"] * (final_b - open_trade["entry_b"])
        )
        open_state = {
            "entry_signal_date": open_trade["entry_signal_date"],
            "entry_date": open_trade["entry_date"],
            "as_of_date": final_date,
            "direction": "LONG_SPREAD" if position > 0 else "SHORT_SPREAD",
            "beta": open_trade["beta"],
            "entry_z": open_trade["entry_z"],
            "last_z": float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else np.nan,
            "units_a": open_trade["units_a"],
            "units_b": open_trade["units_b"],
            "entry_cost": open_trade["entry_cost"],
            "unrealized_gross_pnl": unrealized,
            "started_before_unseen": bool(open_trade["entry_date"] < unseen_start),
        }

    return daily_df, pd.DataFrame(trades), {"boundary": boundary, "open": open_state}


def sharpe(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * math.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1).min())


def make_period_returns(equity: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = equity.pct_change().fillna(0.0)
    monthly = ((1 + r).groupby(r.index.to_period("M")).prod() - 1).to_frame("return")
    annual = ((1 + r).groupby(r.index.to_period("Y")).prod() - 1).to_frame("return")
    monthly.index = monthly.index.astype(str)
    annual.index = annual.index.astype(str)
    return annual, monthly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-csv", default=DEFAULT_HISTORICAL)
    parser.add_argument("--unseen-csv", default=DEFAULT_UNSEEN)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument(
        "--pairs",
        default=",".join(f"{a}/{b}" for a, b in DEFAULT_PAIRS),
        help="Comma-separated frozen pairs, e.g. CBOE/IHG,LLY/PCAR,CBOE/HLT,ED/LMT",
    )
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)

    print(f"Loading historical prices from {args.historical_csv} ...")
    hist_all = load_prices(args.historical_csv)
    print(f"Loaded {len(hist_all):,} historical observations; latest raw date = {hist_all.index.max().date()}")

    print(f"Loading LOCKED unseen prices from {args.unseen_csv} ...")
    unseen = load_prices(args.unseen_csv)
    unseen_start = unseen.index.min()
    unseen_end = unseen.index.max()
    print(f"Loaded {len(unseen):,} unseen observations; {unseen_start.date()} to {unseen_end.date()}")

    # Crucial fix: the historical file may overlap the holdout. Keep only data
    # strictly before the unseen start. Do NOT use later historical observations.
    hist = hist_all.loc[hist_all.index < unseen_start].copy()
    excluded = len(hist_all) - len(hist)
    if excluded:
        print(f"Excluding {excluded:,} historical rows on/after unseen start ({unseen_start.date()}).")
    if hist.empty:
        raise ValueError("No historical observations remain before the unseen start.")

    overlap = hist.index.intersection(unseen.index)
    if len(overlap):
        raise ValueError("Unexpected overlap remains after historical cutoff.")

    needed_tickers = sorted({x for pair in pairs for x in pair})
    missing_hist = [t for t in needed_tickers if t not in hist.columns]
    missing_unseen = [t for t in needed_tickers if t not in unseen.columns]
    if missing_hist or missing_unseen:
        raise ValueError(
            f"Missing required tickers. Historical: {missing_hist}; unseen: {missing_unseen}"
        )

    common = sorted(set(needed_tickers) & set(hist.columns) & set(unseen.columns))
    combined = pd.concat([hist[common], unseen[common]], axis=0).sort_index()

    print("\nFROZEN STRATEGY")
    print("  Pairs:")
    for a, b in pairs:
        print(f"    {a}/{b}")
    print(f"  Entry z:      {ENTRY_Z}")
    print("  Exit:         directional zero-crossing")
    print(f"  Stop z:       {STOP_Z}")
    print(f"  Costs:        {COST_BPS:.1f} bps")
    print(f"  Allocation:   {len(pairs)}/{SLEEVE_COUNT} sleeves ({1-len(pairs)/SLEEVE_COUNT:.2%} cash)")

    sleeve = args.initial_capital / SLEEVE_COUNT
    feature_map: Dict[str, dict] = {}
    print("\nPrecomputing causal features ...")
    for a, b in pairs:
        pair = f"{a}/{b}"
        feature_map[pair] = prepare_features(combined[a], combined[b], FIT_DAYS)

    pair_daily = []
    pair_trades = []
    pair_stats = []
    boundaries = []
    open_positions = []

    for a, b in pairs:
        pair = f"{a}/{b}"
        daily, trades, state = run_pair(
            feature_map[pair], unseen_start, unseen_end, sleeve,
            ENTRY_Z, STOP_Z, COST_BPS,
        )
        daily = daily.copy()
        daily["pair"] = pair
        pair_daily.append(daily)

        if not trades.empty:
            trades = trades.copy()
            trades["pair"] = pair
            pair_trades.append(trades)

        if state["boundary"] is not None:
            boundaries.append({"pair": pair, **state["boundary"]})
        if state["open"] is not None:
            open_positions.append({"pair": pair, **state["open"]})

        gross = float(daily["gross_pnl"].sum())
        cost = float(daily["cost"].sum())
        net = float(daily["net_pnl"].sum())
        pair_stats.append({
            "pair": pair,
            "gross_pnl": gross,
            "cost": cost,
            "net_pnl": net,
            "net_sharpe": sharpe(daily["net_pnl"] / sleeve),
            "gross_sharpe": sharpe(daily["gross_pnl"] / sleeve),
            "n_closed_trades": len(trades),
            "win_rate": (trades["net_pnl"] > 0).mean() if len(trades) else np.nan,
        })

    # Align pair daily series by date and sum into the portfolio.
    pair_map = {d["pair"].iloc[0]: d for d in pair_daily}
    portfolio = pd.DataFrame(index=unseen.index)
    portfolio["gross_pnl"] = 0.0
    portfolio["cost"] = 0.0
    portfolio["net_pnl"] = 0.0
    for pair in pair_map:
        d = pair_map[pair].reindex(unseen.index).fillna(0.0)
        portfolio["gross_pnl"] += d["gross_pnl"]
        portfolio["cost"] += d["cost"]
        portfolio["net_pnl"] += d["net_pnl"]

    equity = args.initial_capital + portfolio["net_pnl"].cumsum()
    # Fixed sleeves: only len(pairs)/SLEEVE_COUNT of capital is deployed.
    # Returns are still calculated on full account equity, with the rest in cash.
    daily_returns = portfolio["net_pnl"] / args.initial_capital
    gross_returns = portfolio["gross_pnl"] / args.initial_capital
    annual, monthly = make_period_returns(equity)

    all_trades = pd.concat(pair_trades, ignore_index=True) if pair_trades else pd.DataFrame()
    trade_stats = summarize_trade_log(all_trades)

    summary = {
        "initial_capital": args.initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / args.initial_capital - 1),
        "CAGR": float((equity.iloc[-1] / equity.iloc[0]) ** (365.25 / max((equity.index[-1] - equity.index[0]).days, 1)) - 1),
        "net_sharpe": sharpe(daily_returns),
        "gross_sharpe": sharpe(gross_returns),
        "max_drawdown": max_drawdown(equity),
        "total_gross_pnl": float(portfolio["gross_pnl"].sum()),
        "total_cost": float(portfolio["cost"].sum()),
        "cost_pct_of_gross_abs": float(portfolio["cost"].sum() / abs(portfolio["gross_pnl"].sum())) if abs(portfolio["gross_pnl"].sum()) > 1e-12 else np.nan,
        "n_pairs": len(pairs),
        **trade_stats,
        "best_day_return": float(daily_returns.max()),
        "worst_day_return": float(daily_returns.min()),
        "historical_cutoff": hist.index.max(),
        "unseen_start": unseen_start,
        "unseen_end": unseen_end,
        "cash_fraction": 1 - len(pairs) / SLEEVE_COUNT,
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True) if prefix.parent != Path(".") else None

    pd.DataFrame([summary]).to_csv(f"{prefix}_summary.csv", index=False)
    pd.DataFrame([(f"{a}/{b}") for a, b in pairs], columns=["pair"]).to_csv(f"{prefix}_selection.csv", index=False)
    portfolio.to_csv(f"{prefix}_daily.csv")
    equity.to_csv(f"{prefix}_equity.csv", header=["equity"])
    annual.to_csv(f"{prefix}_annual.csv")
    monthly.to_csv(f"{prefix}_monthly.csv")
    pd.DataFrame(pair_stats).sort_values("net_pnl", ascending=False).to_csv(f"{prefix}_pair_stats.csv", index=False)
    pd.concat(pair_daily).to_csv(f"{prefix}_pair_daily.csv")
    (all_trades if not all_trades.empty else pd.DataFrame()).to_csv(f"{prefix}_trades.csv", index=False)
    pd.DataFrame(boundaries).to_csv(f"{prefix}_boundary_state.csv", index=False)
    pd.DataFrame(open_positions).to_csv(f"{prefix}_open_positions.csv", index=False)

    # Equity / drawdown plots.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity.index, equity.values, linewidth=1.5)
    ax.set_title("Locked Out-of-Sample Equity")
    ax.set_ylabel("Account Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_equity_curve.png", dpi=150)
    plt.close(fig)

    dd = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dd.index, dd.values, linewidth=1.2)
    ax.axhline(0, linewidth=0.7)
    ax.set_title("Locked Out-of-Sample Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_drawdown.png", dpi=150)
    plt.close(fig)

    print("\n=== LOCKED OUT-OF-SAMPLE RESULT ===")
    print(f"Research cutoff: {hist.index.max().date()}")
    print(f"Unseen period:   {unseen_start.date()} to {unseen_end.date()}")
    print(f"Pairs:           {len(pairs)}")
    print(f"Entry z:         {ENTRY_Z:.1f}")
    print("Exit:            directional zero-crossing")
    print(f"Costs:           {COST_BPS:.1f} bps")
    print(f"Initial equity:  ${summary['initial_capital']:,.2f}")
    print(f"Final equity:    ${summary['final_equity']:,.2f}")
    print(f"Total return:    {summary['total_return']:.2%}")
    print(f"CAGR:            {summary['CAGR']:.2%}")
    print(f"Net Sharpe:      {summary['net_sharpe']:.2f}")
    print(f"Gross Sharpe:    {summary['gross_sharpe']:.2f}")
    print(f"Max drawdown:    {summary['max_drawdown']:.2%}")
    print(f"Gross P&L:       ${summary['total_gross_pnl']:,.2f}")
    print(f"Costs:           ${summary['total_cost']:,.2f}")
    print(f"Closed trades:   {summary['n_closed_trades']}")
    print(f"Win rate:        {summary['win_rate']:.2%}" if pd.notna(summary['win_rate']) else "Win rate:        N/A")
    print(f"Average holding: {summary['avg_holding_days']:.1f} days" if pd.notna(summary['avg_holding_days']) else "Average holding: N/A")

    print("\nPair results:")
    print(pd.DataFrame(pair_stats).sort_values("net_pnl", ascending=False).to_string(index=False))
    print("\nAnnual returns:")
    print(annual.to_string())

    print("\nSaved outputs with prefix:", prefix)
    print("NOTE: Historical rows on/after the unseen start were excluded.")
    print("NOTE: backtest_summary.csv was NOT used for selection because it contains results extending into the holdout period.")


if __name__ == "__main__":
    main()
