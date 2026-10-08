#!/usr/bin/env python3
"""Plot hedged spreads for the frozen pairs-trading strategy.

For each pair A/B:
    A_t = alpha + beta * B_t + error_t
    spread_t = A_t - beta * B_t - alpha

Defaults:
    Pairs: CBOE/IHG, CBOE/HLT, LLY/PCAR, ED/LMT
    Hedge fit cutoff: 2026-04-13
    Hedge fit: last 1250 overlapping daily observations
    Rolling window: 60 days (visualisation only)

Outputs one PNG per pair, a combined PNG, and CSVs containing the
hedged spread and rolling statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


PAIRS = [
    ("CBOE", "IHG"),
    ("CBOE", "HLT"),
    ("LLY", "PCAR"),
    ("ED", "LMT"),
]

DEFAULT_RAW_DATA = "raw_data.csv"
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_CUTOFF = "2026-04-13"
DEFAULT_FIT_DAYS = 1250
DEFAULT_ROLLING_WINDOW = 60
DEFAULT_SIGMA = 3.0
DEFAULT_OUTPUT_DIR = "pairs_research/frozen_strategy/hedge_plots"


def load_daily_close(path: str, start_date: str, cutoff: str) -> pd.DataFrame:
    """Load the project's yfinance-style MultiIndex CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = pd.read_csv(path, header=None, low_memory=False)
    if raw.shape[0] < 4:
        raise ValueError("Expected the project's 4-row+ yfinance CSV format.")

    ticker_row = raw.iloc[0].astype(str).tolist()
    field_row = raw.iloc[1].astype(str).tolist()
    dates = pd.to_datetime(raw.iloc[3:, 0], errors="coerce")

    if dates.notna().sum() == 0:
        dates = pd.Series(
            pd.bdate_range(start=start_date, periods=raw.shape[0] - 3),
            index=raw.index[3:],
        )

    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    out.index.name = "Date"

    active_ticker = None
    for j in range(1, raw.shape[1]):
        ticker = ticker_row[j].strip()
        field = field_row[j].strip().lower()

        if ticker and ticker.lower() != "nan":
            active_ticker = ticker

        if active_ticker and field == "close":
            out[active_ticker] = pd.to_numeric(
                raw.iloc[3:, j], errors="coerce"
            ).to_numpy()

    out = out.loc[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.loc[out.index <= pd.Timestamp(cutoff)]


def fit_hedge(a: pd.Series, b: pd.Series, fit_days: int):
    """Fit A = alpha + beta*B using the last fit_days observations."""
    x = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    if len(x) < fit_days:
        raise ValueError(
            f"Only {len(x)} overlapping observations; need {fit_days}."
        )

    fit = x.iloc[-fit_days:]
    X = sm.add_constant(fit["B"].to_numpy())
    model = sm.OLS(fit["A"].to_numpy(), X).fit()

    alpha = float(model.params[0])
    beta = float(model.params[1])
    return beta, alpha


def make_spread(prices: pd.DataFrame, a: str, b: str,
                fit_days: int, rolling_window: int):
    if a not in prices.columns or b not in prices.columns:
        missing = [t for t in (a, b) if t not in prices.columns]
        raise ValueError(f"Missing ticker(s): {missing}")

    df = prices[[a, b]].dropna().copy()
    beta, alpha = fit_hedge(df[a], df[b], fit_days)

    df["hedge_leg"] = beta * df[b]
    df["spread"] = df[a] - df["hedge_leg"] - alpha
    df["spread"]/=df[a]

    df["rolling_mean"] = df["spread"].rolling(
        rolling_window, min_periods=rolling_window
    ).mean()
    df["rolling_std"] = df["spread"].rolling(
        rolling_window, min_periods=rolling_window
    ).std()
    df["z_score"] = (
        (df["spread"] - df["rolling_mean"]) / df["rolling_std"]
    )

    return df, beta, alpha


def plot_pair(df, a, b, beta, alpha, rolling_window, sigma, path):
    upper = df["rolling_mean"] + sigma * df["rolling_std"]
    lower = df["rolling_mean"] - sigma * df["rolling_std"]

    fig, ax = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.35]}
    )

    ax[0].plot(df.index, df[a], label=a, linewidth=1.3)
    ax[0].plot(df.index, df["hedge_leg"],
               label=f"{beta:.4f} × {b}", linewidth=1.3)
    ax[0].set_title(f"{a}/{b}: Price vs Hedged Leg")
    ax[0].set_ylabel("Price")
    ax[0].grid(alpha=0.25)
    ax[0].legend()

    ax[1].plot(df.index, df["spread"], label="Hedged spread", linewidth=1.4)
    ax[1].plot(df.index, df["rolling_mean"],
               label=f"{rolling_window}-day mean", linewidth=1.1)
    ax[1].plot(df.index, upper, "--", label=f"+{sigma:.0f}σ", linewidth=0.9)
    ax[1].plot(df.index, lower, "--", label=f"-{sigma:.0f}σ", linewidth=0.9)
    ax[1].axhline(0, linewidth=0.8)
    ax[1].set_title(
        f"Hedged Spread = {a} - ({beta:.4f} × {b}) - {alpha:.4f}"
    )
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Spread")
    ax[1].grid(alpha=0.25)
    ax[1].legend()

    fig.suptitle(f"{a}/{b} Hedge", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_combined(results, path, rolling_window, sigma):
    fig, axes = plt.subplots(
        len(results), 1, figsize=(14, 3.2 * len(results)), sharex=False
    )
    axes = np.atleast_1d(axes)

    for ax, (name, df, beta, _) in zip(axes, results):
        upper = df["rolling_mean"] + sigma * df["rolling_std"]
        lower = df["rolling_mean"] - sigma * df["rolling_std"]

        ax.plot(df.index, df["spread"], label="Spread", linewidth=1.3)
        ax.plot(df.index, df["rolling_mean"],
                label=f"{rolling_window}d mean", linewidth=1.0)
        ax.plot(df.index, upper, "--", label=f"+{sigma:.0f}σ", linewidth=0.8)
        ax.plot(df.index, lower, "--", label=f"-{sigma:.0f}σ", linewidth=0.8)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"{name} | beta={beta:.6f}")
        ax.set_ylabel("Spread")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Date")
    fig.suptitle("Hedged Spreads — Frozen Pairs Strategy", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-data", default=DEFAULT_RAW_DATA)
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--fit-days", type=int, default=DEFAULT_FIT_DAYS)
    p.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    if args.fit_days < 50:
        raise ValueError("--fit-days must be at least 50")
    if args.rolling_window < 5:
        raise ValueError("--rolling-window must be at least 5")
    if args.sigma <= 0:
        raise ValueError("--sigma must be positive")

    prices = load_daily_close(
        args.raw_data, args.start_date, args.cutoff
    )
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for a, b in PAIRS:
        df, beta, alpha = make_spread(
            prices, a, b, args.fit_days, args.rolling_window
        )
        results.append((f"{a}/{b}", df, beta, alpha))

        png = outdir / f"{a}_{b}_hedge.png"
        csv = outdir / f"{a}_{b}_spread.csv"

        plot_pair(
            df, a, b, beta, alpha,
            args.rolling_window, args.sigma, png
        )

        df[[
            a, b, "hedge_leg", "spread",
            "rolling_mean", "rolling_std", "z_score"
        ]].to_csv(csv)

        print(
            f"{a}/{b}: beta={beta:.6f}, alpha={alpha:.6f}, "
            f"observations={len(df):,}"
        )
        print(f"  plot: {png}")
        print(f"  data: {csv}")

    combined = outdir / "all_hedged_spreads.png"
    plot_combined(
        results, combined,
        args.rolling_window, args.sigma
    )

    print(f"\nCombined plot: {combined.resolve()}")


if __name__ == "__main__":
    main()
