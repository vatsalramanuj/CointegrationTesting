"""
Phase 0 - minimal single-file version: download daily prices via yfinance,
validate completeness, clean/align them, and make a couple of basic plots.

Usage:
    pip install yfinance pandas matplotlib
    python download_stock_prices.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# --- Universe, organized into named sector clusters -----------------------
# Grouping matters for downstream analysis: the screening scripts
# (screen_pairs_train_test.py, backtest_rolling_rescreen.py, study_metric_predictiveness.py) test
# every pair in whatever universe it's given, and the number of pairs grows
# as n*(n-1)/2 — so testing one giant flat list of 100+ tickers makes the
# Bonferroni correction punishingly strict (alpha / ~5000 pairs is nearly
# impossible to clear) and mixes pairs with no plausible shared economic
# driver into the same multiple-testing pool as pairs that do. Prefer
# testing ONE cluster at a time (backtest_rolling_rescreen.py and
# study_metric_predictiveness.py take --groups-csv to restrict testing to
# same-group pairs) rather than the full union below.
CLUSTERS = {
    "homebuilders": ["DHI", "LEN", "PHM", "NVR", "TOL", "KBH"],
    "railroads": ["UNP", "CSX", "NSC"],
    "managed_care": ["UNH", "ELV", "HUM", "CNC", "MOH"],
    "exchanges": ["CME", "ICE", "NDAQ", "CBOE"],
    "payments": ["V", "MA", "PYPL", "FIS", "GPN"],
    "telecom": ["VZ", "T", "TMUS"],
    "regulated_utilities": ["DUK", "SO", "AEP", "EXC", "XEL", "ED"],
    "waste_management": ["WM", "RSG"],
    "parcel_logistics": ["UPS", "FDX"],
    "defense": ["LMT", "RTX", "NOC", "GD", "HII"],
    "heavy_machinery": ["CAT", "DE", "CMI", "PCAR"],
    "chemicals": ["DOW", "LYB", "EMN", "CE", "DD"],
    "consumer_staples": ["KO", "PEP", "MDLZ", "KHC", "GIS"],
    "hotels": ["MAR", "HLT", "H", "WH", "IHG"],
    "casinos": ["LVS", "WYNN", "MGM", "CZR"],
    "insurance": ["PGR", "ALL", "TRV", "CB", "AIG"],
    "pharma": ["LLY", "JNJ", "MRK", "PFE", "ABBV", "BMY"],
    "gold_miners": ["NEM", "GOLD", "AEM", "KGC"],
    "copper_diversified_miners": ["FCX", "SCCO", "TECK", "BHP", "RIO"],
    "steel": ["NUE", "STLD"],
    # "X" (US Steel) removed: acquired by Nippon Steel and delisted, so
    # yfinance can no longer return its history under this symbol — it was
    # silently downloading as an entirely-empty column. See
    # check_download_completeness() below, which would now catch this
    # automatically instead of failing silently downstream.

    # --- New clusters added for more pair candidates ---
    "money_center_regional_banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC"],
    "reits_retail_industrial": ["SPG", "O", "PLD", "PSA", "AVB"],
    "semiconductors": ["NVDA", "AMD", "INTC", "TXN", "AVGO", "QCOM"],
    "airlines": ["DAL", "UAL", "AAL", "LUV"],
}

ETF_SANITY_PAIRS = [("GLD", "GDX")]
ETF_TICKERS = sorted({t for pair in ETF_SANITY_PAIRS for t in pair})

ETF_PREDICT_PAIRS = [("FANG", "VLO")]
ETF_PREDICT_TICKERS = sorted({t for pair in ETF_PREDICT_PAIRS for t in pair})

# Union of every cluster, for a single efficient batch download. Downstream
# analysis should still subset this via --tickers rather than testing the
# whole union at once (see the comment on CLUSTERS above).
TICKERS = sorted(set().union(*CLUSTERS.values()) | set(ETF_TICKERS))
START_DATE = "2015-01-01"


def check_download_completeness(raw: pd.DataFrame, tickers: list, max_missing_frac: float = 0.20):
    """
    Look at each requested ticker's downloaded 'Close' series and flag any
    that came back fully or mostly empty — the exact failure mode that
    silently produced a 100%-empty 'X' column in the past (a delisted /
    unavailable ticker still gets a column in the MultiIndex result, just
    with no data in it). Returns (ok_tickers, problem_tickers) so the
    caller can decide what to do rather than silently shipping a broken
    ticker downstream.
    """
    ok_tickers, problem_tickers = [], []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            problem_tickers.append((t, "not present in download result at all"))
            continue
        close = raw[t]["Close"]
        missing_frac = close.isna().mean()
        if missing_frac >= 0.999:
            problem_tickers.append((t, "100% missing — likely delisted, renamed, or an invalid symbol"))
        elif missing_frac > max_missing_frac:
            problem_tickers.append((t, f"{missing_frac:.0%} missing — partial data, check listing date"))
        else:
            ok_tickers.append(t)

    if problem_tickers:
        print(f"\nWarning: {len(problem_tickers)} of {len(tickers)} ticker(s) have incomplete/failed "
              f"downloads and will be EXCLUDED from raw_data.csv:")
        for t, reason in problem_tickers:
            print(f"  {t}: {reason}")
        print(
            "If a ticker here is one you expect to have data, check its current symbol on "
            "Yahoo Finance directly (delistings, mergers, and symbol changes are the usual cause)."
        )
    return ok_tickers, [t for t, _ in problem_tickers]


def download_data(tickers, start):
    """Download adjusted daily closes for a list of tickers, validate
    completeness, and save only the tickers that actually have usable data."""
    print(f"Downloading {len(tickers)} tickers from {start}...")
    raw = yf.download(
        tickers, start=start, auto_adjust=True, group_by="ticker",
        threads=True, progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        ok_tickers, dropped_tickers = check_download_completeness(raw, tickers)
        if not ok_tickers:
            raise RuntimeError(
                "No tickers had usable data after the download — check your network connection "
                "and ticker list before re-running."
            )

        # Save raw_data.csv with ONLY the tickers that actually have usable
        # data, so downstream scripts never see a 100%-empty column again.
        # index=True (the default) keeps the real Date index — previously
        # this was saved with index=False, which meant downstream scripts
        # had to reconstruct approximate business-day dates from scratch.
        # Real dates are always more accurate than that reconstruction.
        df = raw[ok_tickers].copy()
        df.to_csv("raw_data_unseen.csv")
        closes = pd.DataFrame({t: raw[t]["Close"] for t in ok_tickers})
        volumes = pd.DataFrame({t: raw[t]["Volume"] for t in ok_tickers})
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    return closes.sort_index(), volumes.sort_index()


def clean_and_align(df, max_missing_frac=0.05, ffill_limit=2):
    """Drop tickers with too much missing data, fill small gaps, align dates."""
    missing_frac = df.isna().mean()
    keep = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    dropped = sorted(set(df.columns) - set(keep))
    if dropped:
        print(f"Dropping tickers with too much missing data: {dropped}")
    cleaned = df[keep].ffill(limit=ffill_limit).dropna(how="any")
    print(f"Aligned panel: {len(cleaned)} days x {cleaned.shape[1]} tickers")
    return cleaned


def plot_price_levels(df, tickers):
    """Raw price levels for a sample of tickers."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in tickers:
        ax.plot(df.index, df[t], label=t, linewidth=1.2)
    ax.set_title("Adjusted close price levels")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_normalized_pair(df, pair):
    """Normalize both legs of a pair to 1.0 at the start and plot together."""
    a, b = pair
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[a] / df[a].iloc[0], label=a, linewidth=1.2)
    ax.plot(df.index, df[b] / df[b].iloc[0], label=b, linewidth=1.2)
    ax.set_title(f"Normalized price: {a} vs {b}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_trade_volume(df, tickers):
    """Plot daily trading volume for a sample of tickers as bar charts,
    one subplot per ticker (volume scales vary a lot between names, so a
    single overlaid chart would be dominated by the highest-volume ticker)."""
    fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 2.2 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, t in zip(axes, tickers):
        ax.bar(df.index, df[t], width=1.0, color="steelblue")
        ax.set_ylabel(t, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title("Daily trading volume")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()

def plot_prices_volumes(cleaned_prices, cleaned_volumes, tickers):
    fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 2.5 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]  # subplots() returns a bare Axes, not a list, when there's only 1
    for ax, t in zip(axes, tickers):
        ax.bar(cleaned_volumes.index, cleaned_volumes[t], width=1.0, color="steelblue", alpha=0.4)
        ax.set_ylabel(f"{t} volume", fontsize=9, color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax.grid(alpha=0.3)

        price_ax = ax.twinx()  # separate y-axis so price isn't crushed by volume's scale
        price_ax.plot(cleaned_prices.index, cleaned_prices[t], label=t, linewidth=1.2, color="darkorange")
        price_ax.set_ylabel("price ($)", fontsize=9, color="darkorange")
        price_ax.tick_params(axis="y", labelcolor="darkorange")

    axes[0].set_title("Price vs. trading volume")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()


if __name__ == "__main__":
    raw, volumes = download_data(TICKERS, START_DATE)
    clean_prices = clean_and_align(raw)
    clean_volumes = clean_and_align(volumes)

    # plot_trade_volume(clean_volumes, list(CLUSTERS["homebuilders"])[:3])

    # plot_price_levels(clean_prices, CLUSTERS["homebuilders"])

    plot_prices_volumes(clean_prices, clean_volumes, CLUSTERS["homebuilders"][:5])

    for pair in ETF_PREDICT_PAIRS:
        if pair[0] in clean_prices.columns and pair[1] in clean_prices.columns:
            plot_normalized_pair(clean_prices, pair)

    plt.show()
