"""
Cointegration & Correlation Analysis for a Basket of Tickers
=============================================================

Reads a yfinance-style CSV and computes:
    1. Pearson correlation matrix of returns
    2. Pearson correlation matrix of raw day-over-day price changes
    3. Pairwise Engle-Granger cointegration p-values
    4. Ranked cointegrated pairs with Bonferroni and BH-FDR corrections

The loader is designed for CSVs with either:
    - a Date column + two-row MultiIndex header, or
    - NO Date column, with two header rows such as:

        GDX,GDX,GDX,GDX,GDX,XOM,XOM,XOM,**XOM,**XOM,...
        Open,High,Low,**Close,**Volume,Open,High,Low,**Close,**Volume,...

For a CSV without dates, dates are reconstructed as business days starting
at --start-date. The cointegration calculations themselves only require
aligned observations.

Requirements:
    pip install pandas numpy matplotlib seaborn statsmodels

Usage:
    python explore_correlation_cointegration.py
"""

import argparse
import itertools
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests


# Try to pull a default START_DATE from download_stock_prices.py.
try:
    from download_stock_prices import START_DATE as DEFAULT_START_DATE
except ImportError:
    DEFAULT_START_DATE = None


# ----------------------------------------------------------------------
# 1. LOAD & RESHAPE DATA
# ----------------------------------------------------------------------
def _clean_header(value) -> str:
    """Remove whitespace and literal '*' characters from a header value."""
    if value is None:
        return ""
    return str(value).strip().replace("*", "")


def _peek_header_shape(csv_path: str) -> bool:
    """
    Return True if the CSV appears to have a leading Date/index column.

    For the user's no-date yfinance export, the first two rows look like:
        GDX,GDX,...
        Open,High,Low,Close,Volume,...

    Therefore the first field on row 2 is an OHLCV field and there is no
    leading date column.
    """
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        row1 = next(reader, [])
        row2 = next(reader, [])

    if not row1 or not row2:
        raise ValueError("CSV does not contain the expected two header rows.")

    first_field = _clean_header(row2[0]).lower()

    ohlcv_fields = {
        "open",
        "high",
        "low",
        "close",
        "adj close",
        "volume",
    }

    return first_field not in ohlcv_fields


def _load_no_date_two_header_csv(
    csv_path: str,
    start_date: str,
) -> pd.DataFrame:
    """
    Robustly load the actual no-date/two-header yfinance CSV.

    We deliberately do NOT use pandas MultiIndex/xs() here. The raw CSV can
    contain literal '**' prefixes in some header cells, e.g. '**Close'.
    Reading the first two rows manually makes the field detection explicit
    and avoids accidentally selecting 'Adj Close'.
    """
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        ticker_row = next(reader, [])
        field_row = next(reader, [])

    if not ticker_row or not field_row:
        raise ValueError(
            "Could not read the first two header rows from the CSV."
        )

    if len(ticker_row) != len(field_row):
        raise ValueError(
            "The two CSV header rows have different numbers of columns: "
            f"{len(ticker_row)} vs {len(field_row)}."
        )

    tickers = [_clean_header(x) for x in ticker_row]
    fields = [_clean_header(x) for x in field_row]

    close_indices = [
        i for i, field in enumerate(fields)
        if field.lower() == "close"
    ]

    if not close_indices:
        raise ValueError(
            "Could not find any 'Close' columns in the second header row.\n"
            f"Fields found: {sorted(set(fields))}"
        )

    # A selected Close column must also have a ticker name.
    usable = [
        i for i in close_indices
        if tickers[i] != ""
    ]

    if not usable:
        raise ValueError(
            "Close columns were found, but none has a usable ticker name."
        )

    close_indices = usable
    close_tickers = [tickers[i] for i in close_indices]

    # Read the actual observations. There is no header row here because the
    # first two rows were already consumed conceptually.
    data = pd.read_csv(
        csv_path,
        skiprows=2,
        header=None,
        dtype=str,
        encoding="utf-8-sig",
    )

    if data.empty:
        raise ValueError("The CSV contains no price observations.")

    # Some CSV writers can leave trailing columns. Only use indices that
    # actually exist in the data.
    valid_indices = [
        i for i in close_indices
        if i < data.shape[1]
    ]

    if not valid_indices:
        raise ValueError(
            "The Close columns identified in the header do not exist "
            "in the data rows."
        )

    close_tickers = [
        tickers[i] for i in valid_indices
    ]

    prices = data.iloc[:, valid_indices].copy()
    prices.columns = close_tickers

    # Convert strings such as numeric prices to floats. Bad values become NaN.
    for col in prices.columns:
        prices[col] = pd.to_numeric(
            prices[col].astype(str).str.strip(),
            errors="coerce",
        )

    # Drop completely empty tickers.
    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        raise ValueError(
            "No usable Close price columns were found after numeric "
            "conversion.\n"
            "Check that the CSV contains numeric Close values."
        )

    # The expected CSV has one Close column per ticker. If duplicate ticker
    # names somehow remain, keep the first one rather than silently creating
    # duplicate DataFrame columns.
    if prices.columns.duplicated().any():
        duplicated = prices.columns[
            prices.columns.duplicated()
        ].tolist()

        print(
            "Warning: duplicate ticker names found in Close columns: "
            f"{sorted(set(duplicated))}. Keeping the first occurrence."
        )

        prices = prices.loc[
            :,
            ~prices.columns.duplicated(keep="first")
        ]

    if start_date is None:
        raise ValueError(
            "This CSV has no date column, so --start-date is required "
            "to reconstruct the index.\n"
            "Example: python explore_correlation_cointegration.py --start-date 2015-01-01"
        )

    prices.index = pd.bdate_range(
        start=start_date,
        periods=len(prices),
    )
    prices.index.name = "Date"

    print(
        f"Loaded {prices.shape[1]} tickers and "
        f"{prices.shape[0]} observations."
    )
    print("Using 'Close' prices.")
    print("\nNon-null observations per ticker:")
    print(prices.notna().sum().to_string())

    return prices.sort_index()


def load_price_matrix(
    csv_path: str,
    price_col_hint: str = "Close",
    start_date: str = None,
) -> pd.DataFrame:
    """
    Load a yfinance-style CSV and return:

        index   = Date
        columns = Ticker
        values  = Close prices

    Supported formats:

    A2. Two-row ticker/field header with NO date column.
        This is the format used by the user's raw_data.csv.

    A. MultiIndex header WITH a date/index column.

    B. Single-header columns such as Close_AAPL.

    C. Long format with Date, Ticker, Close.

    D. Generic CSV where numeric columns are already price series.
    """

    # --------------------------------------------------------------
    # Case A2: two header rows, no date column
    # --------------------------------------------------------------
    try:
        has_date_col = _peek_header_shape(csv_path)
    except Exception as exc:
        raise ValueError(
            f"Could not inspect CSV header: {exc}"
        ) from exc

    if not has_date_col:
        return _load_no_date_two_header_csv(
            csv_path,
            start_date=start_date,
        )

    # --------------------------------------------------------------
    # Case A: MultiIndex header WITH date/index column
    # --------------------------------------------------------------
    try:
        df_multi = pd.read_csv(
            csv_path,
            header=[0, 1],
            index_col=0,
            parse_dates=True,
        )

        if isinstance(df_multi.columns, pd.MultiIndex):
            cleaned_columns = [
                (
                    _clean_header(ticker),
                    _clean_header(field),
                )
                for ticker, field in df_multi.columns
            ]

            df_multi.columns = pd.MultiIndex.from_tuples(
                cleaned_columns
            )

            level1 = df_multi.columns.get_level_values(1)

            # Prefer the requested field, but explicitly fall back from
            # Adj Close to Close only if the requested field is absent.
            if price_col_hint in level1:
                price_field = price_col_hint
            elif "Close" in level1:
                price_field = "Close"
            elif "Adj Close" in level1:
                price_field = "Adj Close"
            else:
                price_field = None

            if price_field is not None:
                wide = df_multi.xs(
                    price_field,
                    axis=1,
                    level=1,
                ).copy()

                for col in wide.columns:
                    wide[col] = pd.to_numeric(
                        wide[col],
                        errors="coerce",
                    )

                wide = wide.dropna(
                    axis=1,
                    how="all",
                )

                if not wide.empty:
                    wide.index.name = "Date"
                    return wide.sort_index()

    except Exception:
        # Continue to the other supported formats.
        pass

    # --------------------------------------------------------------
    # Fallback: normal single-header CSV
    # --------------------------------------------------------------
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The CSV contains no observations.")

    # Find date column.
    date_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower() in ("date", "datetime")
        ),
        df.columns[0],
    )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[date_col]
    ).set_index(date_col).sort_index()

    # --------------------------------------------------------------
    # Case C: long format
    # --------------------------------------------------------------
    ticker_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower() == "ticker"
        ),
        None,
    )

    if ticker_col is not None:
        price_field = (
            price_col_hint
            if price_col_hint in df.columns
            else "Close"
        )

        if price_field not in df.columns:
            raise ValueError(
                f"Could not find '{price_col_hint}' or 'Close' "
                "in the long-format CSV."
            )

        wide = df.pivot(
            columns=ticker_col,
            values=price_field,
        )

        return wide.sort_index()

    # --------------------------------------------------------------
    # Case B: columns such as Close_AAPL / Adj Close_MSFT
    # --------------------------------------------------------------
    price_cols = [
        c for c in df.columns
        if str(c).startswith(price_col_hint)
        or str(c).startswith("Close")
        or str(c).startswith("Adj Close")
    ]

    if price_cols:
        wide = df[price_cols].copy()

        wide.columns = [
            str(c).split("_", 1)[-1]
            if "_" in str(c)
            else str(c)
            for c in wide.columns
        ]

        for col in wide.columns:
            wide[col] = pd.to_numeric(
                wide[col],
                errors="coerce",
            )

        return wide.sort_index()

    # --------------------------------------------------------------
    # Final fallback: numeric columns
    # --------------------------------------------------------------
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if not numeric_cols:
        raise ValueError(
            "Could not identify price columns automatically. "
            "Please check the CSV structure."
        )

    return df[numeric_cols].sort_index()


# ----------------------------------------------------------------------
# 2. CORRELATION OF RETURNS
# ----------------------------------------------------------------------
def compute_return_correlation(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Log returns -> Pearson correlation matrix."""
    valid_prices = prices.dropna(
        how="all",
        axis=1,
    )

    log_prices = np.log(valid_prices)
    returns = log_prices.diff().dropna(how="all")

    # Pairwise correlation can handle missing observations. Requiring
    # completely populated columns would unnecessarily discard tickers.
    return returns.corr(method="pearson")


# ----------------------------------------------------------------------
# 2b. CORRELATION OF RAW PRICE CHANGES
# ----------------------------------------------------------------------
def compute_price_change_correlation(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Raw day-over-day price changes:
        Close_t - Close_(t-1)

    This is different from percentage/log-return correlation because it
    operates on the absolute dollar-change scale.
    """
    clean_prices = prices.dropna(
        how="all",
        axis=1,
    )

    price_changes = clean_prices.diff().dropna(
        how="all"
    )

    return price_changes.corr(method="pearson")


# ----------------------------------------------------------------------
# 3. PAIRWISE ENGLE-GRANGER COINTEGRATION
# ----------------------------------------------------------------------
def compute_cointegration_matrix(
    prices: pd.DataFrame,
    significance: float = 0.05,
    min_obs: int = 500,
):
    """
    Run pairwise Engle-Granger cointegration tests.

    IMPORTANT:
    Each pair uses only observations where BOTH tickers are valid.

    This avoids the previous failure mode where:
        prices.dropna(axis=1, how="any")

    discarded an entire ticker because of just one missing value.

    Parameters
    ----------
    prices:
        Wide price matrix.

    significance:
        Alpha used for Bonferroni and BH-FDR corrections.

    min_obs:
        Minimum number of overlapping observations required for a pair.
    """

    if prices.empty:
        raise ValueError("Price matrix is empty.")

    # First filter by individual observation count.
    valid_tickers = [
        col
        for col in prices.columns
        if prices[col].notna().sum() >= min_obs
    ]

    if len(valid_tickers) < 2:
        counts = prices.notna().sum().sort_values()

        raise ValueError(
            f"Only {len(valid_tickers)} ticker(s) have at least "
            f"{min_obs} valid observations. Cannot form pairs.\n\n"
            "Observation counts:\n"
            f"{counts.to_string()}"
        )

    prices = prices[valid_tickers]

    n_pairs = len(
        list(itertools.combinations(valid_tickers, 2))
    )

    print(
        f"Testing {len(valid_tickers)} tickers "
        f"({n_pairs} pairs)..."
    )

    pval_matrix = pd.DataFrame(
        np.ones(
            (
                len(valid_tickers),
                len(valid_tickers),
            )
        ),
        index=valid_tickers,
        columns=valid_tickers,
    )

    results = []

    for a, b in itertools.combinations(
        valid_tickers,
        2,
    ):
        # Pairwise overlap only.
        pair = prices[[a, b]].dropna()

        n_obs = len(pair)

        pvalue = np.nan
        score = np.nan

        if n_obs >= min_obs:
            series_a = pair[a].to_numpy(dtype=float)
            series_b = pair[b].to_numpy(dtype=float)

            # coint() can fail for constant / pathological series.
            try:
                score, pvalue, _ = coint(
                    series_a,
                    series_b,
                    trend="c",
                    autolag="aic",
                )

            except Exception as exc:
                print(
                    f"Warning: coint failed for {a}/{b}: {exc}"
                )

        pval_for_matrix = (
            float(pvalue)
            if pd.notna(pvalue)
            else 1.0
        )

        pval_matrix.loc[a, b] = pval_for_matrix
        pval_matrix.loc[b, a] = pval_for_matrix

        results.append(
            {
                "Ticker A": a,
                "Ticker B": b,
                "n_obs": n_obs,
                "coint_stat": score,
                "coint_pvalue": pvalue,
            }
        )

    # Diagonal is zero purely for heatmap display.
    for ticker in valid_tickers:
        pval_matrix.loc[ticker, ticker] = 0.0

    if not results:
        raise ValueError(
            "No ticker pairs could be tested. "
            "Check the input data and min_obs."
        )

    pairs_df = pd.DataFrame(results)

    # --------------------------------------------------------------
    # Multiple-testing corrections
    # --------------------------------------------------------------
    valid_mask = pairs_df["coint_pvalue"].notna()
    n_tests = int(valid_mask.sum())

    if n_tests == 0:
        raise ValueError(
            "All cointegration tests returned NaN.\n"
            "Possible causes:\n"
            "  - insufficient overlapping data\n"
            "  - constant price series\n"
            "  - invalid price columns"
        )

    raw_pvalues = pairs_df.loc[
        valid_mask,
        "coint_pvalue",
    ].to_numpy()

    # Bonferroni
    pairs_df["pvalue_bonferroni"] = np.nan
    pairs_df["significant_bonferroni"] = False

    bonf_reject, bonf_pvals, _, _ = multipletests(
        raw_pvalues,
        alpha=significance,
        method="bonferroni",
    )

    pairs_df.loc[
        valid_mask,
        "pvalue_bonferroni",
    ] = bonf_pvals

    pairs_df.loc[
        valid_mask,
        "significant_bonferroni",
    ] = bonf_reject

    # Benjamini-Hochberg FDR
    pairs_df["pvalue_fdr_bh"] = np.nan
    pairs_df["significant_fdr_bh"] = False

    fdr_reject, fdr_pvals, _, _ = multipletests(
        raw_pvalues,
        alpha=significance,
        method="fdr_bh",
    )

    pairs_df.loc[
        valid_mask,
        "pvalue_fdr_bh",
    ] = fdr_pvals

    pairs_df.loc[
        valid_mask,
        "significant_fdr_bh",
    ] = fdr_reject

    # Uncorrected significance.
    pairs_df["cointegrated_at_5pct_uncorrected"] = (
        pairs_df["coint_pvalue"] < significance
    )

    # Rank by raw p-value.
    pairs_df = (
        pairs_df
        .sort_values(
            "coint_pvalue",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    print(
        f"\n{n_tests} valid pairs tested."
    )

    print(
        f"At alpha={significance}, expect approximately "
        f"{n_tests * significance:.1f} false positives by chance "
        f"alone if using the raw p-value cutoff."
    )

    return pval_matrix, pairs_df


# ----------------------------------------------------------------------
# 4. PLOTTING — SQUARE HEATMAPS
# ----------------------------------------------------------------------
def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    cmap: str,
    fmt: str,
    vmin=None,
    vmax=None,
    out_path=None,
):
    """Plot and optionally save a square heatmap."""
    n = len(matrix)

    fig_size = max(
        6,
        min(
            0.5 * n + 3,
            16,
        ),
    )

    fig, ax = plt.subplots(
        figsize=(fig_size, fig_size)
    )

    sns.heatmap(
        matrix,
        annot=n <= 20,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=12,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    if out_path:
        plt.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Saved: {out_path}")

    plt.show()


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Correlation & Engle-Granger cointegration analysis"
        )
    )

    parser.add_argument(
        "--csv",
        default="raw_data.csv",
        help="Path to the input CSV.",
    )

    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help=(
            "P-value threshold for multiple-testing corrections "
            "(default: 0.05)."
        ),
    )

    parser.add_argument(
        "--min-obs",
        type=int,
        default=500,
        help=(
            "Minimum overlapping observations required per pair "
            "(default: 500)."
        ),
    )

    parser.add_argument(
        "--out-prefix",
        default="analysis",
        help="Prefix for saved CSV/PNG outputs.",
    )

    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=(
            "Start date used to reconstruct the date index when the "
            "CSV has no date column. Defaults to download_stock_prices.START_DATE "
            "if that module is importable."
        ),
    )

    args = parser.parse_args()

    print(
        f"Loading prices from {args.csv} ..."
    )

    prices = load_price_matrix(
        args.csv,
        price_col_hint="Close",
        start_date=args.start_date,
    )

    print(
        f"\nPrice matrix shape: {prices.shape}"
    )

    print(
        f"Date range: "
        f"{prices.index.min().date()} "
        f"to "
        f"{prices.index.max().date()}"
    )

    # --------------------------------------------------------------
    # Correlation of returns
    # --------------------------------------------------------------
    # Uncomment if desired.
    #
    # print("\nComputing return correlation matrix...")
    # corr = compute_return_correlation(prices)
    # plot_heatmap(
    #     corr,
    #     title="Return Correlation Matrix",
    #     cmap="coolwarm",
    #     fmt=".2f",
    #     vmin=-1,
    #     vmax=1,
    #     out_path=(
    #         f"{args.out_prefix}_correlation_heatmap.png"
    #     ),
    # )

    # --------------------------------------------------------------
    # Correlation of raw day-over-day price changes
    # --------------------------------------------------------------
    # Uncomment if desired.
    #
    # print(
    #     "\nComputing raw price-change correlation matrix..."
    # )
    # price_change_corr = compute_price_change_correlation(prices)
    # plot_heatmap(
    #     price_change_corr,
    #     title=(
    #         "Day-over-Day Price Change Correlation Matrix"
    #     ),
    #     cmap="coolwarm",
    #     fmt=".2f",
    #     vmin=-1,
    #     vmax=1,
    #     out_path=(
    #         f"{args.out_prefix}_price_change_"
    #         "correlation_heatmap.png"
    #     ),
    # )

    # --------------------------------------------------------------
    # Engle-Granger cointegration
    # --------------------------------------------------------------
    print(
        "\nRunning pairwise Engle-Granger cointegration tests "
        "(this can take a while for many tickers)..."
    )

    pval_matrix, pairs_df = (
        compute_cointegration_matrix(
            prices,
            significance=args.significance,
            min_obs=args.min_obs,
        )
    )

    plot_heatmap(
        pval_matrix,
        title="Engle-Granger Cointegration p-values",
        cmap="viridis_r",
        fmt=".2f",
        vmin=0,
        vmax=1,
        out_path=(
            f"{args.out_prefix}_cointegration_heatmap.png"
        ),
    )

    # --------------------------------------------------------------
    # Raw significant pairs
    # --------------------------------------------------------------
    print(
        f"\nTop pairs by raw p-value "
        f"(uncorrected, p < {args.significance}):"
    )

    sig_pairs_raw = pairs_df[
        pairs_df["cointegrated_at_5pct_uncorrected"]
    ]

    if sig_pairs_raw.empty:
        print(
            "  None found at this significance level."
        )
    else:
        print(
            sig_pairs_raw[
                [
                    "Ticker A",
                    "Ticker B",
                    "n_obs",
                    "coint_pvalue",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------------
    # Bonferroni
    # --------------------------------------------------------------
    print(
        f"\nPairs surviving Bonferroni correction "
        f"(alpha={args.significance}):"
    )

    sig_pairs_bonf = pairs_df[
        pairs_df["significant_bonferroni"]
    ]

    if sig_pairs_bonf.empty:
        print(
            "  None. This is a strict, conservative correction "
            "— surviving it is strong evidence."
        )
    else:
        print(
            sig_pairs_bonf[
                [
                    "Ticker A",
                    "Ticker B",
                    "n_obs",
                    "coint_pvalue",
                    "pvalue_bonferroni",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------------
    # Benjamini-Hochberg FDR
    # --------------------------------------------------------------
    print(
        f"\nPairs surviving Benjamini-Hochberg FDR correction "
        f"(alpha={args.significance}):"
    )

    sig_pairs_fdr = pairs_df[
        pairs_df["significant_fdr_bh"]
    ]

    if sig_pairs_fdr.empty:
        print(
            "  None found."
        )
    else:
        print(
            sig_pairs_fdr[
                [
                    "Ticker A",
                    "Ticker B",
                    "n_obs",
                    "coint_pvalue",
                    "pvalue_fdr_bh",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------------
    output_csv = (
        f"{args.out_prefix}_cointegration_pairs.csv"
    )

    pairs_df.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"\nFull pairwise results saved to "
        f"{output_csv}"
    )


if __name__ == "__main__":
    main()
