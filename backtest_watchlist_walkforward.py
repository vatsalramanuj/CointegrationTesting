"""
Comprehensive fixed-watchlist backtest v3 for the 13 pairs classified as WATCHLIST
by pairs_backtest_engine.py.

Purpose
-------
This is a SECOND-STAGE research backtest. It deliberately does NOT re-select
pairs. The 13-pair universe is fixed from the previous walk-forward evidence
classification, and every pair is evaluated over the full available history.

For every pair:
    1. Split the full history into anchored walk-forward folds.
    2. Estimate the pair model on TRAIN only.
    3. Tune entry/exit thresholds on VALIDATION only.
    4. Apply the selected thresholds once to an untouched TEST period.
    5. Stitch all test periods into a genuine out-of-sample equity curve.
    6. Record train cointegration diagnostics per fold.
    7. Run cost-sensitivity tests using the already-selected thresholds,
       avoiding another layer of parameter fitting.

Portfolio test:
    - Uses exactly these 13 fixed pairs.
    - Re-tunes thresholds on each validation slice.
    - Scores candidates with a leakage-safe reliability-adjusted validation Sharpe:
        score = validation Sharpe * sqrt(min(1, trades / trade_target)).
    - Selects up to --portfolio-top-k pairs using validation information only.
    - Supports static allocation and active-signal reallocation. With active
      reallocation, capital is redistributed only among pairs that actually have
      an open signal on that day, subject to --active-max-pair-weight.
    - Default settings are intentionally oriented toward opportunity capture:
      lower entry thresholds, top-k=10, risk target=15%, active max pair weight=50%.
    - Optional risk-target/top-k parameter sweeps are available.
    - Uses a CONTINUOUS walk-forward schedule by default: every out-of-sample
      trading day belongs to exactly one test block. The preceding block is
      used as validation for the next block, while the first OOS block is
      calibrated from a validation slice inside the initial training window.
      This removes the large calendar gaps present in the earlier design.
    - Chains every test slice, including all-cash slices, so calendar time remains intact.

Optional model comparison:
    --hedge-modes can contain static,rolling,kalman. Running multiple modes
    is more expensive, so the default is rolling only.

This script imports the tested engine from pairs_backtest_engine.py rather than
copying its trading mathematics, so fixes made in that engine remain shared.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

import pairs_backtest_engine as bt
load_price_matrix = bt.load_price_matrix
DEFAULT_START_DATE = bt.DEFAULT_START_DATE


WATCHLIST_PAIRS = [
    ("ALL", "JPM"),
    ("LEN", "UPS"),
    ("HLT", "RSG"),
    ("KBH", "LMT"),
    ("MDLZ", "SO"),
    ("DUK", "RSG"),
    ("CMI", "TMUS"),
    ("CZR", "UNH"),
    ("PEP", "XEL"),
    ("ED", "NOC"),
    ("PEP", "SO"),
    ("DHI", "SCCO"),
    ("NSC", "UNP"),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _safe_float(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}_{pair[1]}"


def _annualized_return_from_daily(returns: pd.Series) -> float:
    """CAGR from the same stitched equity curve used for total return.

    Use actual elapsed calendar time rather than ``len(returns) / 252``.
    The latter can mis-state CAGR when the return series contains gaps,
    stitched walk-forward segments, or any non-standard trading calendar.
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan

    equity = float((1.0 + r).prod())
    if equity <= 0:
        return -1.0

    idx = pd.to_datetime(r.index, errors="coerce")
    valid = ~idx.isna()
    idx = idx[valid]
    if len(idx) < 2:
        return np.nan

    years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 24.0 * 3600.0)
    if years <= 0:
        return np.nan
    return equity ** (1.0 / years) - 1.0


def build_continuous_walk_forward_boundaries(
    n: int,
    initial_train_frac: float,
    n_folds: int,
    initial_val_frac: float = 0.20,
    min_train_obs: int = 60,
) -> list[dict]:
    """Build contiguous walk-forward folds with no OOS calendar gaps.

    The post-initial-training interval is partitioned into ``n_folds``
    contiguous test blocks. For fold 0, a calibration/validation slice is
    carved from the end of the initial training window so the first OOS block
    can be traded without peeking at it. For later folds, the immediately
    preceding OOS block becomes validation, and the current block becomes test.

    Thus every date after the initial training boundary is represented exactly
    once in a test block, while parameter selection for each test block uses
    only information available before that block starts.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not (0.0 < initial_train_frac < 1.0):
        raise ValueError("initial_train_frac must be between 0 and 1")
    if not (0.0 < initial_val_frac < 1.0):
        raise ValueError("initial_val_frac must be between 0 and 1")

    initial_train_end = int(n * initial_train_frac)
    remaining = n - initial_train_end
    if initial_train_end < min_train_obs + 20:
        raise ValueError("Initial training window is too short for continuous walk-forward calibration.")
    if remaining < n_folds * 20:
        raise ValueError("Not enough observations for the requested continuous walk-forward folds.")

    # Exact integer boundaries ensure the complete OOS interval is covered
    # without gaps or overlaps. The final block absorbs any remainder.
    oos_bounds = [
        initial_train_end + int(round(k * remaining / n_folds))
        for k in range(n_folds + 1)
    ]
    oos_bounds[0] = initial_train_end
    oos_bounds[-1] = n

    calibration_size = max(30, int(initial_train_end * initial_val_frac))
    calibration_size = min(calibration_size, initial_train_end - min_train_obs)
    first_train_end = initial_train_end - calibration_size
    if first_train_end < min_train_obs:
        raise ValueError("Initial training window leaves too little data for calibration.")

    folds: list[dict] = []
    for k in range(n_folds):
        if k == 0:
            train_end = first_train_end
            val_start = first_train_end
            val_end = initial_train_end
        else:
            val_start = oos_bounds[k - 1]
            val_end = oos_bounds[k]
            train_end = val_start

        test_start = oos_bounds[k]
        test_end = oos_bounds[k + 1]

        if train_end < min_train_obs or val_end <= val_start or test_end <= test_start:
            continue

        folds.append({
            "fold": k,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "continuous_oos": True,
        })

    if len(folds) != n_folds:
        raise ValueError(
            f"Could only construct {len(folds)} of {n_folds} continuous walk-forward folds."
        )
    return folds


def _get_walk_forward_folds(
    n: int,
    initial_train_frac: float,
    n_folds: int,
    val_frac_within_fold: float,
    continuous_walk_forward: bool,
    continuous_initial_val_frac: float,
) -> list[dict]:
    if continuous_walk_forward:
        return build_continuous_walk_forward_boundaries(
            n, initial_train_frac, n_folds,
            initial_val_frac=continuous_initial_val_frac,
        )

    legacy = bt.build_fold_boundaries(
        n, initial_train_frac, n_folds, val_frac_within_fold
    )
    out = []
    for f in legacy:
        out.append({
            **f,
            "test_start": f["val_end"],
            "test_end": f["fold_end"],
            "val_start": f["train_end"],
            "continuous_oos": False,
        })
    return out


def _classify_pair(summary: dict) -> str:
    """Research classification for the fixed-watchlist stage.

    This is intentionally stricter than the earlier watchlist label. A pair
    must demonstrate repeated, trade-sufficient positive performance across
    the fixed-pair walk-forward evaluation before it can be called robust.
    """
    selected = int(summary.get("valid_test_folds", 0))
    sufficient = int(summary.get("sufficient_trade_folds", 0))
    positive_sufficient = int(summary.get("positive_sufficient_folds", 0))
    total_trades = int(summary.get("total_completed_trades", 0))
    wf_sharpe = _safe_float(summary.get("walk_forward_sharpe"))
    wf_return = _safe_float(summary.get("walk_forward_return"))
    wf_dd = _safe_float(summary.get("walk_forward_max_drawdown"))
    mean_cost = _safe_float(summary.get("mean_cost_pct"))

    if (
        selected >= 3
        and sufficient >= 3
        and positive_sufficient / max(sufficient, 1) >= 2 / 3
        and total_trades >= 8
        and wf_sharpe >= 0.75
        and wf_return > 0
        and wf_dd > -0.25
        and (not np.isfinite(mean_cost) or mean_cost < 0.30)
    ):
        return "robust"

    if (
        sufficient >= 2
        and positive_sufficient / max(sufficient, 1) >= 0.50
        and total_trades >= 4
        and wf_sharpe > 0
        and wf_return > 0
        and wf_dd > -0.35
    ) or (
        wf_sharpe >= 0.50 and wf_return > 0 and selected >= 2
    ):
        return "watchlist"

    return "insufficient_evidence"


def _validation_quality_score(best: dict, trade_target: int = 5) -> float:
    """Reliability-adjusted validation Sharpe used for portfolio selection.

    A high Sharpe from one or two validation trades should not dominate a pair
    with somewhat lower Sharpe but repeated evidence. The square-root trade
    factor is deliberately mild and saturates at ``trade_target``.
    """
    sharpe = _safe_float(best.get("net_sharpe"))
    trades = _safe_float(best.get("completed_trades"), default=0.0)
    if not np.isfinite(sharpe) or trades <= 0:
        return np.nan
    target = max(int(trade_target), 1)
    reliability = np.sqrt(min(1.0, trades / target))
    return float(sharpe * reliability)


def _cointegration_diagnostics(train_a: pd.Series, train_b: pd.Series,
                               significance: float, stability_windows: int) -> dict:
    try:
        pvalue = float(coint(
            train_a, train_b, trend="c",
            autolag=None if bt.COINT_AUTOLAG == "none" else bt.COINT_AUTOLAG,
        )[1])
    except (ValueError, np.linalg.LinAlgError):
        pvalue = np.nan

    beta, intercept = bt.static_ols_hedge_ratio(train_a, train_b)
    spread = bt.compute_spread(train_a, train_b, beta, intercept).dropna()
    half_life = bt.estimate_half_life(spread)
    stability = bt.rolling_cointegration_pass_rate(
        train_a,
        train_b,
        n_windows=stability_windows,
        pvalue_threshold=significance,
        min_window=max(50, len(train_a) // max(stability_windows + 1, 2)),
    )
    try:
        adf_p = float(bt.adfuller(spread, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        # adfuller is imported through the engine module only indirectly in some
        # environments; use the engine result if available, otherwise leave NaN.
        adf_p = np.nan

    return {
        "train_coint_pvalue": pvalue,
        "train_beta": beta,
        "train_half_life_days": half_life,
        "train_coint_stability": stability,
        "train_spread_std": float(spread.std(ddof=1)) if len(spread) else np.nan,
        "train_spread_adf_pvalue": adf_p,
        "train_obs": len(spread),
    }


def _make_config(args, hedge_mode: str, zscore_mode: str | None = None) -> bt.BacktestConfig:
    return bt.BacktestConfig(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        cooldown_days=args.cooldown_days,
        reentry_z=args.reentry_z,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        short_borrow_bps=args.short_borrow_bps,
        financing_bps=args.financing_bps,
        capital=args.capital,
        risk_target=args.risk_target,
        max_gross_leverage=args.max_gross_leverage,
        hedge_mode=hedge_mode,
        hedge_window=args.hedge_window,
        kalman_delta=args.kalman_delta,
        zscore_mode=zscore_mode or args.zscore_mode,
        zscore_window=args.zscore_window,
    )


# ---------------------------------------------------------------------------
# Fixed-pair walk-forward
# ---------------------------------------------------------------------------

def run_pair_walk_forward(
    prices: pd.DataFrame,
    pair: tuple[str, str],
    config: bt.BacktestConfig,
    entry_grid: Sequence[float],
    exit_grid: Sequence[float],
    n_folds: int,
    initial_train_frac: float,
    val_frac_within_fold: float,
    min_validation_trades: int,
    min_test_trades: int,
    significance: float,
    stability_windows: int,
    diagnostics_out: list[dict],
    continuous_walk_forward: bool = True,
    continuous_initial_val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b = pair
    p = prices[[a, b]].dropna()
    folds = _get_walk_forward_folds(
        len(p), initial_train_frac, n_folds, val_frac_within_fold,
        continuous_walk_forward, continuous_initial_val_frac,
    )

    fold_rows: list[dict] = []
    stitched_returns: list[pd.Series] = []
    threshold_rows: list[dict] = []

    print(f"\n[{a}/{b}] {len(p)} usable observations, {len(folds)} folds")

    for fidx, f in enumerate(folds, start=1):
        train = p.iloc[:f["train_end"]]
        val = p.iloc[f["val_start"]:f["val_end"]]
        test = p.iloc[f["test_start"]:f["test_end"]]

        diag = _cointegration_diagnostics(train[a], train[b], significance, stability_windows)
        diagnostics_out.append({
            "fold": f["fold"], "Ticker A": a, "Ticker B": b,
            "train_end": train.index[-1], **diag,
        })

        best, val_sweep = bt.tune_pair(
            train[[a, b]],
            val[[a, b]],
            pair,
            config,
            entry_grid,
            exit_grid,
            min_validation_trades=min_validation_trades,
        )

        if pd.isna(best.get("entry_z", np.nan)):
            fold_rows.append({
                "fold": f["fold"], "Ticker A": a, "Ticker B": b,
                "train_end": train.index[-1],
                "val_start": val.index[0], "val_end": val.index[-1],
                "test_start": test.index[0], "test_end": test.index[-1],
                **diag,
                "validation_sharpe": np.nan,
                "chosen_entry_z": np.nan,
                "chosen_exit_z": np.nan,
                "test_sharpe": np.nan,
                "test_sortino": np.nan,
                "test_return": 0.0,
                "test_max_drawdown": 0.0,
                "test_completed_trades": 0,
                "test_cost_pct": np.nan,
                "test_trade_sufficient": False,
                "capital_exhausted": False,
                "continuous_oos": bool(continuous_walk_forward),
            })
            stitched_returns.append(pd.Series(0.0, index=test.index, name=f"{a}/{b}"))
            continue

        fold_cfg = bt.BacktestConfig(**vars(config))
        fold_cfg.entry_z = float(best["entry_z"])
        fold_cfg.exit_z = float(best["exit_z"])

        test_res = bt._run_backtest_on_segment(
            test[a], test[b], train[a], train[b], fold_cfg,
            allocated_capital=config.capital,
        )
        ts = test_res["trade_stats"]
        ntr = int(ts["n_completed_trades"])
        sufficient = ntr >= min_test_trades

        threshold_rows.append({
            "fold": f["fold"], "Ticker A": a, "Ticker B": b,
            "validation_sharpe": best["net_sharpe"],
            "chosen_entry_z": fold_cfg.entry_z,
            "chosen_exit_z": fold_cfg.exit_z,
            "validation_completed_trades": best.get("completed_trades", np.nan),
            "validation_cost_pct": best.get("cost_pct_of_abs_gross", np.nan),
        })

        fold_rows.append({
            "fold": f["fold"], "Ticker A": a, "Ticker B": b,
            "train_end": train.index[-1],
            "val_start": val.index[0], "val_end": val.index[-1],
            "test_start": test.index[0], "test_end": test.index[-1],
            **diag,
            "validation_sharpe": best["net_sharpe"],
            "chosen_entry_z": fold_cfg.entry_z,
            "chosen_exit_z": fold_cfg.exit_z,
            "test_sharpe": test_res["sharpe"],
            "test_sortino": test_res["sortino"],
            "test_return": test_res["total_return"],
            "test_max_drawdown": test_res["max_drawdown"],
            "test_completed_trades": ntr,
            "test_cost_pct": test_res["cost_pct_of_abs_gross"],
            "test_trade_sufficient": sufficient,
            "capital_exhausted": test_res["capital_exhausted"],
            "continuous_oos": bool(continuous_walk_forward),
        })

        if not test_res["detail"].empty:
            sr = test_res["detail"]["daily_return"].reindex(test.index, fill_value=0.0).copy()
        else:
            sr = pd.Series(0.0, index=test.index)
        sr.name = f"{a}/{b}"
        stitched_returns.append(sr)

        print(
            f"  Fold {fidx}/{len(folds)} | train {train.index[-1].date()} | "
            f"val {best['entry_z']:.2f}/{best['exit_z']:.2f} | "
            f"test Sharpe {test_res['sharpe']:.2f} | trades {ntr}",
            flush=True,
        )

    fold_df = pd.DataFrame(fold_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    if stitched_returns:
        combined = pd.concat(stitched_returns).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")].fillna(0.0)
        equity = (1.0 + combined).cumprod()
        stitched = {
            "walk_forward_sharpe": bt._annualized_sharpe(combined),
            "walk_forward_sortino": bt._sortino(combined),
            "walk_forward_return": float(equity.iloc[-1] - 1.0),
            "walk_forward_cagr": _annualized_return_from_daily(combined),
            "walk_forward_max_drawdown": bt._max_drawdown(equity),
            "walk_forward_vol": bt._annualized_vol(combined),
            "walk_forward_max_drawdown_days": bt._max_drawdown_duration(equity),
        }
    else:
        combined = pd.Series(dtype=float)
        stitched = {
            "walk_forward_sharpe": np.nan,
            "walk_forward_sortino": np.nan,
            "walk_forward_return": np.nan,
            "walk_forward_cagr": np.nan,
            "walk_forward_max_drawdown": np.nan,
            "walk_forward_vol": np.nan,
            "walk_forward_max_drawdown_days": 0,
        }

    fold_valid = fold_df.dropna(subset=["test_sharpe"]) if not fold_df.empty else fold_df
    sufficient_df = fold_valid[fold_valid["test_trade_sufficient"]] if not fold_valid.empty else fold_valid
    positive_sufficient = int((sufficient_df["test_sharpe"] > 0).sum()) if not sufficient_df.empty else 0

    summary = {
        "Ticker A": a,
        "Ticker B": b,
        "available_folds": len(folds),
        "valid_test_folds": len(fold_valid),
        "positive_folds": int((fold_valid["test_sharpe"] > 0).sum()) if not fold_valid.empty else 0,
        "positive_fold_fraction": float((fold_valid["test_sharpe"] > 0).mean()) if not fold_valid.empty else np.nan,
        "sufficient_trade_folds": len(sufficient_df),
        "positive_sufficient_folds": positive_sufficient,
        "positive_sufficient_fold_fraction": positive_sufficient / len(sufficient_df) if len(sufficient_df) else np.nan,
        "mean_test_sharpe": float(fold_valid["test_sharpe"].mean()) if not fold_valid.empty else np.nan,
        "median_test_sharpe": float(fold_valid["test_sharpe"].median()) if not fold_valid.empty else np.nan,
        "worst_test_sharpe": float(fold_valid["test_sharpe"].min()) if not fold_valid.empty else np.nan,
        "mean_test_return": float(fold_valid["test_return"].mean()) if not fold_valid.empty else np.nan,
        "worst_test_drawdown": float(fold_valid["test_max_drawdown"].min()) if not fold_valid.empty else np.nan,
        "mean_cost_pct": float(fold_valid["test_cost_pct"].mean()) if not fold_valid["test_cost_pct"].dropna().empty else np.nan,
        "median_cost_pct": float(fold_valid["test_cost_pct"].median()) if not fold_valid["test_cost_pct"].dropna().empty else np.nan,
        "mean_completed_trades": float(fold_valid["test_completed_trades"].mean()) if not fold_valid.empty else 0.0,
        "median_completed_trades": float(fold_valid["test_completed_trades"].median()) if not fold_valid.empty else 0.0,
        "total_completed_trades": int(fold_valid["test_completed_trades"].sum()) if not fold_valid.empty else 0,
        "continuous_oos": bool(continuous_walk_forward),
        "oos_start": fold_df["test_start"].min() if not fold_df.empty else pd.NaT,
        "oos_end": fold_df["test_end"].max() if not fold_df.empty else pd.NaT,
        **stitched,
    }
    summary["evidence_class"] = _classify_pair(summary)
    summary["robust_candidate"] = summary["evidence_class"] == "robust"

    return fold_df, threshold_df, pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------

def run_cost_sensitivity(
    prices: pd.DataFrame,
    pair: tuple[str, str],
    fold_df: pd.DataFrame,
    config: bt.BacktestConfig,
    costs: Sequence[float],
    results_out: list[dict],
):
    a, b = pair
    p = prices[[a, b]].dropna()
    for cost_bps in costs:
        fold_returns = []
        fold_sharpes = []
        trade_total = 0
        for _, row in fold_df.dropna(subset=["chosen_entry_z"]).iterrows():
            train = p.loc[:row["train_end"]]
            test = p.loc[row["test_start"]:row["test_end"]]
            cfg = bt.BacktestConfig(**vars(config))
            cfg.entry_z = float(row["chosen_entry_z"])
            cfg.exit_z = float(row["chosen_exit_z"])
            cfg.commission_bps = float(cost_bps)
            res = bt._run_backtest_on_segment(
                test[a], test[b], train[a], train[b], cfg,
                allocated_capital=cfg.capital,
            )
            fold_sharpes.append(res["sharpe"])
            trade_total += int(res["trade_stats"]["n_completed_trades"])
            if not res["detail"].empty:
                fold_returns.append(res["detail"]["daily_return"])
        if fold_returns:
            combined = pd.concat(fold_returns).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            eq = (1.0 + combined).cumprod()
            results_out.append({
                "Ticker A": a, "Ticker B": b,
                "commission_bps": cost_bps,
                "walk_forward_sharpe": bt._annualized_sharpe(combined),
                "walk_forward_return": float(eq.iloc[-1] - 1.0),
                "walk_forward_cagr": _annualized_return_from_daily(combined),
                "walk_forward_max_drawdown": bt._max_drawdown(eq),
                "mean_fold_sharpe": float(pd.Series(fold_sharpes).mean()) if fold_sharpes else np.nan,
                "total_completed_trades": trade_total,
            })


# ---------------------------------------------------------------------------
# Fixed-watchlist portfolio
# ---------------------------------------------------------------------------

def run_watchlist_portfolio(
    prices: pd.DataFrame,
    config: bt.BacktestConfig,
    entry_grid: Sequence[float],
    exit_grid: Sequence[float],
    n_folds: int,
    initial_train_frac: float,
    val_frac_within_fold: float,
    min_validation_trades: int,
    portfolio_capital: float,
    max_pair_weight: float,
    allocation_mode: str = "validation_quality",
    portfolio_top_k: int = 10,
    min_portfolio_validation_sharpe: float = 0.0,
    validation_trade_target: int = 5,
    min_portfolio_validation_trades: int = 3,
    max_portfolio_validation_cost_pct: float = 0.30,
    active_reallocation: bool = True,
    active_max_pair_weight: float = 0.50,
    active_total_weight_cap: float = 1.00,
    continuous_walk_forward: bool = True,
    continuous_initial_val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the fixed-watchlist portfolio out of sample.

    Important accounting convention
    --------------------------------
    Each candidate pair is simulated once using ``portfolio_capital`` as its
    notional base. Its daily return is then multiplied by the *actual portfolio
    weight* assigned to that pair on each day. Because the underlying pair
    engine is linear in capital, this is equivalent to scaling its fixed-capital
    P&L while avoiding a second simulation for every weight configuration.

    Active reallocation
    -------------------
    With ``active_reallocation=True`` the portfolio only allocates capital to
    pairs whose executed position is non-zero on that day. Candidate scores are
    still determined exclusively from TRAIN/VALIDATION data; TEST signals are
    used only to decide which already-selected pair is active. Active weights are
    re-normalized and capped by ``active_max_pair_weight``. This is not a
    look-ahead because the signal is executed with the same next-close timing as
    the pair engine.
    """
    valid_modes = {"inverse_vol", "validation_sharpe", "validation_quality", "equal", "hybrid"}
    if allocation_mode not in valid_modes:
        raise ValueError(f"allocation_mode must be one of {sorted(valid_modes)}")
    if portfolio_top_k < 1:
        raise ValueError("portfolio_top_k must be >= 1")
    if max_pair_weight <= 0 or max_pair_weight > 1:
        raise ValueError("max_pair_weight must be in (0, 1]")
    if active_max_pair_weight <= 0 or active_max_pair_weight > 1:
        raise ValueError("active_max_pair_weight must be in (0, 1]")
    if active_total_weight_cap <= 0 or active_total_weight_cap > 1:
        raise ValueError("active_total_weight_cap must be in (0, 1]")
    if min_portfolio_validation_trades < 1:
        raise ValueError("min_portfolio_validation_trades must be >= 1")
    if max_portfolio_validation_cost_pct < 0:
        raise ValueError("max_portfolio_validation_cost_pct must be >= 0")

    folds = _get_walk_forward_folds(
        len(prices), initial_train_frac, n_folds, val_frac_within_fold,
        continuous_walk_forward, continuous_initial_val_frac,
    )
    all_returns = []
    all_active_weight = []
    all_active_pairs = []
    all_base_weight = []
    fold_rows = []

    for fidx, f in enumerate(folds, start=1):
        train = prices.iloc[:f["train_end"]]
        val = prices.iloc[f["val_start"]:f["val_end"]]
        test = prices.iloc[f["test_start"]:f["test_end"]]
        print(f"\nPORTFOLIO fold {fidx}/{len(folds)}")

        candidates = []
        for idx, pair in enumerate(WATCHLIST_PAIRS, start=1):
            a, b = pair
            if a not in train.columns or b not in train.columns:
                continue

            best, _ = bt.tune_pair(
                train[[a, b]], val[[a, b]], pair, config,
                entry_grid, exit_grid,
                min_validation_trades=min_validation_trades,
            )
            val_sharpe = _safe_float(best.get("net_sharpe"))
            val_trades = int(_safe_float(best.get("completed_trades"), default=0.0))
            val_cost_pct = _safe_float(best.get("cost_pct_of_abs_gross"))
            if pd.isna(best.get("entry_z", np.nan)) or not np.isfinite(val_sharpe):
                continue
            if val_sharpe < min_portfolio_validation_sharpe:
                continue
            if val_trades < min_portfolio_validation_trades:
                continue
            if np.isfinite(val_cost_pct) and val_cost_pct > max_portfolio_validation_cost_pct:
                continue

            quality_score = _validation_quality_score(best, validation_trade_target)
            if not np.isfinite(quality_score) or quality_score <= 0:
                continue

            beta, intercept = bt.static_ols_hedge_ratio(train[a], train[b])
            spread = bt.compute_spread(train[a], train[b], beta, intercept).dropna()
            sigma = float(spread.diff().std(ddof=1) * np.sqrt(bt.TRADING_DAYS))
            inv_vol = 1.0 / max(sigma, bt.EPS) if np.isfinite(sigma) else 0.0
            candidates.append({
                "pair": pair,
                "best": best,
                "validation_sharpe": val_sharpe,
                "validation_quality": quality_score,
                "validation_trades": val_trades,
                "validation_cost_pct": val_cost_pct,
                "validation_drawdown": _safe_float(best.get("max_drawdown")),
                "inv_vol": inv_vol,
            })
            print(
                f"  {idx:2d}/{len(WATCHLIST_PAIRS)} {a}/{b} "
                f"-> {best['entry_z']:.2f}/{best['exit_z']:.2f} "
                f"val Sharpe={val_sharpe:.2f}, score={quality_score:.2f}, "
                f"trades={int(best.get('completed_trades', 0))}",
                flush=True,
            )

        if not candidates:
            zero = pd.Series(0.0, index=test.index)
            all_returns.append(zero)
            all_active_weight.append(zero.copy())
            all_active_pairs.append(zero.copy())
            all_base_weight.append(zero.copy())
            fold_rows.append({
                "fold": f["fold"],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "portfolio_pairs_used": 0,
                "trade_sufficient_pairs": 0,
                "completed_trades": 0,
                "allocated_weight": 0.0,
                "avg_base_weight": 0.0,
                "avg_active_weight": 0.0,
                "avg_active_pairs": 0.0,
                "active_reallocation": active_reallocation,
                "continuous_oos": bool(continuous_walk_forward),
                "test_sharpe": np.nan,
                "test_return": 0.0,
                "test_max_drawdown": 0.0,
            })
            continue

        # Selection happens entirely before TEST is examined.
        candidates.sort(key=lambda x: x["validation_quality"], reverse=True)
        candidates = candidates[:portfolio_top_k]

        if allocation_mode == "inverse_vol":
            raw = np.array([x["inv_vol"] for x in candidates], dtype=float)
        elif allocation_mode == "validation_sharpe":
            raw = np.array([max(x["validation_sharpe"], 0.0) for x in candidates], dtype=float)
        elif allocation_mode == "validation_quality":
            raw = np.array([max(x["validation_quality"], 0.0) for x in candidates], dtype=float)
        elif allocation_mode == "hybrid":
            raw = np.array([
                max(x["validation_quality"], 0.0) * x["inv_vol"]
                for x in candidates
            ], dtype=float)
        else:
            raw = np.ones(len(candidates), dtype=float)

        base_weights = bt._capped_weights(raw, max_pair_weight)
        print(
            "  Base allocation: "
            + ", ".join(
                f"{c['pair'][0]}/{c['pair'][1]}={w:.1%}"
                for c, w in zip(candidates, base_weights) if w > 0
            ),
            flush=True,
        )

        pair_returns = []
        pair_active = []
        sufficient_count = 0
        total_trades = 0

        for candidate, base_weight in zip(candidates, base_weights):
            if base_weight <= 0:
                continue

            pair = candidate["pair"]
            a, b = pair
            cfg = bt.BacktestConfig(**vars(config))
            cfg.entry_z = float(candidate["best"]["entry_z"])
            cfg.exit_z = float(candidate["best"]["exit_z"])

            # Simulate at one full unit of portfolio capital. The daily result
            # can then be linearly scaled to any portfolio weight.
            res = bt._run_backtest_on_segment(
                test[a], test[b], train[a], train[b], cfg,
                allocated_capital=portfolio_capital,
            )

            ntr = int(res["trade_stats"]["n_completed_trades"])
            total_trades += ntr
            sufficient_count += int(ntr >= 2)

            detail = res.get("detail")
            if detail is None or detail.empty:
                sr = pd.Series(0.0, index=test.index)
                active = pd.Series(0.0, index=test.index)
            else:
                sr = detail["daily_return"].reindex(test.index, fill_value=0.0).astype(float)
                active = (detail["executed_position"] != 0).astype(float).reindex(test.index, fill_value=0.0)

            pair_returns.append(sr)
            pair_active.append(active)

        if not pair_returns:
            zero = pd.Series(0.0, index=test.index)
            all_returns.append(zero)
            all_active_weight.append(zero.copy())
            all_active_pairs.append(zero.copy())
            all_base_weight.append(zero.copy())
            fold_rows.append({
                "fold": f["fold"], "test_start": test.index[0], "test_end": test.index[-1],
                "portfolio_pairs_used": len(candidates), "trade_sufficient_pairs": sufficient_count,
                "completed_trades": total_trades, "allocated_weight": float(base_weights.sum()),
                "avg_base_weight": float(base_weights.sum()), "avg_active_weight": 0.0,
                "avg_active_pairs": 0.0, "active_reallocation": active_reallocation,
                "test_sharpe": np.nan, "test_return": 0.0, "test_max_drawdown": 0.0,
            })
            continue

        return_matrix = pd.concat(pair_returns, axis=1)
        active_matrix = pd.concat(pair_active, axis=1)
        return_matrix.columns = range(len(return_matrix.columns))
        active_matrix.columns = return_matrix.columns

        base_weight_series = pd.Series(base_weights, index=return_matrix.columns)
        base_weight_daily = pd.Series(float(base_weights.sum()), index=test.index)

        if active_reallocation:
            effective_weight_matrix = pd.DataFrame(0.0, index=test.index, columns=return_matrix.columns)
            for date in test.index:
                raw_active = base_weights * active_matrix.loc[date].to_numpy(dtype=float)
                w = bt._capped_weights(raw_active, active_max_pair_weight)
                total_w = float(w.sum())
                if total_w > active_total_weight_cap and total_w > bt.EPS:
                    w *= active_total_weight_cap / total_w
                effective_weight_matrix.loc[date] = w
            daily_ret = (return_matrix * effective_weight_matrix).sum(axis=1)
            active_weight = effective_weight_matrix.sum(axis=1)
            active_pairs = (effective_weight_matrix > 0).sum(axis=1).astype(float)
        else:
            effective_weight_matrix = pd.DataFrame(
                np.tile(base_weights, (len(test.index), 1)),
                index=test.index,
                columns=return_matrix.columns,
            )
            daily_ret = (return_matrix * effective_weight_matrix).sum(axis=1)
            active_weight = (effective_weight_matrix * active_matrix).sum(axis=1)
            active_pairs = ((effective_weight_matrix * active_matrix) > 0).sum(axis=1).astype(float)

        all_returns.append(daily_ret)
        all_active_weight.append(active_weight)
        all_active_pairs.append(active_pairs)
        all_base_weight.append(base_weight_daily)

        eq = (1.0 + daily_ret).cumprod()
        fold_rows.append({
            "fold": f["fold"],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "portfolio_pairs_used": len(candidates),
            "trade_sufficient_pairs": sufficient_count,
            "completed_trades": total_trades,
            "allocated_weight": float(base_weights.sum()),
            "avg_base_weight": float(base_weights.sum()),
            "avg_active_weight": float(active_weight.mean()),
            "avg_active_pairs": float(active_pairs.mean()),
            "active_reallocation": active_reallocation,
            "continuous_oos": bool(continuous_walk_forward),
            "test_sharpe": bt._annualized_sharpe(daily_ret),
            "test_return": float(eq.iloc[-1] - 1.0),
            "test_max_drawdown": bt._max_drawdown(eq),
        })

    if not all_returns:
        return pd.DataFrame(fold_rows), pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(all_returns).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")].fillna(0.0)
    active_weight = pd.concat(all_active_weight).sort_index()
    active_weight = active_weight[~active_weight.index.duplicated(keep="first")].fillna(0.0)
    active_pairs = pd.concat(all_active_pairs).sort_index()
    active_pairs = active_pairs[~active_pairs.index.duplicated(keep="first")].fillna(0.0)
    base_weight = pd.concat(all_base_weight).sort_index()
    base_weight = base_weight[~base_weight.index.duplicated(keep="first")].fillna(0.0)

    equity = (1.0 + combined).cumprod()

    # In continuous mode, the stitched portfolio must cover every row from
    # the first OOS test block through the end of the final test block. This
    # guards against silently dropping validation-gap days from the investment
    # clock, which was the main issue in the earlier walk-forward design.
    expected_oos_len = folds[-1]["test_end"] - folds[0]["test_start"]
    oos_coverage_ok = (
        len(combined) == expected_oos_len
        if continuous_walk_forward else True
    )
    if continuous_walk_forward and not oos_coverage_ok:
        raise RuntimeError(
            f"Continuous OOS coverage failure: got {len(combined)} rows, "
            f"expected {expected_oos_len}."
        )

    fold_df = pd.DataFrame(fold_rows)
    portfolio_summary = pd.DataFrame([{
        "portfolio": "13-pair fixed watchlist",
        "folds": len(folds),
        "allocation_mode": allocation_mode,
        "portfolio_top_k": portfolio_top_k,
        "min_portfolio_validation_sharpe": min_portfolio_validation_sharpe,
        "validation_trade_target": validation_trade_target,
        "risk_target": config.risk_target,
        "max_pair_weight": max_pair_weight,
        "active_reallocation": active_reallocation,
        "active_max_pair_weight": active_max_pair_weight,
        "active_total_weight_cap": active_total_weight_cap,
        "min_portfolio_validation_trades": min_portfolio_validation_trades,
        "max_portfolio_validation_cost_pct": max_portfolio_validation_cost_pct,
        "continuous_oos": bool(continuous_walk_forward),
        "oos_start": combined.index.min() if not combined.empty else pd.NaT,
        "oos_end": combined.index.max() if not combined.empty else pd.NaT,
        "oos_calendar_days": ((combined.index.max() - combined.index.min()).days if not combined.empty else 0),
        "oos_trading_rows": int(len(combined)),
        "oos_coverage_complete": bool(oos_coverage_ok),
        "walk_forward_sharpe": bt._annualized_sharpe(combined),
        "walk_forward_sortino": bt._sortino(combined),
        "walk_forward_return": float(equity.iloc[-1] - 1.0),
        "walk_forward_cagr": _annualized_return_from_daily(combined),
        "walk_forward_max_drawdown": bt._max_drawdown(equity),
        "walk_forward_max_drawdown_days": bt._max_drawdown_duration(equity),
        "annualized_vol": bt._annualized_vol(combined),
        "avg_allocated_weight": float(base_weight.mean()),
        "avg_active_weight": float(active_weight.mean()),
        "avg_active_pairs": float(active_pairs.mean()),
        "positive_fold_fraction": float((fold_df["test_sharpe"] > 0).mean()),
        "total_completed_trades": int(fold_df["completed_trades"].sum()),
    }])

    portfolio_daily = pd.DataFrame({
        "daily_return": combined,
        "equity": equity,
        "base_weight": base_weight.reindex(combined.index, fill_value=0.0),
        "active_weight": active_weight.reindex(combined.index, fill_value=0.0),
        "active_pairs": active_pairs.reindex(combined.index, fill_value=0.0),
    })
    return fold_df, portfolio_daily, portfolio_summary

def plot_portfolio_equity(
    portfolio_daily: pd.DataFrame,
    portfolio_folds: pd.DataFrame,
    out_path: str,
    title: str = "13-Pair Watchlist — Stitched Out-of-Sample Portfolio Equity",
) -> None:
    """Save the stitched portfolio equity curve and drawdown.

    The plotted equity is exactly the same daily portfolio series used by the
    portfolio summary, so the figure is not a separately re-simulated result.
    Dashed vertical lines mark the starts of successive test folds.
    """
    if portfolio_daily is None or portfolio_daily.empty:
        return

    df = portfolio_daily.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    equity = pd.to_numeric(df["equity"], errors="coerce").dropna()
    if equity.empty:
        return

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})

    axes[0].plot(equity.index, equity.values, linewidth=1.8)
    axes[0].axhline(1.0, linewidth=0.8, alpha=0.5)
    axes[0].set_title(title)
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(drawdown.index, drawdown.values, 0.0, alpha=0.25)
    axes[1].plot(drawdown.index, drawdown.values, linewidth=1.0)
    axes[1].axhline(0.0, linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)

    if portfolio_folds is not None and not portfolio_folds.empty:
        for _, row in portfolio_folds.iterrows():
            start = row.get("test_start")
            if pd.notna(start):
                start = pd.to_datetime(start)
                axes[0].axvline(start, linestyle="--", linewidth=0.8, alpha=0.45)
                axes[1].axvline(start, linestyle="--", linewidth=0.8, alpha=0.45)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive fixed-watchlist pairs backtest v3")
    parser.add_argument("--data-csv", default="raw_data.csv")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--out-prefix", default="watchlist_v3")
    parser.add_argument("--hedge-modes", default="rolling",
                        help="Comma-separated modes: static,rolling,kalman. Running several modes increases runtime.")
    parser.add_argument("--zscore-mode", choices=["static", "rolling"], default="rolling")
    parser.add_argument("--hedge-window", type=int, default=120)
    parser.add_argument("--zscore-window", type=int, default=120)
    parser.add_argument("--kalman-delta", type=float, default=1e-5)
    parser.add_argument("--entry-z", type=float, default=1.75)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--cooldown-days", type=int, default=5)
    parser.add_argument("--reentry-z", type=float, default=1.0)
    parser.add_argument("--entry-z-grid", default="1.25,1.5,1.75,2.0,2.25")
    parser.add_argument("--exit-z-grid", default="0.25,0.5,0.75")
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--cost-grid", default="2,5,10,20",
                        help="Commission-bps scenarios for cost sensitivity. Selected thresholds are NOT re-tuned.")
    parser.add_argument("--short-borrow-bps", type=float, default=0.0)
    parser.add_argument("--financing-bps", type=float, default=0.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--risk-target", type=float, default=0.15)
    parser.add_argument("--risk-target-grid", default="0.10,0.15,0.20")
    parser.add_argument("--max-gross-leverage", type=float, default=2.0)
    parser.add_argument("--portfolio-capital", type=float, default=100_000.0)
    parser.add_argument("--max-pair-weight", type=float, default=0.20)
    parser.add_argument("--active-max-pair-weight", type=float, default=0.50,
                        help="Maximum weight of one pair after active-signal reallocation.")
    parser.add_argument("--active-total-weight-cap", type=float, default=1.00,
                        help="Maximum total capital deployed across active pairs after reallocation.")
    parser.add_argument(
        "--portfolio-allocation",
        choices=["validation_quality", "validation_sharpe", "inverse_vol", "equal", "hybrid"],
        default="validation_quality",
        help="How base capital is distributed among selected pairs.",
    )
    parser.add_argument(
        "--portfolio-top-k", type=int, default=6,
        help="Maximum number of watchlist pairs selected per fold.",
    )
    parser.add_argument(
        "--portfolio-top-k-grid", default="3,4,5,6,8,10,13",
        help="Top-k values used by --portfolio-sweep.",
    )
    parser.add_argument(
        "--min-portfolio-validation-sharpe", type=float, default=0.5,
        help="Exclude pairs below this raw validation Sharpe before portfolio selection.",
    )
    parser.add_argument(
        "--min-portfolio-validation-trades", type=int, default=3,
        help="Require at least this many completed validation trades before portfolio selection.",
    )
    parser.add_argument(
        "--max-portfolio-validation-cost-pct", type=float, default=0.30,
        help="Exclude pairs whose validation trading costs exceed this fraction of absolute gross P&L.",
    )
    parser.add_argument("--validation-trade-target", type=int, default=5,
                        help="Trades at which the reliability penalty saturates in the validation score.")
    parser.add_argument("--min-validation-trades", type=int, default=2)
    parser.add_argument("--min-test-trades", type=int, default=2)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--stability-windows", type=int, default=4)
    parser.add_argument("--wf-folds", type=int, default=5)
    parser.add_argument("--wf-initial-train-frac", type=float, default=0.50)
    parser.add_argument("--wf-val-frac", type=float, default=0.50,
                        help="Legacy/gapped mode only: validation fraction inside each fold.")
    parser.add_argument("--continuous-initial-val-frac", type=float, default=0.20,
                        help="Fraction of the initial training window reserved for first-fold calibration in continuous mode.")
    parser.add_argument("--legacy-gapped-folds", action="store_true",
                        help="Use the previous validation-gap walk-forward design instead of continuous OOS folds.")
    parser.add_argument("--coint-autolag", choices=["aic", "bic", "t-stat", "none"], default="aic")
    parser.add_argument("--no-active-reallocation", action="store_true",
                        help="Disable dynamic allocation to currently active signals; useful as a control.")
    parser.add_argument("--portfolio-sweep", action="store_true",
                        help="Run risk-target x top-k portfolio sweep and save a comparison CSV.")
    parser.add_argument("--no-cost-sensitivity", action="store_true")
    parser.add_argument("--no-portfolio", action="store_true")
    args = parser.parse_args()

    bt.COINT_AUTOLAG = args.coint_autolag
    entry_grid = _parse_grid(args.entry_z_grid)
    exit_grid = _parse_grid(args.exit_z_grid)
    cost_grid = _parse_grid(args.cost_grid)
    risk_grid = _parse_grid(args.risk_target_grid)
    top_k_grid = [int(x) for x in _parse_grid(args.portfolio_top_k_grid)]
    hedge_modes = [x.strip() for x in args.hedge_modes.split(",") if x.strip()]
    continuous_walk_forward = not args.legacy_gapped_folds

    prices = load_price_matrix(args.data_csv, start_date=args.start_date)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")

    missing = sorted({t for pair in WATCHLIST_PAIRS for t in pair if t not in prices.columns})
    if missing:
        raise ValueError(f"The following watchlist tickers are missing from {args.data_csv}: {missing}")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print("13-pair fixed watchlist v3:")
    print(", ".join(f"{a}/{b}" for a, b in WATCHLIST_PAIRS))
    print(f"Data: {prices.index[0]} -> {prices.index[-1]} ({len(prices)} rows)")
    print(f"Hedge modes: {hedge_modes}")
    print(f"Entry/exit grid: {entry_grid} / {exit_grid}")
    print(f"Default risk target: {args.risk_target:.1%}; base top-k: {args.portfolio_top_k}; active max pair weight: {args.active_max_pair_weight:.1%}")
    print(f"Walk-forward mode: {'continuous' if continuous_walk_forward else 'legacy/gapped'}; "
          f"portfolio validation filters: Sharpe >= {args.min_portfolio_validation_sharpe:.2f}, "
          f"trades >= {args.min_portfolio_validation_trades}, "
          f"cost <= {args.max_portfolio_validation_cost_pct:.0%}")

    all_pair_summaries = []
    all_fold_rows = []
    all_threshold_rows = []
    all_diagnostics = []
    all_cost_rows = []

    for mode in hedge_modes:
        print(f"\n{'=' * 78}\nMODEL: hedge={mode}, zscore={args.zscore_mode}\n{'=' * 78}")
        config = _make_config(args, mode)

        for pair in WATCHLIST_PAIRS:
            fold_df, threshold_df, summary_df = run_pair_walk_forward(
                prices, pair, config, entry_grid, exit_grid,
                args.wf_folds, args.wf_initial_train_frac, args.wf_val_frac,
                args.min_validation_trades, args.min_test_trades,
                args.significance, args.stability_windows, all_diagnostics,
                continuous_walk_forward=continuous_walk_forward,
                continuous_initial_val_frac=args.continuous_initial_val_frac,
            )
            summary_df["hedge_mode"] = mode
            summary_df["zscore_mode"] = args.zscore_mode
            fold_df["hedge_mode"] = mode
            fold_df["zscore_mode"] = args.zscore_mode
            threshold_df["hedge_mode"] = mode
            threshold_df["zscore_mode"] = args.zscore_mode
            all_pair_summaries.append(summary_df)
            all_fold_rows.append(fold_df)
            all_threshold_rows.append(threshold_df)

            if not args.no_cost_sensitivity:
                run_cost_sensitivity(prices, pair, fold_df, config, cost_grid, all_cost_rows)

    pair_summary = pd.concat(all_pair_summaries, ignore_index=True) if all_pair_summaries else pd.DataFrame()
    fold_results = pd.concat(all_fold_rows, ignore_index=True) if all_fold_rows else pd.DataFrame()
    threshold_results = pd.concat(all_threshold_rows, ignore_index=True) if all_threshold_rows else pd.DataFrame()
    diagnostics = pd.DataFrame(all_diagnostics)
    cost_results = pd.DataFrame(all_cost_rows)

    if not pair_summary.empty:
        pair_summary = pair_summary.sort_values(
            ["hedge_mode", "evidence_class", "walk_forward_sharpe"],
            ascending=[True, True, False],
            key=lambda s: s.map({"robust": 0, "watchlist": 1, "insufficient_evidence": 2}) if s.name == "evidence_class" else s,
        )

    pair_summary.to_csv(f"{prefix}_pair_summary.csv", index=False)
    fold_results.to_csv(f"{prefix}_fold_results.csv", index=False)
    threshold_results.to_csv(f"{prefix}_threshold_selection.csv", index=False)
    diagnostics.to_csv(f"{prefix}_train_diagnostics.csv", index=False)
    if not cost_results.empty:
        cost_results.to_csv(f"{prefix}_cost_sensitivity.csv", index=False)

    if not args.no_portfolio:
        primary_mode = hedge_modes[0]
        active_reallocation = not args.no_active_reallocation

        configs_to_run = [(args.risk_target, args.portfolio_top_k, "base")]
        if args.portfolio_sweep:
            configs_to_run = [
                (rt, tk, f"risk{rt:.3f}_topk{tk}")
                for rt in risk_grid for tk in top_k_grid
            ]

        sweep_rows = []
        for risk_target, top_k, label in configs_to_run:
            print(
                f"\n{'=' * 78}\nPORTFOLIO V3: hedge={primary_mode}, risk={risk_target:.1%}, "
                f"top-k={top_k}, allocation={args.portfolio_allocation}, "
                f"active_reallocation={active_reallocation}\n{'=' * 78}"
            )
            config = _make_config(args, primary_mode)
            config.risk_target = float(risk_target)

            fold_df, portfolio_daily, portfolio_summary = run_watchlist_portfolio(
                prices, config, entry_grid, exit_grid,
                args.wf_folds, args.wf_initial_train_frac, args.wf_val_frac,
                args.min_validation_trades, args.portfolio_capital, args.max_pair_weight,
                allocation_mode=args.portfolio_allocation,
                portfolio_top_k=int(top_k),
                min_portfolio_validation_sharpe=args.min_portfolio_validation_sharpe,
                validation_trade_target=args.validation_trade_target,
                min_portfolio_validation_trades=args.min_portfolio_validation_trades,
                max_portfolio_validation_cost_pct=args.max_portfolio_validation_cost_pct,
                active_reallocation=active_reallocation,
                active_max_pair_weight=args.active_max_pair_weight,
                active_total_weight_cap=args.active_total_weight_cap,
                continuous_walk_forward=continuous_walk_forward,
                continuous_initial_val_frac=args.continuous_initial_val_frac,
            )

            row = portfolio_summary.iloc[0].to_dict() if not portfolio_summary.empty else {
                "portfolio": "13-pair fixed watchlist"
            }
            row["run_label"] = label
            sweep_rows.append(row)

            suffix = "portfolio" if label == "base" else f"portfolio_{label}"
            fold_df.to_csv(f"{prefix}_{suffix}_folds.csv", index=False)
            portfolio_daily.to_csv(f"{prefix}_{suffix}_daily.csv")
            portfolio_summary.to_csv(f"{prefix}_{suffix}_summary.csv", index=False)

            if not portfolio_daily.empty:
                plot_path = f"{prefix}_{suffix}_equity.png"
                plot_portfolio_equity(
                    portfolio_daily, fold_df, plot_path,
                    title=(
                        f"13-Pair Watchlist V3 — OOS Portfolio Equity "
                        f"(risk={risk_target:.1%}, top-k={top_k}, active={active_reallocation})"
                    ),
                )
                print(f"Saved portfolio equity graph: {plot_path}")

            if not portfolio_summary.empty:
                print(portfolio_summary.to_string(index=False))

        sweep_df = pd.DataFrame(sweep_rows)
        if not sweep_df.empty:
            sweep_df.to_csv(f"{prefix}_portfolio_sweep_summary.csv", index=False)
            print("\n=== PORTFOLIO CONFIGURATION SUMMARY ===")
            cols = [
                "run_label", "risk_target", "portfolio_top_k", "allocation_mode",
                "active_reallocation", "active_total_weight_cap", "continuous_oos",
                "walk_forward_sharpe", "walk_forward_return",
                "walk_forward_cagr", "walk_forward_max_drawdown", "annualized_vol",
                "avg_active_weight", "avg_active_pairs", "total_completed_trades",
            ]
            cols = [c for c in cols if c in sweep_df.columns]
            print(sweep_df[cols].to_string(index=False))

    if not pair_summary.empty:
        primary = pair_summary[pair_summary["hedge_mode"] == hedge_modes[0]].copy()
        print("\n=== PRIMARY MODEL PAIR SUMMARY ===")
        cols = [
            "Ticker A", "Ticker B", "evidence_class", "valid_test_folds",
            "sufficient_trade_folds", "positive_sufficient_folds",
            "walk_forward_sharpe", "walk_forward_return",
            "walk_forward_max_drawdown", "total_completed_trades",
        ]
        print(primary[cols].to_string(index=False))

    print("\nSaved outputs:")
    for suffix in [
        "pair_summary.csv", "fold_results.csv", "threshold_selection.csv",
        "train_diagnostics.csv", "cost_sensitivity.csv", "portfolio_summary.csv",
        "portfolio_sweep_summary.csv",
    ]:
        path = f"{prefix}_{suffix}"
        if os.path.exists(path):
            print(f"  {path}")


if __name__ == "__main__":
    main()
