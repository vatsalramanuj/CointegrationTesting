"""
Pairs Trading Backtest v2
=========================

A research-oriented, walk-forward backtester for cointegrated pairs.

Key changes from the original version:
    * Pair selection is performed on TRAIN data, not on the full sample.
    * Total return is consistent with the fixed-capital P&L model.
    * Trade counts distinguish entries/exits from completed trades.
    * Signal -> next-close execution timing is explicit and consistent.
    * Transaction costs include configurable commission + slippage.
    * Optional short-borrow and financing costs are charged while a trade is open.
    * Stops include cooldown/re-entry protection so a broken pair is not
      immediately re-entered over and over.
    * Static, rolling and simple Kalman-filter hedge ratios are supported.
    * Static train z-scores or leakage-safe rolling z-scores are supported.
    * Half-life and rolling cointegration-stability diagnostics are available.
    * Risk-based position sizing and gross-exposure caps replace the old
      "one unit of spread" capital proxy.
    * Validation and anchored walk-forward testing are supported.
    * Walk-forward pair selection is repeated inside each fold.
    * Pair robustness summaries include available-vs-selected folds, minimum-trade rules,
      stitched walk-forward Sharpe/return/CAGR/max-drawdown, and worst-fold Sharpe.
    * An optional multi-pair portfolio backtest is included with a true per-pair weight cap.

This remains a research backtester, not a production execution system.
It does not model exact exchange execution, borrow availability, dividends,
auction mechanics, liquidity impact, or tax.

Dependencies:
    pip install pandas numpy matplotlib seaborn statsmodels

Examples:
    python pairs_backtest_engine.py --pair-selection raw --top-n 20
    python pairs_backtest_engine.py --pair-selection fdr_bh --walk-forward
    python pairs_backtest_engine.py --walk-forward --portfolio --top-n 10
    python pairs_backtest_engine.py --hedge-mode rolling --zscore-mode rolling
    python pairs_backtest_engine.py --walk-forward --top-n 10 --no-plots --coint-autolag=none
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

from explore_correlation_cointegration import load_price_matrix, DEFAULT_START_DATE


TRADING_DAYS = 252
EPS = 1e-12
COINT_AUTOLAG = "aic"


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _fmt_date(value):
    if hasattr(value, "date"):
        return str(value.date())
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _annualized_sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2 or r.std(ddof=1) <= EPS:
        return np.nan
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS))


def _annualized_vol(returns: pd.Series) -> float:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    downside = r[r < 0]
    if len(downside) == 0:
        return np.inf if r.mean() > 0 else np.nan
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    if downside_dev <= EPS:
        return np.nan
    return float((r.mean() / downside_dev) * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    e = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if e.empty:
        return np.nan
    running_max = e.cummax()
    dd = e / running_max - 1.0
    return float(max(dd.min(), -1.0))


def _max_drawdown_duration(equity: pd.Series) -> int:
    e = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if e.empty:
        return 0
    running_max = e.cummax()
    underwater = e < running_max - EPS
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def _cagr_from_equity(equity: pd.Series) -> float:
    """Compound annual growth rate from a dated equity curve."""
    e = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) < 2 or e.iloc[0] <= 0 or e.iloc[-1] <= 0:
        return np.nan
    try:
        days = (pd.Timestamp(e.index[-1]) - pd.Timestamp(e.index[0])).days
    except (TypeError, ValueError):
        days = 0
    if days <= 0:
        return np.nan
    years = days / 365.25
    return float((e.iloc[-1] / e.iloc[0]) ** (1.0 / years) - 1.0)


def _profit_factor(trade_log: pd.DataFrame) -> float:
    if trade_log.empty:
        return np.nan
    closed = trade_log[~trade_log["stopped_open_at_end"]]
    if closed.empty:
        return np.nan
    wins = closed.loc[closed["net_pnl"] > 0, "net_pnl"].sum()
    losses = -closed.loc[closed["net_pnl"] <= 0, "net_pnl"].sum()
    if losses <= EPS:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def _bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 0,
                         seed: int = 42) -> tuple[float, float]:
    if n_boot <= 0:
        return np.nan, np.nan
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    samples = r[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    sharpes = np.divide(
        means, stds, out=np.full(n_boot, np.nan), where=stds > EPS
    ) * np.sqrt(TRADING_DAYS)
    return float(np.nanpercentile(sharpes, 2.5)), float(np.nanpercentile(sharpes, 97.5))


def _fmt_optional(x, digits=2):
    return "nan" if pd.isna(x) else f"{x:.{digits}f}"


# ----------------------------------------------------------------------
# Cointegration / pair selection -- IMPORTANT: run on TRAIN only.
# ----------------------------------------------------------------------
def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def load_pair_universe(csv_path: Optional[str]) -> Optional[set[tuple[str, str]]]:
    """Optional whitelist from an external CSV.

    This is deliberately only a universe restriction. The ranking/significance
    decision is recomputed on the current TRAIN slice, preventing the external
    CSV's p-values from selecting the test-period winners.
    """
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    required = {"Ticker A", "Ticker B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {missing}")
    pairs = {_pair_key(a, b) for a, b in zip(df["Ticker A"], df["Ticker B"]) if a != b}
    return pairs


def _prepare_pair_series(prices: pd.DataFrame, a: str, b: str) -> tuple[pd.Series, pd.Series]:
    x = pd.to_numeric(prices[a], errors="coerce")
    y = pd.to_numeric(prices[b], errors="coerce")
    df = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    df = df[(df["a"] > 0) & (df["b"] > 0)]
    return df["a"], df["b"]


def static_ols_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> tuple[float, float]:
    x = sm.add_constant(pd.to_numeric(price_b, errors="coerce").to_numpy())
    y = pd.to_numeric(price_a, errors="coerce").to_numpy()
    model = OLS(y, x, missing="drop").fit()
    intercept, beta = model.params
    return float(beta), float(intercept)


def compute_spread(price_a: pd.Series, price_b: pd.Series,
                   beta: float, intercept: float) -> pd.Series:
    return pd.to_numeric(price_a, errors="coerce") - beta * pd.to_numeric(price_b, errors="coerce") - intercept


def estimate_half_life(spread: pd.Series) -> float:
    s = pd.Series(spread).dropna()
    if len(s) < 30:
        return np.nan
    lag = s.shift(1)
    delta = s.diff()
    reg = pd.concat([delta.rename("delta"), lag.rename("lag")], axis=1).dropna()
    if len(reg) < 20:
        return np.nan
    model = OLS(reg["delta"].to_numpy(), sm.add_constant(reg["lag"].to_numpy())).fit()
    b = float(model.params[1])
    if b >= 0:
        return np.inf
    hl = -np.log(2.0) / b
    return float(hl) if np.isfinite(hl) and hl > 0 else np.nan


def rolling_cointegration_pass_rate(price_a: pd.Series, price_b: pd.Series,
                                    n_windows: int = 0,
                                    pvalue_threshold: float = 0.05,
                                    min_window: int = 100) -> float:
    """Fraction of chronological rolling windows that pass EG cointegration.

    This is a diagnostic/filter, not a claim that rolling tests are independent.
    """
    if n_windows <= 0:
        return np.nan
    n = len(price_a)
    window = max(min_window, n // max(n_windows, 1))
    if window >= n:
        return np.nan
    starts = np.linspace(0, n - window, n_windows, dtype=int)
    pvals = []
    for start in np.unique(starts):
        xa = price_a.iloc[start:start + window]
        xb = price_b.iloc[start:start + window]
        try:
            p = coint(xa, xb, trend="c", autolag=None if COINT_AUTOLAG == "none" else COINT_AUTOLAG)[1]
            if np.isfinite(p):
                pvals.append(float(p))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not pvals:
        return np.nan
    return float(np.mean(np.array(pvals) <= pvalue_threshold))


def compute_train_pair_candidates(
    prices: pd.DataFrame,
    pair_selection: str,
    significance: float = 0.05,
    top_n: int = 20,
    min_obs: int = 252,
    pair_universe: Optional[set[tuple[str, str]]] = None,
    min_half_life: float = 1.0,
    max_half_life: float = np.inf,
    stability_windows: int = 0,
    min_coint_stability: float = 0.0,
    max_cost_to_expected_move: float = np.inf,
    entry_z_for_cost: float = 2.0,
    exit_z_for_cost: float = 0.5,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """Compute train-only pair candidates efficiently.

    Stage 1: run Engle-Granger over the permitted universe and apply the
    multiple-testing correction.

    Stage 2: only for the preliminary selected set, compute the more expensive
    diagnostics (hedge ratio, half-life, ADF, stability and cost filter).

    This keeps the statistical correction based on the full tested universe,
    while avoiding unnecessary diagnostics on hundreds of pairs that already
    fail the initial selection step.
    """
    tickers = list(prices.columns)
    raw_rows = []

    # ---- Stage 1: the only operation that must touch the whole pair universe.
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            if pair_universe is not None and _pair_key(a, b) not in pair_universe:
                continue
            xa, xb = _prepare_pair_series(prices, a, b)
            if len(xa) < min_obs:
                continue
            try:
                pvalue = float(coint(
                    xa, xb, trend="c",
                    autolag=None if COINT_AUTOLAG == "none" else COINT_AUTOLAG,
                )[1])
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(pvalue):
                raw_rows.append({
                    "Ticker A": a,
                    "Ticker B": b,
                    "coint_pvalue": pvalue,
                    "n_obs": len(xa),
                })

    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        return raw

    valid = raw["coint_pvalue"].notna().to_numpy()
    raw["pvalue_bonferroni"] = np.nan
    raw["pvalue_fdr_bh"] = np.nan
    raw["significant_bonferroni"] = False
    raw["significant_fdr_bh"] = False
    if valid.any():
        bonf_reject, bonf_p, _, _ = multipletests(
            raw.loc[valid, "coint_pvalue"].to_numpy(),
            alpha=significance, method="bonferroni",
        )
        fdr_reject, fdr_p, _, _ = multipletests(
            raw.loc[valid, "coint_pvalue"].to_numpy(),
            alpha=significance, method="fdr_bh",
        )
        raw.loc[valid, "pvalue_bonferroni"] = bonf_p
        raw.loc[valid, "pvalue_fdr_bh"] = fdr_p
        raw.loc[valid, "significant_bonferroni"] = bonf_reject
        raw.loc[valid, "significant_fdr_bh"] = fdr_reject

    if pair_selection == "bonferroni":
        preliminary = raw[raw["significant_bonferroni"]].copy()
    elif pair_selection == "fdr_bh":
        preliminary = raw[raw["significant_fdr_bh"]].copy()
    elif pair_selection == "raw":
        preliminary = raw.sort_values("coint_pvalue").head(top_n).copy()
    else:
        raise ValueError(f"Unknown pair_selection: {pair_selection}")

    # Do not compute expensive diagnostics on the entire universe.
    # For corrected modes, sort by corrected p-value first and only keep a
    # modest candidate pool if top_n is smaller than the significant set.
    if pair_selection != "raw":
        pcol = "pvalue_bonferroni" if pair_selection == "bonferroni" else "pvalue_fdr_bh"
        preliminary = preliminary.sort_values([pcol, "coint_pvalue"]).head(max(top_n, 1)).copy()

    if preliminary.empty:
        return preliminary.reset_index(drop=True)

    rows = []
    for _, base in preliminary.iterrows():
        a, b = base["Ticker A"], base["Ticker B"]
        xa, xb = _prepare_pair_series(prices, a, b)
        try:
            beta, intercept = static_ols_hedge_ratio(xa, xb)
            spread = compute_spread(xa, xb, beta, intercept).dropna()
            hl = estimate_half_life(spread)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if np.isfinite(min_half_life) and (not np.isfinite(hl) or hl < min_half_life):
            continue
        if np.isfinite(max_half_life) and (not np.isfinite(hl) or hl > max_half_life):
            continue

        stability = rolling_cointegration_pass_rate(
            xa, xb, n_windows=stability_windows,
            pvalue_threshold=significance, min_window=max(50, min_obs // 2),
        )
        if stability_windows > 0 and (not np.isfinite(stability) or stability < min_coint_stability):
            continue

        try:
            spread_adf_pvalue = float(adfuller(spread, autolag="AIC")[1])
        except (ValueError, np.linalg.LinAlgError):
            spread_adf_pvalue = np.nan

        spread_std = float(spread.std(ddof=1))
        mean_notional = float((xa + abs(beta) * xb).mean())
        roundtrip_cost = 2.0 * (transaction_cost_bps + slippage_bps) / 10_000.0 * mean_notional
        expected_move = max(entry_z_for_cost - exit_z_for_cost, 0.1) * spread_std
        cost_ratio = roundtrip_cost / expected_move if expected_move > EPS else np.inf
        if cost_ratio > max_cost_to_expected_move:
            continue

        out = base.to_dict()
        out.update({
            "beta_train": beta,
            "intercept_train": intercept,
            "half_life_days": hl,
            "coint_stability": stability,
            "spread_adf_pvalue": spread_adf_pvalue,
            "spread_std": spread_std,
            "mean_notional": mean_notional,
            "roundtrip_cost_est": roundtrip_cost,
            "cost_to_expected_move": cost_ratio,
            "n_obs": len(spread),
        })
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    selected = pd.DataFrame(rows)
    if pair_selection != "raw":
        pcol = "pvalue_bonferroni" if pair_selection == "bonferroni" else "pvalue_fdr_bh"
        selected = selected.sort_values([pcol, "coint_pvalue", "half_life_days"]).head(top_n)
    else:
        selected = selected.sort_values("coint_pvalue").head(top_n)
    return selected.reset_index(drop=True)


# ----------------------------------------------------------------------
# Hedge-ratio models
# ----------------------------------------------------------------------
def rolling_ols_parameters(price_a: pd.Series, price_b: pd.Series,
                           window: int) -> tuple[pd.Series, pd.Series]:
    """Leakage-safe rolling OLS parameters; estimates may use today's close."""
    a = pd.to_numeric(price_a, errors="coerce")
    b = pd.to_numeric(price_b, errors="coerce")
    mean_a = a.rolling(window).mean()
    mean_b = b.rolling(window).mean()
    cov_ab = a.rolling(window).cov(b)
    var_b = b.rolling(window).var()
    beta = cov_ab / var_b.replace(0, np.nan)
    intercept = mean_a - beta * mean_b
    return beta, intercept


def kalman_hedge_parameters(price_a: pd.Series, price_b: pd.Series,
                            delta: float = 1e-5, observation_var: Optional[float] = None
                            ) -> tuple[pd.Series, pd.Series]:
    """Simple two-state Kalman regression for intercept/beta.

    State: [intercept, beta]. Transition is a random walk. The observation
    equation is A_t = intercept_t + beta_t * B_t + noise.
    """
    a = pd.to_numeric(price_a, errors="coerce")
    b = pd.to_numeric(price_b, errors="coerce")
    valid = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if valid.empty:
        return pd.Series(index=a.index, dtype=float), pd.Series(index=a.index, dtype=float)

    vals = valid.to_numpy(dtype=float)
    y0 = vals[:, 0]
    x0 = vals[:, 1]
    r = float(observation_var) if observation_var is not None else max(float(np.var(y0 - np.polyval(np.polyfit(x0, y0, 1), x0))), 1e-6)
    q = np.diag([delta, delta])
    theta = np.array([0.0, 0.0], dtype=float)
    P = np.eye(2) * 1e5
    betas = np.full(len(valid), np.nan)
    intercepts = np.full(len(valid), np.nan)

    for i, (y, x) in enumerate(zip(y0, x0)):
        P = P + q
        H = np.array([[1.0, x]])
        pred = float(H @ theta)
        S = float(H @ P @ H.T + r)
        K = (P @ H.T / S).reshape(-1)
        theta = theta + K * (y - pred)
        P = P - np.outer(K, H.reshape(-1)) @ P
        intercepts[i] = theta[0]
        betas[i] = theta[1]

    beta = pd.Series(index=valid.index, data=betas).reindex(a.index).ffill()
    intercept = pd.Series(index=valid.index, data=intercepts).reindex(a.index).ffill()
    return beta, intercept


def model_parameters(price_a: pd.Series, price_b: pd.Series, mode: str,
                     rolling_window: int, kalman_delta: float) -> tuple[pd.Series, pd.Series]:
    if mode == "static":
        beta, intercept = static_ols_hedge_ratio(price_a, price_b)
        return (
            pd.Series(beta, index=price_a.index, dtype=float),
            pd.Series(intercept, index=price_a.index, dtype=float),
        )
    if mode == "rolling":
        return rolling_ols_parameters(price_a, price_b, rolling_window)
    if mode == "kalman":
        return kalman_hedge_parameters(price_a, price_b, delta=kalman_delta)
    raise ValueError(f"Unknown hedge mode: {mode}")


def compute_model_spread(price_a: pd.Series, price_b: pd.Series,
                         beta: pd.Series, intercept: pd.Series) -> pd.Series:
    return price_a - beta * price_b - intercept


def compute_zscore_from_spread(spread: pd.Series, mode: str,
                               train_mean: Optional[float] = None,
                               train_std: Optional[float] = None,
                               rolling_window: int = 120) -> pd.Series:
    if mode == "static":
        if train_mean is None or train_std is None or train_std <= EPS:
            return pd.Series(np.nan, index=spread.index)
        return (spread - train_mean) / train_std

    if mode == "rolling":
        # Use only information available before today's close for normalization.
        mean = spread.shift(1).rolling(rolling_window, min_periods=max(20, rolling_window // 2)).mean()
        std = spread.shift(1).rolling(rolling_window, min_periods=max(20, rolling_window // 2)).std()
        return (spread - mean) / std.replace(0, np.nan)

    raise ValueError(f"Unknown zscore mode: {mode}")


# ----------------------------------------------------------------------
# Signal generation
# ----------------------------------------------------------------------
def generate_positions(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    cooldown_days: int = 5,
    reentry_z: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Return target position and a stop-event flag.

    A stop puts the pair into a re-arm state. Re-entry is disabled until both
    the cooldown has elapsed and |z| has returned below reentry_z.
    """
    if exit_z >= entry_z:
        raise ValueError("exit_z must be smaller than entry_z")

    positions = []
    stop_events = []
    position = 0
    cooldown = 0
    blocked = False

    for z in zscore:
        stop_event = False
        if cooldown > 0:
            cooldown -= 1

        if np.isnan(z):
            positions.append(0 if blocked else position)
            stop_events.append(False)
            continue

        az = abs(float(z))

        if blocked:
            if cooldown <= 0 and az <= reentry_z:
                blocked = False
            positions.append(0)
            stop_events.append(False)
            continue

        if position == 0:
            if z > entry_z:
                position = -1
            elif z < -entry_z:
                position = 1
        else:
            if az < exit_z:
                position = 0
            elif az > stop_z:
                position = 0
                blocked = True
                cooldown = max(int(cooldown_days), 0)
                stop_event = True

        positions.append(position)
        stop_events.append(stop_event)

    idx = zscore.index
    return pd.Series(positions, index=idx, dtype=float), pd.Series(stop_events, index=idx, dtype=bool)


# ----------------------------------------------------------------------
# Backtest engine
# ----------------------------------------------------------------------
@dataclass
class BacktestConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    cooldown_days: int = 5
    reentry_z: float = 1.0
    commission_bps: float = 5.0
    slippage_bps: float = 1.0
    short_borrow_bps: float = 0.0
    financing_bps: float = 0.0
    capital: float = 100_000.0
    risk_target: float = 0.10
    max_gross_leverage: float = 2.0
    hedge_mode: str = "static"
    hedge_window: int = 120
    kalman_delta: float = 1e-5
    zscore_mode: str = "static"
    zscore_window: int = 120


def _estimate_units_from_train(
    train_a: pd.Series,
    train_b: pd.Series,
    beta_train: float,
    spread_train: pd.Series,
    allocated_capital: float,
    risk_target: float,
    max_gross_leverage: float,
) -> tuple[float, float, float]:
    """Return fixed spread units, starting notional and annualized spread risk."""
    spread_diff_std = float(spread_train.diff().std(ddof=1))
    if not np.isfinite(spread_diff_std) or spread_diff_std <= EPS:
        units_risk = 1.0
        annual_spread_sigma = np.nan
    else:
        annual_spread_sigma = spread_diff_std * np.sqrt(TRADING_DAYS)
        units_risk = allocated_capital * risk_target / annual_spread_sigma

    start_notional = float(train_a.iloc[-1] + abs(beta_train) * train_b.iloc[-1])
    max_units = allocated_capital * max_gross_leverage / max(start_notional, EPS)
    units = float(max(min(units_risk, max_units), 0.0))
    if not np.isfinite(units) or units <= EPS:
        units = max_units if max_units > EPS else 1.0
    return units, start_notional, annual_spread_sigma


def _position_legs(position: float, units: float, beta: float,
                   price_a: float, price_b: float) -> tuple[float, float, float, float]:
    leg_a = position * units
    leg_b = -position * beta * units
    short_a = max(-leg_a, 0.0) * price_a
    short_b = max(-leg_b, 0.0) * price_b
    gross = abs(leg_a) * price_a + abs(leg_b) * price_b
    short_notional = short_a + short_b
    return leg_a, leg_b, gross, short_notional


def compute_trade_log(detail: pd.DataFrame) -> pd.DataFrame:
    """Trade runs based on the executed position, with explicit entry/exit costs."""
    p = detail["executed_position"]
    change = p.diff().fillna(p)
    trade_id = ((p != p.shift(1)) & (p != 0)).cumsum()

    rows = []
    active_id = None
    current = None
    for date, row in detail.iterrows():
        pos = float(row["executed_position"])
        prev = float(row["previous_position"])
        if active_id is None and pos != 0:
            active_id = int(trade_id.loc[date])
            current = {
                "trade_id": active_id,
                "entry_date": date,
                "direction": "Long spread" if pos > 0 else "Short spread",
                "holding_days": 0,
                "gross_pnl": 0.0,
                "transaction_cost": 0.0,
                "carry_cost": 0.0,
                "net_pnl": 0.0,
                "stopped": False,
            }
        if active_id is not None and pos != 0:
            current["holding_days"] += 1
            current["gross_pnl"] += float(row["gross_pnl"])
            current["transaction_cost"] += float(row["transaction_cost"])
            current["carry_cost"] += float(row["carry_cost"])
            if bool(row["stop_event"]):
                current["stopped"] = True
        if active_id is not None and pos == 0 and prev != 0:
            current["exit_date"] = date
            current["stopped_open_at_end"] = False
            current["total_cost"] = current["transaction_cost"] + current["carry_cost"]
            current["net_pnl"] = current["gross_pnl"] - current["total_cost"]
            rows.append(current)
            active_id = None
            current = None

    if current is not None:
        current["exit_date"] = detail.index[-1]
        current["stopped_open_at_end"] = True
        current["total_cost"] = current["transaction_cost"] + current["carry_cost"]
        current["net_pnl"] = current["gross_pnl"] - current["total_cost"]
        rows.append(current)

    return pd.DataFrame(rows)


def summarize_trades(trade_log: pd.DataFrame) -> dict:
    if trade_log.empty:
        return {
            "n_completed_trades": 0,
            "n_open_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "expectancy": np.nan,
            "avg_holding_days": np.nan,
            "max_trade_loss": np.nan,
        }
    closed = trade_log[~trade_log["stopped_open_at_end"]]
    wins = closed[closed["net_pnl"] > 0]
    losses = closed[closed["net_pnl"] <= 0]
    return {
        "n_completed_trades": int(len(closed)),
        "n_open_trades": int(len(trade_log) - len(closed)),
        "win_rate": float(len(wins) / len(closed)) if len(closed) else np.nan,
        "avg_win": float(wins["net_pnl"].mean()) if len(wins) else np.nan,
        "avg_loss": float(losses["net_pnl"].mean()) if len(losses) else np.nan,
        "profit_factor": _profit_factor(trade_log),
        "expectancy": float(closed["net_pnl"].mean()) if len(closed) else np.nan,
        "avg_holding_days": float(closed["holding_days"].mean()) if len(closed) else np.nan,
        "max_trade_loss": float(closed.loc[closed["net_pnl"] < 0, "net_pnl"].min())
            if (closed["net_pnl"] < 0).any() else 0.0,
    }


def _prepare_backtest_segment(
    seg_a: pd.Series,
    seg_b: pd.Series,
    train_a: pd.Series,
    train_b: pd.Series,
    config: BacktestConfig,
    allocated_capital: Optional[float] = None,
) -> dict:
    """Prepare everything that is independent of entry/exit thresholds.

    This is deliberately separated from the threshold-dependent simulation so
    a validation sweep does not repeatedly recompute rolling/Kalman hedge ratios.
    """
    seg = pd.concat([seg_a.rename("a"), seg_b.rename("b")], axis=1).dropna()
    seg = seg[(seg["a"] > 0) & (seg["b"] > 0)]
    seg_a = seg["a"]
    seg_b = seg["b"]
    if len(seg) < 5:
        return {"empty": True, "index": seg.index}

    capital = float(allocated_capital if allocated_capital is not None else config.capital)

    beta_train, intercept_train = static_ols_hedge_ratio(train_a, train_b)
    train_beta_series, train_intercept_series = model_parameters(
        train_a, train_b, config.hedge_mode, config.hedge_window, config.kalman_delta
    )

    if config.hedge_mode == "static":
        beta_test = pd.Series(beta_train, index=seg.index, dtype=float)
        intercept_test = pd.Series(intercept_train, index=seg.index, dtype=float)
    else:
        combined_a = pd.concat([train_a, seg_a])
        combined_b = pd.concat([train_b, seg_b])
        beta_all, intercept_all = model_parameters(
            combined_a, combined_b, config.hedge_mode, config.hedge_window, config.kalman_delta
        )
        beta_test = beta_all.reindex(seg.index)
        intercept_test = intercept_all.reindex(seg.index)
        beta_test = beta_test.replace([np.inf, -np.inf], np.nan).ffill().fillna(beta_train)
        intercept_test = intercept_test.replace([np.inf, -np.inf], np.nan).ffill().fillna(intercept_train)

    train_spread = compute_model_spread(train_a, train_b, train_beta_series, train_intercept_series).dropna()
    if len(train_spread) < 30:
        return {"empty": True, "index": seg.index}

    train_mean = float(train_spread.mean())
    train_std = float(train_spread.std(ddof=1))
    spread = compute_model_spread(seg_a, seg_b, beta_test, intercept_test)
    z = compute_zscore_from_spread(
        spread, config.zscore_mode,
        train_mean=train_mean, train_std=train_std,
        rolling_window=config.zscore_window,
    )

    units, start_notional, annual_spread_sigma = _estimate_units_from_train(
        train_a, train_b, beta_train, train_spread,
        capital, config.risk_target, config.max_gross_leverage,
    )

    return {
        "empty": False,
        "seg_a": seg_a,
        "seg_b": seg_b,
        "beta_train": beta_train,
        "intercept_train": intercept_train,
        "beta_test": beta_test,
        "intercept_test": intercept_test,
        "train_mean": train_mean,
        "train_std": train_std,
        "spread": spread,
        "z": z,
        "units": units,
        "start_notional": start_notional,
        "annual_spread_sigma": annual_spread_sigma,
        "capital": capital,
    }


def _run_backtest_on_segment(
    seg_a: pd.Series,
    seg_b: pd.Series,
    train_a: pd.Series,
    train_b: pd.Series,
    config: BacktestConfig,
    allocated_capital: Optional[float] = None,
    prepared: Optional[dict] = None,
) -> dict:
    """Backtest a segment with a train-fitted model and next-close execution."""
    if prepared is None:
        prepared = _prepare_backtest_segment(
            seg_a, seg_b, train_a, train_b, config, allocated_capital=allocated_capital,
        )
    if prepared.get("empty", False):
        return _empty_result(prepared["index"])

    seg_a = prepared["seg_a"]
    seg_b = prepared["seg_b"]
    beta_train = prepared["beta_train"]
    intercept_train = prepared["intercept_train"]
    beta_test = prepared["beta_test"]
    intercept_test = prepared["intercept_test"]
    train_mean = prepared["train_mean"]
    train_std = prepared["train_std"]
    spread = prepared["spread"]
    z = prepared["z"]
    capital = prepared["capital"]

    target_position, stop_events = generate_positions(
        z,
        config.entry_z,
        config.exit_z,
        config.stop_z,
        config.cooldown_days,
        config.reentry_z,
    )

    # Position generated at close t is executed at close t+1.
    executed_position = target_position.shift(1).fillna(0.0)
    previous_position = executed_position.shift(1).fillna(0.0)

    # Fixed spread-unit sizing based only on train information.
    units = prepared["units"]
    start_notional = prepared["start_notional"]
    annual_spread_sigma = prepared["annual_spread_sigma"]

    # The signal observed at close t is executed at close t+1. Therefore the
    # executable portfolio carried during interval t+1 -> t+2 is the position
    # and hedge ratio estimated at t. This is especially important for rolling
    # or Kalman beta: changing beta implies actual underlying-leg turnover.
    executed_beta = beta_test.shift(1).fillna(beta_train)
    leg_a = executed_position * units
    leg_b = -executed_position * executed_beta * units

    delta_a_units = leg_a - leg_a.shift(1).fillna(0.0)
    delta_b_units = leg_b - leg_b.shift(1).fillna(0.0)

    next_price_a = seg_a.shift(-1)
    next_price_b = seg_b.shift(-1)
    gross_pnl = (
        leg_a * (next_price_a - seg_a)
        + leg_b * (next_price_b - seg_b)
    ).fillna(0.0)

    # Costs charged at the execution close, using execution-day prices.  This
    # includes opening/closing the pair AND hedge-ratio rebalancing.
    commission_rate = (config.commission_bps + config.slippage_bps) / 10_000.0
    transaction_notional = (
        delta_a_units.abs() * seg_a + delta_b_units.abs() * seg_b
    )
    transaction_cost = commission_rate * transaction_notional

    # Carry costs accrue while the position is open.
    gross_open_notional = (leg_a.abs() * seg_a + leg_b.abs() * seg_b)
    short_notional = (
        np.maximum(-leg_a, 0.0) * seg_a + np.maximum(-leg_b, 0.0) * seg_b
    )
    carry_cost = (
        gross_open_notional * config.financing_bps / 10_000.0 / TRADING_DAYS
        + short_notional * config.short_borrow_bps / 10_000.0 / TRADING_DAYS
    )

    net_pnl = gross_pnl - transaction_cost - carry_cost
    equity_curve = capital + net_pnl.cumsum()
    capital_exhausted_mask = equity_curve <= 0
    capital_exhausted = bool(capital_exhausted_mask.any())
    capital_exhausted_date = None

    if capital_exhausted:
        idx = capital_exhausted_mask.idxmax()
        capital_exhausted_date = idx
        keep = equity_curve.index <= idx
        seg_a = seg_a.loc[keep]
        seg_b = seg_b.loc[keep]
        beta_test = beta_test.loc[keep]
        intercept_test = intercept_test.loc[keep]
        executed_beta = executed_beta.loc[keep]
        spread = spread.loc[keep]
        z = z.loc[keep]
        target_position = target_position.loc[keep]
        executed_position = executed_position.loc[keep]
        previous_position = previous_position.loc[keep]
        stop_events = stop_events.loc[keep]
        gross_pnl = gross_pnl.loc[keep]
        transaction_cost = transaction_cost.loc[keep]
        carry_cost = carry_cost.loc[keep]
        net_pnl = net_pnl.loc[keep]
        equity_curve = equity_curve.loc[keep]

    # Fixed-capital return accounting: no artificial daily compounding.
    daily_return = net_pnl / capital
    gross_daily_return = gross_pnl / capital
    total_return = float(net_pnl.sum() / capital) if len(net_pnl) else np.nan
    gross_total_return = float(gross_pnl.sum() / capital) if len(gross_pnl) else np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_dd = max_drawdown = float(max(drawdown.min(), -1.0)) if len(drawdown) else np.nan

    detail = pd.DataFrame({
        "price_a": seg_a,
        "price_b": seg_b,
        "beta": executed_beta,
        "signal_beta": beta_test,
        "intercept": intercept_test,
        "spread": spread,
        "zscore": z,
        "target_position": target_position,
        "executed_position": executed_position,
        "previous_position": previous_position,
        "stop_event": stop_events,
        "gross_pnl": gross_pnl,
        "transaction_cost": transaction_cost,
        "carry_cost": carry_cost,
        "net_pnl": net_pnl,
        "equity_curve": equity_curve,
        "daily_return": daily_return,
        "gross_daily_return": gross_daily_return,
    })
    trade_log = compute_trade_log(detail)
    trade_stats = summarize_trades(trade_log)

    gross_pnl_total = float(gross_pnl.sum())
    tx_total = float(transaction_cost.sum())
    carry_total = float(carry_cost.sum())
    total_cost = tx_total + carry_total

    return {
        "beta": beta_train,
        "intercept": intercept_train,
        "train_spread_mean": train_mean,
        "train_spread_std": train_std,
        "units": units,
        "start_notional": start_notional,
        "annual_spread_sigma": annual_spread_sigma,
        "capital_base": capital,
        "gross_sharpe": _annualized_sharpe(gross_daily_return),
        "sharpe": _annualized_sharpe(daily_return),
        "sortino": _sortino(daily_return),
        "annualized_vol": _annualized_vol(daily_return),
        "total_return": total_return,
        "gross_total_return": gross_total_return,
        "max_drawdown": max_dd,
        "max_drawdown_days": _max_drawdown_duration(equity_curve),
        "n_entries": int(((executed_position != 0) & (previous_position == 0)).sum()),
        "n_exits": int(((executed_position == 0) & (previous_position != 0)).sum()),
        "n_flips": int(((executed_position != 0) & (previous_position != 0) &
                         (np.sign(executed_position) != np.sign(previous_position))).sum()),
        "total_gross_pnl": gross_pnl_total,
        "total_transaction_cost": tx_total,
        "total_carry_cost": carry_total,
        "total_cost": total_cost,
        "cost_pct_of_abs_gross": total_cost / abs(gross_pnl_total) if abs(gross_pnl_total) > EPS else np.nan,
        "capital_exhausted": capital_exhausted,
        "capital_exhausted_date": capital_exhausted_date,
        "detail": detail,
        "trade_log": trade_log,
        "trade_stats": trade_stats,
    }


def _empty_result(index: pd.Index) -> dict:
    detail = pd.DataFrame(index=index)
    empty = {
        "beta": np.nan, "intercept": np.nan, "train_spread_mean": np.nan,
        "train_spread_std": np.nan, "units": np.nan, "start_notional": np.nan,
        "annual_spread_sigma": np.nan, "capital_base": np.nan,
        "gross_sharpe": np.nan, "sharpe": np.nan, "sortino": np.nan,
        "annualized_vol": np.nan, "total_return": np.nan, "gross_total_return": np.nan,
        "max_drawdown": np.nan, "max_drawdown_days": 0,
        "n_entries": 0, "n_exits": 0, "n_flips": 0,
        "total_gross_pnl": np.nan, "total_transaction_cost": np.nan,
        "total_carry_cost": np.nan, "total_cost": np.nan, "cost_pct_of_abs_gross": np.nan,
        "capital_exhausted": False, "capital_exhausted_date": None,
        "detail": detail, "trade_log": pd.DataFrame(), "trade_stats": summarize_trades(pd.DataFrame()),
    }
    return empty


# ----------------------------------------------------------------------
# Train -> validation -> test for one pair
# ----------------------------------------------------------------------
def tune_pair(
    train: pd.DataFrame,
    val: pd.DataFrame,
    pair: tuple[str, str],
    base_config: BacktestConfig,
    entry_grid: Sequence[float],
    exit_grid: Sequence[float],
    min_validation_trades: int = 3,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    a, b = pair
    # Expensive hedge/z-score preparation is identical across threshold cells.
    prepared = _prepare_backtest_segment(
        val[a], val[b], train[a], train[b], base_config, allocated_capital=base_config.capital,
    )
    if prepared.get("empty", False):
        return {"entry_z": np.nan, "exit_z": np.nan, "net_sharpe": np.nan}, pd.DataFrame()

    for entry in entry_grid:
        for exit_ in exit_grid:
            if exit_ >= entry:
                continue
            cfg = BacktestConfig(**vars(base_config))
            cfg.entry_z = float(entry)
            cfg.exit_z = float(exit_)
            res = _run_backtest_on_segment(
                val[a], val[b], train[a], train[b], cfg,
                allocated_capital=cfg.capital, prepared=prepared,
            )
            rows.append({
                "Ticker A": a,
                "Ticker B": b,
                "entry_z": entry,
                "exit_z": exit_,
                "net_sharpe": res["sharpe"],
                "gross_sharpe": res["gross_sharpe"],
                "total_return": res["total_return"],
                "max_drawdown": res["max_drawdown"],
                "completed_trades": res["trade_stats"]["n_completed_trades"],
                "cost_pct_of_abs_gross": res["cost_pct_of_abs_gross"],
            })
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["net_sharpe"]) if not df.empty else df
    if valid.empty:
        return {"entry_z": np.nan, "exit_z": np.nan, "net_sharpe": np.nan}, df
    # Require several completed trades so a one-trade high Sharpe does not win.
    valid = valid[valid["completed_trades"] >= int(min_validation_trades)]
    if valid.empty:
        return {"entry_z": np.nan, "exit_z": np.nan, "net_sharpe": np.nan}, df
    # Prefer robust combinations: maximize Sharpe, then lower cost, then more trades.
    best = valid.sort_values(
        ["net_sharpe", "cost_pct_of_abs_gross", "completed_trades"],
        ascending=[False, True, False],
    ).iloc[0]
    return best.to_dict(), df


# ----------------------------------------------------------------------
# Walk-forward individual-pair evaluation
# ----------------------------------------------------------------------
def build_fold_boundaries(n: int, initial_train_frac: float, n_folds: int,
                          val_frac_within_fold: float) -> list[dict]:
    initial_train_end = int(n * initial_train_frac)
    remaining = n - initial_train_end
    if remaining < n_folds * 30:
        raise ValueError("Not enough observations for the requested walk-forward folds.")
    fold_size = remaining // n_folds
    folds = []
    for k in range(n_folds):
        train_end = initial_train_end + k * fold_size
        fold_end = initial_train_end + (k + 1) * fold_size if k < n_folds - 1 else n
        val_size = int((fold_end - train_end) * val_frac_within_fold)
        val_end = train_end + val_size
        if train_end < 30 or val_end <= train_end or fold_end <= val_end:
            continue
        folds.append({"fold": k, "train_end": train_end, "val_end": val_end, "fold_end": fold_end})
    return folds


def walk_forward_pair(
    prices: pd.DataFrame,
    pair: tuple[str, str],
    pair_selection_candidates: Optional[pd.DataFrame],
    config: BacktestConfig,
    entry_grid: Sequence[float],
    exit_grid: Sequence[float],
    n_folds: int,
    initial_train_frac: float,
    val_frac_within_fold: float,
    train_candidate_reselection: bool = False,
) -> dict:
    """Walk-forward a single preselected pair.

    The flag train_candidate_reselection exists for portfolio orchestration;
    pair-specific candidate selection itself happens in the portfolio function.
    """
    a, b = pair
    p = prices[[a, b]].dropna()
    folds = build_fold_boundaries(len(p), initial_train_frac, n_folds, val_frac_within_fold)
    fold_rows = []
    stitched_returns = []

    for f in folds:
        train = p.iloc[:f["train_end"]]
        val = p.iloc[f["train_end"]:f["val_end"]]
        test = p.iloc[f["val_end"]:f["fold_end"]]
        best, val_df = tune_pair(train, val, pair, config, entry_grid, exit_grid, min_validation_trades=3)
        if pd.isna(best["entry_z"]):
            fold_rows.append({"fold": f["fold"], "pair": f"{a}/{b}", "test_sharpe": np.nan})
            continue
        fold_cfg = BacktestConfig(**vars(config))
        fold_cfg.entry_z = float(best["entry_z"])
        fold_cfg.exit_z = float(best["exit_z"])
        test_res = _run_backtest_on_segment(test[a], test[b], train[a], train[b], fold_cfg, allocated_capital=config.capital)
        fold_rows.append({
            "fold": f["fold"],
            "pair": f"{a}/{b}",
            "train_end": train.index[-1],
            "val_start": val.index[0],
            "val_end": val.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "chosen_entry_z": fold_cfg.entry_z,
            "chosen_exit_z": fold_cfg.exit_z,
            "validation_sharpe": best["net_sharpe"],
            "test_sharpe": test_res["sharpe"],
            "test_total_return": test_res["total_return"],
            "test_max_drawdown": test_res["max_drawdown"],
            "test_completed_trades": test_res["trade_stats"]["n_completed_trades"],
            "capital_exhausted": test_res["capital_exhausted"],
            "coint_pvalue_train": np.nan,
        })
        if not test_res["detail"].empty:
            stitched_returns.append(test_res["detail"]["daily_return"])

    if stitched_returns:
        combined = pd.concat(stitched_returns).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        equity = (1.0 + combined).cumprod()
        stitched = {
            "sharpe": _annualized_sharpe(combined),
            "sortino": _sortino(combined),
            "annualized_vol": _annualized_vol(combined),
            "total_return": float(equity.iloc[-1] - 1.0),
            "max_drawdown": _max_drawdown(equity),
        }
    else:
        stitched = {"sharpe": np.nan, "sortino": np.nan, "annualized_vol": np.nan,
                    "total_return": np.nan, "max_drawdown": np.nan}

    return {"fold_df": pd.DataFrame(fold_rows), "stitched": stitched}


# ----------------------------------------------------------------------
# Portfolio walk-forward
# ----------------------------------------------------------------------
def _pair_candidates_for_fold(
    train_prices: pd.DataFrame,
    pair_selection: str,
    significance: float,
    top_n: int,
    pair_universe: Optional[set[tuple[str, str]]],
    selection_kwargs: dict,
) -> pd.DataFrame:
    selected = compute_train_pair_candidates(
        train_prices,
        pair_selection=pair_selection,
        significance=significance,
        top_n=top_n,
        pair_universe=pair_universe,
        **selection_kwargs,
    )
    return selected


def _capped_weights(raw_weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Project positive weights onto the simplex with an upper cap."""
    w = np.asarray(raw_weights, dtype=float)
    w[~np.isfinite(w)] = 0.0
    n = len(w)
    if n == 0:
        return w
    # A strict per-pair cap can leave some capital in cash when too few
    # independent pairs are available. That is preferable to violating the
    # risk limit by renormalizing weights back above the cap.
    if max_weight <= 0:
        return np.zeros(n)
    if np.all(w <= 0):
        w = np.ones(n)
    w = np.maximum(w, 0.0)
    w /= w.sum()

    result = np.zeros(n)
    free = np.ones(n, dtype=bool)
    remaining = 1.0
    while free.any():
        base = w[free]
        if base.sum() <= EPS:
            base = np.ones(base.size)
        proposed = remaining * base / base.sum()
        over = proposed > max_weight + 1e-12
        idx = np.flatnonzero(free)
        if not over.any():
            result[idx] = proposed
            break
        result[idx[over]] = max_weight
        remaining -= max_weight * int(over.sum())
        free[idx[over]] = False
        if remaining <= EPS:
            break
    return result


def walk_forward_portfolio(
    prices: pd.DataFrame,
    pair_selection: str,
    significance: float,
    top_n: int,
    pair_universe: Optional[set[tuple[str, str]]],
    config: BacktestConfig,
    entry_grid: Sequence[float],
    exit_grid: Sequence[float],
    n_folds: int,
    initial_train_frac: float,
    val_frac_within_fold: float,
    portfolio_capital: float,
    max_pair_weight: float,
    selection_kwargs: dict,
    min_validation_trades: int = 2,
    min_test_trades: int = 2,
) -> dict:
    folds = build_fold_boundaries(len(prices), initial_train_frac, n_folds, val_frac_within_fold)
    portfolio_daily_returns = []
    selected_rows = []
    fold_summaries = []

    for f in folds:
        print(f"  Fold {f['fold'] + 1}/{len(folds)}: selecting pairs on train...", flush=True)
        train = prices.iloc[:f["train_end"]]
        val = prices.iloc[f["train_end"]:f["val_end"]]
        test = prices.iloc[f["val_end"]:f["fold_end"]]
        candidates = _pair_candidates_for_fold(
            train, pair_selection, significance, top_n, pair_universe, selection_kwargs,
        )
        if candidates.empty:
            fold_summaries.append({"fold": f["fold"], "selected_pairs": 0})
            continue

        pair_results = []
        # Tune each selected pair independently on this fold's validation segment.
        for pair_idx, (_, row) in enumerate(candidates.iterrows(), start=1):
            print(f"    Tuning pair {pair_idx}/{len(candidates)}: {row['Ticker A']}/{row['Ticker B']}", flush=True)
            pair = (row["Ticker A"], row["Ticker B"])
            if pair[0] not in val.columns or pair[1] not in val.columns:
                continue
            best, _ = tune_pair(
                train[[pair[0], pair[1]]], val[[pair[0], pair[1]]], pair, config, entry_grid, exit_grid,
                min_validation_trades=min_validation_trades,
            )
            if pd.isna(best["entry_z"]):
                continue
            pair_cfg = BacktestConfig(**vars(config))
            pair_cfg.entry_z = float(best["entry_z"])
            pair_cfg.exit_z = float(best["exit_z"])
            # Weight by inverse train spread volatility, capped per pair.
            beta, intercept = static_ols_hedge_ratio(train[pair[0]], train[pair[1]])
            train_spread = compute_spread(train[pair[0]], train[pair[1]], beta, intercept).dropna()
            spread_ann_sigma = train_spread.diff().std(ddof=1) * np.sqrt(TRADING_DAYS)
            risk_score = 1.0 / max(float(spread_ann_sigma), EPS) if np.isfinite(spread_ann_sigma) else 0.0
            pair_results.append({
                "pair": pair,
                "cfg": pair_cfg,
                "risk_score": risk_score,
                "selection_row": row,
            })

        if not pair_results:
            fold_summaries.append({"fold": f["fold"], "selected_pairs": 0})
            continue

        weights_raw = np.array([x["risk_score"] for x in pair_results], dtype=float)
        weights = _capped_weights(weights_raw, max_pair_weight)

        fold_returns = []
        for idx, item in enumerate(pair_results):
            pair = item["pair"]
            allocated = portfolio_capital * float(weights[idx])
            res = _run_backtest_on_segment(
                test[pair[0]], test[pair[1]], train[pair[0]], train[pair[1]],
                item["cfg"], allocated_capital=allocated,
            )
            if not res["detail"].empty:
                sr = res["detail"]["daily_return"].reindex(test.index, fill_value=0.0)
                fold_returns.append(float(weights[idx]) * sr)
            selected_rows.append({
                "fold": f["fold"],
                "Ticker A": pair[0],
                "Ticker B": pair[1],
                "weight": float(weights[idx]),
                "train_coint_pvalue": item["selection_row"]["coint_pvalue"],
                "train_fdr_pvalue": item["selection_row"]["pvalue_fdr_bh"],
                "half_life_days": item["selection_row"]["half_life_days"],
                "chosen_entry_z": item["cfg"].entry_z,
                "chosen_exit_z": item["cfg"].exit_z,
                "pair_test_sharpe": res["sharpe"],
                "pair_test_return": res["total_return"],
                "pair_test_max_dd": res["max_drawdown"],
                "completed_trades": res["trade_stats"]["n_completed_trades"],
                "trade_sufficient": bool(res["trade_stats"]["n_completed_trades"] >= min_test_trades),
            })

        if fold_returns:
            fold_daily = pd.concat(fold_returns, axis=1).sum(axis=1)
            portfolio_daily_returns.append(fold_daily)
            fold_selected = [r for r in selected_rows if r["fold"] == f["fold"]]
            fold_summaries.append({
                "fold": f["fold"],
                "selected_pairs": len(pair_results),
                "sufficient_trade_pairs": int(sum(bool(r["trade_sufficient"]) for r in fold_selected)),
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "fold_sharpe": _annualized_sharpe(fold_daily),
                "fold_return": float((1 + fold_daily).prod() - 1),
                "fold_max_drawdown": _max_drawdown((1 + fold_daily).cumprod()),
            })

    if portfolio_daily_returns:
        combined = pd.concat(portfolio_daily_returns).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        equity = (1.0 + combined).cumprod()
        metrics = {
            "sharpe": _annualized_sharpe(combined),
            "sortino": _sortino(combined),
            "annualized_vol": _annualized_vol(combined),
            "total_return": float(equity.iloc[-1] - 1.0),
            "max_drawdown": _max_drawdown(equity),
            "max_drawdown_days": _max_drawdown_duration(equity),
            "equity": equity,
            "daily_returns": combined,
        }
    else:
        metrics = {
            "sharpe": np.nan, "sortino": np.nan, "annualized_vol": np.nan,
            "total_return": np.nan, "max_drawdown": np.nan, "max_drawdown_days": 0,
            "equity": None, "daily_returns": pd.Series(dtype=float),
        }

    return {
        "metrics": metrics,
        "selected_pairs": pd.DataFrame(selected_rows),
        "folds": pd.DataFrame(fold_summaries),
    }


# ----------------------------------------------------------------------
# Walk-forward with PAIR RESELECTION inside each TRAIN fold
# ----------------------------------------------------------------------
def walk_forward_reselect_pairs(
    prices: pd.DataFrame,
    pair_selection: str,
    significance: float,
    top_n: int,
    pair_universe: Optional[set[tuple[str, str]]],
    config: BacktestConfig,
    entry_grid: Sequence[float],
    exit_grid: Sequence[float],
    n_folds: int,
    initial_train_frac: float,
    val_frac_within_fold: float,
    selection_kwargs: dict,
    min_validation_trades: int = 2,
    min_test_trades: int = 2,
) -> dict:
    """Repeat pair discovery, tuning and final testing every fold.

    The returned pair summary contains both active-fold statistics and a
    stitched walk-forward equity curve in which a pair earns zero return on
    folds where it was not selected. This avoids treating a pair selected in
    only one period as if that were a continuous strategy.
    """
    folds = build_fold_boundaries(len(prices), initial_train_frac, n_folds, val_frac_within_fold)
    fold_rows = []
    pair_rows = []
    all_candidate_rows = []
    pair_return_series: dict[tuple[str, str], dict[int, pd.Series]] = {}
    pair_selected_folds: dict[tuple[str, str], set[int]] = {}

    for fold_pos, f in enumerate(folds, start=1):
        print(f"  Fold {fold_pos}/{len(folds)}: selecting pairs on train...", flush=True)
        train = prices.iloc[:f["train_end"]]
        val = prices.iloc[f["train_end"]:f["val_end"]]
        test = prices.iloc[f["val_end"]:f["fold_end"]]

        candidates = compute_train_pair_candidates(
            train,
            pair_selection=pair_selection,
            significance=significance,
            top_n=top_n,
            pair_universe=pair_universe,
            **selection_kwargs,
        )
        selected_pair_count = len(candidates)
        # A pair is considered selected for this fold before threshold tuning.
        # Its walk-forward return is zero unless a valid test backtest produces
        # actual returns, which correctly models being flat when not traded.
        for a, b in zip(candidates.get("Ticker A", []), candidates.get("Ticker B", [])):
            key = _pair_key(a, b)
            pair_return_series.setdefault(key, {})[f["fold"]] = pd.Series(0.0, index=test.index)
            pair_selected_folds.setdefault(key, set()).add(f["fold"])

        for pair_idx, (_, c) in enumerate(candidates.iterrows(), start=1):
            pair = (c["Ticker A"], c["Ticker B"])
            print(f"    Fold {fold_pos}: validating/testing pair {pair_idx}/{len(candidates)}: {pair[0]}/{pair[1]}", flush=True)
            best, _ = tune_pair(
                train[[pair[0], pair[1]]],
                val[[pair[0], pair[1]]],
                pair, config, entry_grid, exit_grid,
                min_validation_trades=min_validation_trades,
            )
            if pd.isna(best["entry_z"]):
                pair_rows.append({
                    "fold": f["fold"],
                    "Ticker A": pair[0],
                    "Ticker B": pair[1],
                    "train_end": train.index[-1],
                    "val_start": val.index[0],
                    "val_end": val.index[-1],
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    "train_coint_pvalue": c["coint_pvalue"],
                    "train_bonf_pvalue": c["pvalue_bonferroni"],
                    "train_fdr_pvalue": c["pvalue_fdr_bh"],
                    "train_half_life": c["half_life_days"],
                    "train_coint_stability": c["coint_stability"],
                    "validation_sharpe": np.nan,
                    "chosen_entry_z": np.nan,
                    "chosen_exit_z": np.nan,
                    "test_sharpe": np.nan,
                    "test_sortino": np.nan,
                    "test_return": 0.0,
                    "test_max_drawdown": 0.0,
                    "test_completed_trades": 0,
                    "test_cost_pct": np.nan,
                    "capital_exhausted": False,
                    "test_fold_trade_sufficient": False,
                })
                continue

            pair_cfg = BacktestConfig(**vars(config))
            pair_cfg.entry_z = float(best["entry_z"])
            pair_cfg.exit_z = float(best["exit_z"])
            test_res = _run_backtest_on_segment(
                test[pair[0]], test[pair[1]],
                train[pair[0]], train[pair[1]],
                pair_cfg, allocated_capital=config.capital,
            )
            stats = test_res["trade_stats"]
            completed = int(stats["n_completed_trades"])
            sufficient = completed >= int(min_test_trades)

            # Replace the zero placeholder with actual test returns for this
            # pair/fold, even when there are too few trades.
            key = _pair_key(*pair)
            series = test_res["detail"].get("daily_return", pd.Series(dtype=float)).reindex(test.index, fill_value=0.0)
            pair_return_series.setdefault(key, {})[f["fold"]] = series

            pair_rows.append({
                "fold": f["fold"],
                "Ticker A": pair[0],
                "Ticker B": pair[1],
                "train_end": train.index[-1],
                "val_start": val.index[0],
                "val_end": val.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "train_coint_pvalue": c["coint_pvalue"],
                "train_bonf_pvalue": c["pvalue_bonferroni"],
                "train_fdr_pvalue": c["pvalue_fdr_bh"],
                "train_half_life": c["half_life_days"],
                "train_coint_stability": c["coint_stability"],
                "validation_sharpe": best["net_sharpe"],
                "chosen_entry_z": pair_cfg.entry_z,
                "chosen_exit_z": pair_cfg.exit_z,
                "test_sharpe": test_res["sharpe"],
                "test_sortino": test_res["sortino"],
                "test_return": test_res["total_return"],
                "test_max_drawdown": test_res["max_drawdown"],
                "test_completed_trades": completed,
                "test_cost_pct": test_res["cost_pct_of_abs_gross"],
                "capital_exhausted": test_res["capital_exhausted"],
                "test_fold_trade_sufficient": sufficient,
            })

        if not candidates.empty:
            csave = candidates.copy()
            csave.insert(0, "fold", f["fold"])
            all_candidate_rows.append(csave)

        fold_pair_rows = [
            r for r in pair_rows if r["fold"] == f["fold"] and pd.notna(r.get("test_sharpe"))
        ]
        sufficient_rows = [r for r in fold_pair_rows if r.get("test_fold_trade_sufficient", False)]
        fold_rows.append({
            "fold": f["fold"],
            "train_end": train.index[-1],
            "val_end": val.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "selected_pairs": selected_pair_count,
            "positive_test_sharpes": int(sum(r["test_sharpe"] > 0 for r in sufficient_rows)),
            "sufficient_trade_pairs": len(sufficient_rows),
            "median_test_sharpe": float(np.median([r["test_sharpe"] for r in sufficient_rows])) if sufficient_rows else np.nan,
            "mean_test_return": float(np.mean([r["test_return"] for r in sufficient_rows])) if sufficient_rows else np.nan,
        })

    pair_detail = pd.DataFrame(pair_rows)
    available_folds = len(folds)
    summary_rows = []
    for pair, selected_set in pair_selected_folds.items():
        a, b = pair
        sub = pair_detail[(pair_detail["Ticker A"] == a) & (pair_detail["Ticker B"] == b)].copy()
        if sub.empty:
            continue
        active = sub[pd.notna(sub["test_sharpe"])].copy()
        sufficient = active[active["test_fold_trade_sufficient"]].copy()
        positive_active = int((active["test_sharpe"] > 0).sum())
        positive_sufficient = int((sufficient["test_sharpe"] > 0).sum())
        selected_count = len(selected_set)

        if pair in pair_return_series and pair_return_series[pair]:
            series_blocks = []
            for f in folds:
                block = pair_return_series[pair].get(f["fold"])
                if block is None:
                    test_index = prices.iloc[f["val_end"]:f["fold_end"]].index
                    block = pd.Series(0.0, index=test_index)
                series_blocks.append(block.reindex(prices.iloc[f["val_end"]:f["fold_end"]].index, fill_value=0.0))
            stitched_returns = pd.concat(series_blocks).sort_index()
            stitched_returns = stitched_returns[~stitched_returns.index.duplicated(keep="first")]
            equity = (1.0 + stitched_returns).cumprod()
            wf_sharpe = _annualized_sharpe(stitched_returns)
            wf_sortino = _sortino(stitched_returns)
            wf_return = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
            wf_cagr = _cagr_from_equity(equity)
            wf_dd = _max_drawdown(equity)
        else:
            stitched_returns = pd.Series(dtype=float)
            wf_sharpe = wf_sortino = wf_return = wf_cagr = wf_dd = np.nan

        summary_rows.append({
            "Ticker A": a,
            "Ticker B": b,
            "available_folds": available_folds,
            "selected_folds": selected_count,
            "selected_fraction": selected_count / available_folds if available_folds else np.nan,
            "valid_test_folds": len(active),
            "positive_folds": positive_active,
            "positive_fold_fraction": positive_active / len(active) if len(active) else np.nan,
            "sufficient_trade_folds": len(sufficient),
            "positive_sufficient_folds": positive_sufficient,
            "positive_sufficient_fold_fraction": positive_sufficient / len(sufficient) if len(sufficient) else np.nan,
            "mean_test_sharpe": float(active["test_sharpe"].mean()) if len(active) else np.nan,
            "median_test_sharpe": float(active["test_sharpe"].median()) if len(active) else np.nan,
            "worst_test_sharpe": float(active["test_sharpe"].min()) if len(active) else np.nan,
            "walk_forward_sharpe": wf_sharpe,
            "walk_forward_sortino": wf_sortino,
            "walk_forward_return": wf_return,
            "walk_forward_cagr": wf_cagr,
            "walk_forward_max_drawdown": wf_dd,
            "mean_test_return": float(active["test_return"].mean()) if len(active) else np.nan,
            "worst_test_drawdown": float(active["test_max_drawdown"].min()) if len(active) else np.nan,
            "mean_cost_pct": float(active["test_cost_pct"].mean()) if len(active) else np.nan,
            "median_cost_pct": float(active["test_cost_pct"].median()) if len(active) else np.nan,
            "mean_completed_trades": float(active["test_completed_trades"].mean()) if len(active) else np.nan,
            "median_completed_trades": float(active["test_completed_trades"].median()) if len(active) else np.nan,
            "total_completed_trades": int(active["test_completed_trades"].sum()) if len(active) else 0,
            "min_trade_rule": int(min_test_trades),
        })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        # Three-level evidence classification.
        #
        # robust: enough independent walk-forward evidence, enough trades,
        #         at least half of the trade-sufficient folds profitable,
        #         and a positive/meaningful stitched out-of-sample result.
        # watchlist: some economically interesting evidence, but not enough
        #            consistency/trades to call it robust.
        # insufficient_evidence: too little usable evidence or clearly
        #                         unconvincing stitched performance.
        robust_mask = (
            (summary["selected_folds"] >= 2)
            & (summary["sufficient_trade_folds"] >= 2)
            & (summary["positive_sufficient_fold_fraction"] >= 0.5)
            & (summary["total_completed_trades"] >= 5)
            & (summary["walk_forward_sharpe"] >= 0.50)
            & (summary["walk_forward_return"] > 0.0)
            & (summary["walk_forward_max_drawdown"] > -0.25)
            & (summary["mean_cost_pct"] < 0.30)
        )

        watchlist_mask = (
            ~robust_mask
            & (summary["sufficient_trade_folds"] >= 1)
            & (summary["total_completed_trades"] >= 2)
            & (summary["walk_forward_sharpe"] > 0.0)
            & (
                (summary["selected_folds"] >= 2)
                | (summary["walk_forward_sharpe"] >= 0.50)
            )
        )

        summary["evidence_class"] = np.select(
            [robust_mask, watchlist_mask],
            ["robust", "watchlist"],
            default="insufficient_evidence",
        )
        # Kept for backward compatibility with scripts that consumed the old flag.
        summary["robust_candidate"] = summary["evidence_class"].eq("robust")
        summary["classification_reason"] = np.select(
            [
                summary["evidence_class"].eq("robust"),
                summary["evidence_class"].eq("watchlist"),
            ],
            [
                "multi-fold, trade-sufficient, positive walk-forward evidence",
                "promising out-of-sample evidence, but insufficient consistency/trade count for robust classification",
            ],
            default="insufficient usable evidence or negative/weak stitched walk-forward performance",
        )
        summary = summary.sort_values(
            ["evidence_class", "walk_forward_sharpe", "sufficient_trade_folds", "total_completed_trades"],
            key=lambda col: col.map({"robust": 0, "watchlist": 1, "insufficient_evidence": 2}) if col.name == "evidence_class" else col,
            ascending=[True, False, False, False],
        )

    candidate_detail = pd.concat(all_candidate_rows, ignore_index=True) if all_candidate_rows else pd.DataFrame()
    return {
        "folds": pd.DataFrame(fold_rows),
        "pair_detail": pair_detail,
        "summary": summary,
        "candidate_detail": candidate_detail,
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_equity_curve(detail: pd.DataFrame, ticker_a: str, ticker_b: str, out_path: str,
                      title_suffix: str = ""):
    if detail.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(detail.index, detail["equity_curve"], linewidth=1.5)
    axes[0].set_title(f"Equity Curve — {ticker_a} / {ticker_b}{title_suffix}")
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.3)
    axes[1].plot(detail.index, detail["zscore"], linewidth=1)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_ylabel("Spread z-score")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_walk_forward_equity(equity: pd.Series, fold_df: pd.DataFrame, out_path: str, title: str):
    if equity is None or len(equity) == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(equity.index, equity.values, linewidth=1.5)
    if not fold_df.empty and "test_start" in fold_df.columns:
        for _, row in fold_df.iterrows():
            if pd.notna(row.get("test_start")):
                axes[0].axvline(row["test_start"], linestyle="--", linewidth=0.7, alpha=0.5)
    axes[0].set_title(title)
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.3)

    if not fold_df.empty and "fold_sharpe" in fold_df.columns:
        vals = pd.to_numeric(fold_df["fold_sharpe"], errors="coerce").fillna(0)
        axes[1].bar(fold_df["fold"].astype(str), vals)
        axes[1].axhline(0, linewidth=0.8)
        axes[1].set_ylabel("Fold Sharpe")
        axes[1].set_xlabel("Fold")
        axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_heatmap(sweep_df: pd.DataFrame, ticker_a: str, ticker_b: str, out_path: str,
                       title: str):
    if sweep_df.empty:
        return
    pivot = sweep_df.pivot(index="exit_z", columns="entry_z", values="net_sharpe")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot.sort_index(ascending=False), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Net Sharpe"})
    ax.set_title(title)
    ax.set_xlabel("Entry z-score")
    ax.set_ylabel("Exit z-score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Leakage-aware pairs trading backtest v2")
    parser.add_argument("--data-csv", default="raw_data.csv")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--out-prefix", default="pairs_research/backtest_results/backtest_v2")

    # Pair selection -- performed on TRAIN by default.
    parser.add_argument("--pair-selection", choices=["bonferroni", "fdr_bh", "raw"], default="fdr_bh")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--min-obs", type=int, default=252)
    parser.add_argument("--pairs-csv", default=None,
                        help="Optional pair-universe whitelist. Its p-values are NOT used for selection.")
    parser.add_argument("--min-half-life", type=float, default=1.0)
    parser.add_argument("--max-half-life", type=float, default=60.0)
    parser.add_argument("--stability-windows", type=int, default=4,
                        help="Number of chronological rolling EG windows used to report train cointegration stability; 0 disables the diagnostic/filter.")
    parser.add_argument("--min-coint-stability", type=float, default=0.0)
    parser.add_argument("--max-cost-to-expected-move", type=float, default=0.50,
                        help="Reject a pair when estimated round-trip costs exceed this fraction of the expected z-score mean-reversion move.")

    # Strategy.
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--cooldown-days", type=int, default=5)
    parser.add_argument("--reentry-z", type=float, default=1.0)
    parser.add_argument("--hedge-mode", choices=["static", "rolling", "kalman"], default="rolling")
    parser.add_argument("--hedge-window", type=int, default=120)
    parser.add_argument("--kalman-delta", type=float, default=1e-5)
    parser.add_argument("--zscore-mode", choices=["static", "rolling"], default="rolling")
    parser.add_argument("--zscore-window", type=int, default=120)

    # Costs / sizing.
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--short-borrow-bps", type=float, default=0.0)
    parser.add_argument("--financing-bps", type=float, default=0.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--risk-target", type=float, default=0.10,
                        help="Target annualized volatility of a single pair's allocated capital.")
    parser.add_argument("--max-gross-leverage", type=float, default=2.0)

    # Validation.
    parser.add_argument("--entry-z-grid", default="1.5,2.0,2.5,3.0")
    parser.add_argument("--exit-z-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--wf-folds", type=int, default=5)
    parser.add_argument("--wf-initial-train-frac", type=float, default=0.5)
    parser.add_argument("--wf-val-frac", type=float, default=0.5)
    parser.add_argument("--portfolio", action="store_true",
                        help="Run the full train-selected, walk-forward multi-pair portfolio.")
    parser.add_argument("--portfolio-capital", type=float, default=100_000.0)
    parser.add_argument("--max-pair-weight", type=float, default=0.25)
    parser.add_argument("--min-validation-trades", type=int, default=2,
                        help="Minimum completed trades on validation required for a threshold combination to win.")
    parser.add_argument("--min-test-trades", type=int, default=2,
                        help="Minimum completed trades in a test fold for that fold to count as trade-sufficient; lower values are appropriate for short (~6–9 month) test folds.")
    parser.add_argument("--bootstrap-sharpe", type=int, default=0,
                        help="Optional number of Sharpe bootstrap replications; 0 disables.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip per-pair PNG plots (much faster for large runs).")
    parser.add_argument("--coint-autolag", choices=["aic", "bic", "t-stat", "none"], default="aic",
                        help="Engle-Granger autolag method. 'none' is materially faster but less adaptive.")
    parser.add_argument("--sweep", action="store_true",
                        help="Use TRAIN -> VALIDATION -> FINAL TEST: tune thresholds on validation only, then apply once to the untouched test slice.")
    args = parser.parse_args()

    global COINT_AUTOLAG
    COINT_AUTOLAG = args.coint_autolag

    prices = load_price_matrix(args.data_csv, start_date=args.start_date)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")
    pair_universe = load_pair_universe(args.pairs_csv)
    if args.pairs_csv:
        print("WARNING: --pairs-csv is used only as a pair-universe whitelist. "
              "If that whitelist itself was selected from the full sample, it can still introduce selection bias.")

    entry_grid = [float(x) for x in args.entry_z_grid.split(",") if x.strip()]
    exit_grid = [float(x) for x in args.exit_z_grid.split(",") if x.strip()]

    config = BacktestConfig(
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
        hedge_mode=args.hedge_mode,
        hedge_window=args.hedge_window,
        kalman_delta=args.kalman_delta,
        zscore_mode=args.zscore_mode,
        zscore_window=args.zscore_window,
    )

    selection_kwargs = {
        "min_obs": args.min_obs,
        "min_half_life": args.min_half_life,
        "max_half_life": args.max_half_life,
        "stability_windows": args.stability_windows,
        "min_coint_stability": args.min_coint_stability,
        "max_cost_to_expected_move": args.max_cost_to_expected_move,
        "entry_z_for_cost": args.entry_z,
        "exit_z_for_cost": args.exit_z,
        "transaction_cost_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
    }

    if args.portfolio:
        print("\n=== WALK-FORWARD PORTFOLIO ===")
        print("Pair selection and multiple-testing correction are repeated inside each TRAIN fold.")
        portfolio = walk_forward_portfolio(
            prices,
            pair_selection=args.pair_selection,
            significance=args.significance,
            top_n=args.top_n,
            pair_universe=pair_universe,
            config=config,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            n_folds=args.wf_folds,
            initial_train_frac=args.wf_initial_train_frac,
            val_frac_within_fold=args.wf_val_frac,
            portfolio_capital=args.portfolio_capital,
            max_pair_weight=args.max_pair_weight,
            selection_kwargs=selection_kwargs,
            min_validation_trades=args.min_validation_trades,
            min_test_trades=args.min_test_trades,
        )
        m = portfolio["metrics"]
        print(f"Portfolio Sharpe:       {_fmt_optional(m['sharpe'])}")
        print(f"Portfolio Sortino:      {_fmt_optional(m['sortino'])}")
        print(f"Annualized volatility:  {_fmt_optional(m['annualized_vol'] * 100)}%")
        print(f"Total compounded return:{_fmt_optional(m['total_return'] * 100)}%")
        print(f"Max drawdown:           {_fmt_optional(m['max_drawdown'] * 100)}%")
        print(f"Max DD duration:        {m['max_drawdown_days']} days")

        portfolio["selected_pairs"].to_csv(f"{args.out_prefix}_portfolio_selected_pairs.csv", index=False)
        portfolio["folds"].to_csv(f"{args.out_prefix}_portfolio_folds.csv", index=False)
        if m["equity"] is not None:
            m["equity"].to_csv(f"{args.out_prefix}_portfolio_equity.csv", header=["equity"])
            if not args.no_plots:
                plot_walk_forward_equity(
                    m["equity"], portfolio["folds"],
                    f"{args.out_prefix}_portfolio_walk_forward.png",
                    "Walk-Forward Portfolio Equity",
                )
        print(f"\nSaved: {args.out_prefix}_portfolio_selected_pairs.csv")
        print(f"Saved: {args.out_prefix}_portfolio_folds.csv")
        return

    # Initial train/test experiment: pair selection is ALWAYS based on TRAIN only.
    train_end = int(len(prices) * args.train_frac)
    if args.sweep:
        val_end = int(len(prices) * (args.train_frac + args.val_frac))
        if val_end >= len(prices):
            raise ValueError("train-frac + val-frac must leave a final held-out test slice.")
        train = prices.iloc[:train_end]
        val = prices.iloc[train_end:val_end]
        test = prices.iloc[val_end:]
    else:
        train = prices.iloc[:train_end]
        val = None
        test = prices.iloc[train_end:]

    candidates = compute_train_pair_candidates(
        train,
        pair_selection=args.pair_selection,
        significance=args.significance,
        top_n=args.top_n,
        pair_universe=pair_universe,
        **selection_kwargs,
    )

    if candidates.empty:
        print("No pairs survived train-only selection.")
        return

    print("\n=== TRAIN-ONLY PAIR SELECTION ===")
    print(f"Primary selection: {args.pair_selection.upper()} | max cost/expected move: {args.max_cost_to_expected_move:.2f} | max half-life: {args.max_half_life:g} days")
    display_cols = ["Ticker A", "Ticker B", "coint_pvalue", "pvalue_bonferroni",
                    "pvalue_fdr_bh", "half_life_days", "coint_stability",
                    "spread_adf_pvalue", "cost_to_expected_move"]
    print(candidates[[c for c in display_cols if c in candidates.columns]].to_string(index=False))

    summary_rows = []
    for pair_idx, (_, row) in enumerate(candidates.iterrows(), start=1):
        a, b = row["Ticker A"], row["Ticker B"]
        print(f"  Backtesting pair {pair_idx}/{len(candidates)}: {a}/{b}", flush=True)
        if a not in test.columns or b not in test.columns:
            continue
        pair_cfg = BacktestConfig(**vars(config))
        if args.sweep:
            # Tune ONLY on validation, then lock parameters and test ONCE on the
            # untouched final slice. The final test is never searched.
            best, sweep_df = tune_pair(
                train[[a, b]], val[[a, b]], (a, b), pair_cfg, entry_grid, exit_grid,
                min_validation_trades=args.min_validation_trades,
            )
            if not pd.isna(best["entry_z"]):
                pair_cfg.entry_z = float(best["entry_z"])
                pair_cfg.exit_z = float(best["exit_z"])
            if not sweep_df.empty:
                sweep_df.to_csv(f"{args.out_prefix}_{a}_{b}_validation_sweep.csv", index=False)
                if not args.no_plots:
                    plot_sweep_heatmap(
                        sweep_df, a, b,
                        f"{args.out_prefix}_{a}_{b}_validation_sweep.png",
                        f"Threshold sweep — validation slice — {a}/{b}",
                    )
        res = _run_backtest_on_segment(test[a], test[b], train[a], train[b], pair_cfg)

        stats = res["trade_stats"]
        sharpe_low, sharpe_high = _bootstrap_sharpe_ci(
            res["detail"].get("daily_return", pd.Series(dtype=float)),
            n_boot=args.bootstrap_sharpe,
        )
        summary_rows.append({
            "Ticker A": a,
            "Ticker B": b,
            "coint_pvalue_train": row["coint_pvalue"],
            "pvalue_bonferroni_train": row["pvalue_bonferroni"],
            "pvalue_fdr_bh_train": row["pvalue_fdr_bh"],
            "beta_train": res["beta"],
            "entry_z_used": pair_cfg.entry_z,
            "exit_z_used": pair_cfg.exit_z,
            "half_life_days": row["half_life_days"],
            "coint_stability": row["coint_stability"],
            "gross_sharpe": res["gross_sharpe"],
            "net_sharpe": res["sharpe"],
            "sortino": res["sortino"],
            "annualized_vol": res["annualized_vol"],
            "total_return": res["total_return"],
            "gross_total_return": res["gross_total_return"],
            "max_drawdown": res["max_drawdown"],
            "max_drawdown_days": res["max_drawdown_days"],
            "cost_pct_of_abs_gross": res["cost_pct_of_abs_gross"],
            "completed_trades": stats["n_completed_trades"],
            "test_trade_sufficient": bool(stats["n_completed_trades"] >= args.min_test_trades),
            "open_trades": stats["n_open_trades"],
            "win_rate": stats["win_rate"],
            "avg_win": stats["avg_win"],
            "avg_loss": stats["avg_loss"],
            "profit_factor": stats["profit_factor"],
            "expectancy": stats["expectancy"],
            "avg_holding_days": stats["avg_holding_days"],
            "max_trade_loss": stats["max_trade_loss"],
            "n_entries": res["n_entries"],
            "n_exits": res["n_exits"],
            "transaction_cost": res["total_transaction_cost"],
            "carry_cost": res["total_carry_cost"],
            "capital_exhausted": res["capital_exhausted"],
            "sharpe_bootstrap_low": sharpe_low,
            "sharpe_bootstrap_high": sharpe_high,
        })

        if not args.no_plots:
            plot_equity_curve(
                res["detail"], a, b,
                f"{args.out_prefix}_{a}_{b}_equity_curve.png",
                title_suffix=" (train-selected, held-out test)",
            )
        res["detail"].to_csv(f"{args.out_prefix}_{a}_{b}_detail.csv")
        res["trade_log"].to_csv(f"{args.out_prefix}_{a}_{b}_trades.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("net_sharpe", ascending=False)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print("\n=== HELD-OUT TEST SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out_prefix}_summary.csv")

    if args.walk_forward:
        print("\n=== WALK-FORWARD WITH PAIR RESELECTION ===")
        print("Every fold re-runs cointegration tests, multiple-testing correction, threshold tuning, and final testing.")
        wf = walk_forward_reselect_pairs(
            prices,
            pair_selection=args.pair_selection,
            significance=args.significance,
            top_n=args.top_n,
            pair_universe=pair_universe,
            config=config,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            n_folds=args.wf_folds,
            initial_train_frac=args.wf_initial_train_frac,
            val_frac_within_fold=args.wf_val_frac,
            selection_kwargs=selection_kwargs,
        )
        wf["folds"].to_csv(f"{args.out_prefix}_walk_forward_folds.csv", index=False)
        wf["pair_detail"].to_csv(f"{args.out_prefix}_walk_forward_pair_detail.csv", index=False)
        wf["summary"].to_csv(f"{args.out_prefix}_walk_forward_pair_summary.csv", index=False)
        wf["candidate_detail"].to_csv(f"{args.out_prefix}_walk_forward_selected_candidates.csv", index=False)
        print(wf["folds"].to_string(index=False))
        print("\nPair-level walk-forward summary:")
        print(wf["summary"].to_string(index=False))
        print(f"\nSaved: {args.out_prefix}_walk_forward_folds.csv")
        print(f"Saved: {args.out_prefix}_walk_forward_pair_detail.csv")
        print(f"Saved: {args.out_prefix}_walk_forward_pair_summary.csv")



if __name__ == "__main__":
    main()
