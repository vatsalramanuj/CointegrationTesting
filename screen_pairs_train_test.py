#!/usr/bin/env python3
"""
Pairs cointegration / mean-reversion screener  (v2: walk-forward + Johansen)

Input : yfinance CSV with the 3-row header  (Ticker / Price / Date)
Output: one row per pair (CSV).

Usage:
    python screen_pairs_train_test.py raw_data.csv -o pairs_results.csv --train-frac 0.6
    python screen_pairs_train_test.py raw_data.csv --wf-step 21 --wf-lookback 504
    python screen_pairs_train_test.py raw_data.csv --wf-step 0 --no-johansen     # = v1 behaviour

Three evaluations of each pair
------------------------------
1. TRAIN      Engle-Granger (EG) + Johansen cointegration, ADF, half-life, Hurst.
2. TEST-FROZEN  Hedge ratio / spread mean / std frozen from the train window
                (strictest out-of-sample check: do the train parameters still work?).
3. TEST-WALK-FORWARD  Through the test period the hedge ratio is re-fitted every
                --wf-step days using only the preceding --wf-lookback days
                (what you could actually trade). Stationarity of the resulting
                out-of-sample z-score series, hedge-ratio stability and a backtest
                are reported with the prefix wf_.
Plus rolling EG re-tests inside the test window (persistence of cointegration).
"""
import argparse
import warnings
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_close(path, max_missing=0.05):
    """Read the yfinance CSV (Ticker row, Price row, Date row) -> Close DataFrame."""
    df = pd.read_csv(path, header=[0, 1], index_col=0, skiprows=[2])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    close = df.xs("Close", axis=1, level=1).apply(pd.to_numeric, errors="coerce")
    close = close.loc[:, close.isna().mean() <= max_missing]
    close = close.ffill(limit=3).dropna(how="any")
    close = close.loc[:, (close > 0).all()]
    return close


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def ols_hedge(y, x):
    b, a = np.polyfit(x, y, 1)
    return a, b


def half_life(s):
    """Half-life from AR(1): ds = c + k*s_{t-1}; phi = 1 + k."""
    s = np.asarray(s)
    k = np.polyfit(s[:-1], np.diff(s), 1)[0]
    phi = 1.0 + k
    return -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan


def hurst(s, max_lag=50):
    s = np.asarray(s)
    lags = np.arange(2, max(3, min(max_lag, len(s) // 4)))
    tau = [np.std(s[l:] - s[:-l]) for l in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def adf_p(s):
    try:
        return adfuller(s, autolag="AIC")[1]
    except Exception:
        return np.nan


def zero_crossings(z):
    z = np.asarray(z)
    return int(np.sum(np.signbit(z[1:]) != np.signbit(z[:-1])))


def johansen(ly, lx):
    """Johansen test (rank 0 vs >=1), constant in cointegrating relation, 1 lagged diff.
    Returns trace, trace_crit95, maxeig, maxeig_crit95, hedge_ratio (from 1st eigenvector)."""
    try:
        r = coint_johansen(np.column_stack([ly, lx]), det_order=0, k_ar_diff=1)
        v = r.evec[:, 0]
        beta = -v[1] / v[0] if v[0] != 0 else np.nan
        return r.lr1[0], r.cvt[0, 1], r.lr2[0], r.cvm[0, 1], beta
    except Exception:
        return (np.nan,) * 5


def backtest(z, ds, beta, entry=2.0, exit_=0.5, stop=4.0):
    """
    z    : z-score series (length n)
    ds   : daily spread changes aligned with z[1:]  (length n-1), using the hedge
           ratio that was live on that day
    beta : scalar or length n-1 array (live hedge ratio) for gross-notional scaling
    Signal at close t-1 -> position held over day t. Log-spread terms, no costs.
    Returns (n_trades, total_return, sharpe, win_rate).
    """
    n = len(z)
    pos = np.zeros(n)
    cur = 0
    for t in range(n):
        if cur == 0:
            if z[t] > entry:
                cur = -1
            elif z[t] < -entry:
                cur = 1
        elif abs(z[t]) < exit_ or abs(z[t]) > stop:
            cur = 0
        pos[t] = cur
    pos_lag = np.r_[0, pos[:-1]][1:]                      # position held on day t (t=1..n-1)
    pnl = pos_lag * np.asarray(ds) / (1 + np.abs(beta))

    trades, acc, is_open = [], 0.0, False
    for p, d in zip(pos_lag, pnl):
        if p != 0:
            acc += d
            is_open = True
        elif is_open:
            trades.append(acc)
            acc, is_open = 0.0, False
    if is_open:
        trades.append(acc)

    sd = pnl.std()
    sharpe = pnl.mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    win = float(np.mean(np.array(trades) > 0)) if trades else np.nan
    return len(trades), float(pnl.sum()), sharpe, win


def rolling_coint(ly, lx, win, step):
    ps = []
    for i in range(0, len(ly) - win + 1, step):
        try:
            ps.append(coint(ly[i:i + win], lx[i:i + win], autolag="aic")[1])
        except Exception:
            ps.append(np.nan)
    return np.array(ps)


def walk_forward(ly, lx, n_train, step, lookback):
    """
    Re-fit (alpha, beta) every `step` days on the trailing `lookback` days and apply
    to the next `step` days. Returns out-of-sample z (len n-n_train), daily spread
    changes ds (len n-n_train), live betas per day (len n-n_train), block betas.
    """
    n = len(ly)
    z, ds, bday, blocks = [], [], [], []
    start = n_train
    while start < n:
        end = min(start + step, n)
        lo = max(0, start - lookback)
        a, b = ols_hedge(ly[lo:start], lx[lo:start])
        s_fit = ly[lo:start] - (a + b * lx[lo:start])
        mu, sd = s_fit.mean(), s_fit.std(ddof=1)
        s_blk = ly[start:end] - (a + b * lx[start:end])
        z.append((s_blk - mu) / sd)
        ds.append((ly[start:end] - ly[start - 1:end - 1]) - b * (lx[start:end] - lx[start - 1:end - 1]))
        bday.append(np.full(end - start, b))
        blocks.append(b)
        start = end
    return np.concatenate(z), np.concatenate(ds), np.concatenate(bday), np.array(blocks)


# ----------------------------------------------------------------------------
# Per-pair analysis
# ----------------------------------------------------------------------------
G = {}


def _init(logp, n_train, args):
    G["logp"], G["n_train"], G["args"] = logp, n_train, args


def analyse_pair(pair):
    a_name, b_name = pair
    args, n_train, logp = G["args"], G["n_train"], G["logp"]
    ly_all, lx_all = logp[a_name].values, logp[b_name].values
    ly_tr, lx_tr = ly_all[:n_train], lx_all[:n_train]
    ly_te, lx_te = ly_all[n_train:], lx_all[n_train:]

    ret_corr = np.corrcoef(np.diff(ly_tr), np.diff(lx_tr))[0, 1]
    if ret_corr < args.min_corr:
        return None
    row = {"y": a_name, "x": b_name, "train_return_corr": ret_corr}

    # ---------------- TRAIN ----------------
    try:
        eg_p = coint(ly_tr, lx_tr, autolag="aic")[1]
    except Exception:
        return None
    alpha, beta = ols_hedge(ly_tr, lx_tr)
    s_tr = ly_tr - (alpha + beta * lx_tr)
    mu, sd = s_tr.mean(), s_tr.std(ddof=1)
    row.update(
        hedge_ratio=beta, intercept=alpha,
        train_eg_pvalue=eg_p, train_adf_pvalue=adf_p(s_tr),
        train_half_life=half_life(s_tr), train_hurst=hurst(s_tr),
        train_spread_mean=mu, train_spread_std=sd,
    )

    if args.johansen:
        tr, tc, me, mc, jb = johansen(ly_tr, lx_tr)
        row.update(
            train_johansen_trace=tr, train_johansen_trace_crit95=tc,
            train_johansen_maxeig=me, train_johansen_maxeig_crit95=mc,
            train_johansen_cointegrated=bool(tr > tc and me > mc) if np.isfinite(tr) else False,
            johansen_hedge_ratio=jb,
        )
        # fresh Johansen fit on the test window (re-estimated, so not strictly out-of-sample)
        tr2, tc2, me2, mc2, _ = johansen(ly_te, lx_te)
        row.update(
            test_johansen_trace_ratio=tr2 / tc2 if np.isfinite(tr2) and tc2 else np.nan,
            test_johansen_cointegrated=bool(tr2 > tc2 and me2 > mc2) if np.isfinite(tr2) else False,
        )

    # ---------------- TEST: frozen train parameters ----------------
    s_te = ly_te - (alpha + beta * lx_te)
    z_te = (s_te - mu) / sd
    n_tr, ret, sharpe, win = backtest(z_te, np.diff(s_te), beta,
                                      args.entry_z, args.exit_z, args.stop_z)
    years = len(s_te) / 252
    row.update(
        test_adf_pvalue=adf_p(s_te),
        test_half_life=half_life(s_te),
        test_hurst=hurst(s_te),
        test_mean_shift_in_train_sigma=(s_te.mean() - mu) / sd,
        test_std_ratio=s_te.std(ddof=1) / sd,
        test_max_abs_z=np.abs(z_te).max(),
        test_pct_within_2sigma=np.mean(np.abs(z_te) < 2),
        test_zero_crossings_per_year=zero_crossings(z_te) / years,
        bt_trades=n_tr, bt_return=ret, bt_sharpe=sharpe, bt_win_rate=win,
    )

    # ---------------- TEST: walk-forward re-fit ----------------
    if args.wf_step > 0 and len(ly_te) > 2:
        lookback = args.wf_lookback if args.wf_lookback > 0 else n_train
        z_wf, ds_wf, b_day, b_blocks = walk_forward(ly_all, lx_all, n_train, args.wf_step, lookback)
        n_tr, ret, sharpe, win = backtest(z_wf, ds_wf[1:], b_day[1:],
                                          args.entry_z, args.exit_z, args.stop_z)
        row.update(
            wf_adf_pvalue=adf_p(z_wf),
            wf_half_life=half_life(z_wf),
            wf_pct_within_2sigma=np.mean(np.abs(z_wf) < 2),
            wf_max_abs_z=np.abs(z_wf).max(),
            wf_zero_crossings_per_year=zero_crossings(z_wf) / years,
            wf_beta_mean=b_blocks.mean(),
            wf_beta_std=b_blocks.std(),
            wf_beta_drift_vs_train=b_blocks.mean() - beta,
            wf_beta_sign_flips=int(np.sum(np.sign(b_blocks) != np.sign(beta))),
            wf_bt_trades=n_tr, wf_bt_return=ret, wf_bt_sharpe=sharpe, wf_bt_win_rate=win,
        )

    # ---------------- Rolling persistence inside test window ----------------
    if eg_p <= args.screen_p and len(ly_te) >= args.roll_window + args.roll_step:
        ps = rolling_coint(ly_te, lx_te, args.roll_window, args.roll_step)
        row.update(roll_n_windows=int(np.isfinite(ps).sum()),
                   roll_frac_cointegrated=float(np.nanmean(ps < args.alpha)),
                   roll_mean_pvalue=float(np.nanmean(ps)))
    return row


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="pairs_results.csv")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--train-end", default=None, help="e.g. 2023-12-31 (overrides --train-frac)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--screen-p", type=float, default=0.10,
                    help="run rolling test only for pairs with train EG p <= this")
    ap.add_argument("--min-corr", type=float, default=-1.0,
                    help="skip pairs with train return correlation below this (speed-up)")
    ap.add_argument("--wf-step", type=int, default=63,
                    help="walk-forward: re-fit hedge ratio every N days (0 disables)")
    ap.add_argument("--wf-lookback", type=int, default=0,
                    help="walk-forward fit window in days (0 = same length as train)")
    ap.add_argument("--no-johansen", dest="johansen", action="store_false")
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--roll-window", type=int, default=126)
    ap.add_argument("--roll-step", type=int, default=21)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()

    close = load_close(args.csv)
    logp = np.log(close)
    n_train = (int((logp.index <= pd.Timestamp(args.train_end)).sum())
               if args.train_end else int(len(logp) * args.train_frac))
    print(f"{close.shape[1]} tickers, {len(close)} days | "
          f"train: {logp.index[0].date()} -> {logp.index[n_train-1].date()} ({n_train}d) | "
          f"test: {logp.index[n_train].date()} -> {logp.index[-1].date()} ({len(logp)-n_train}d)")

    pairs = list(combinations(close.columns, 2))
    print(f"Testing {len(pairs)} pairs ...")
    with Pool(args.jobs, initializer=_init, initargs=(logp, n_train, args)) as pool:
        rows = [r for r in pool.imap_unordered(analyse_pair, pairs, chunksize=50) if r]
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No results.")

    # multiple-testing correction on the train EG p-values
    res["train_eg_pvalue_fdr"] = multipletests(res["train_eg_pvalue"].fillna(1), method="fdr_bh")[1]

    a = args.alpha
    res["cointegrated_train"] = res["train_eg_pvalue"] < a
    res["cointegrated_train_fdr"] = res["train_eg_pvalue_fdr"] < a
    res["stationary_test"] = res["test_adf_pvalue"] < a
    res["persisted"] = res["cointegrated_train"] & res["stationary_test"]
    res["persisted_fdr"] = res["cointegrated_train_fdr"] & res["stationary_test"]
    if args.wf_step > 0:
        res["stationary_wf"] = res["wf_adf_pvalue"] < a
        res["persisted_wf_fdr"] = res["cointegrated_train_fdr"] & res["stationary_wf"]
    if args.johansen:
        res["eg_and_johansen_train"] = res["cointegrated_train_fdr"] & res["train_johansen_cointegrated"]
    # strictest: every available check agrees
    strict = res["persisted_fdr"].copy()
    if args.wf_step > 0:
        strict &= res["persisted_wf_fdr"]
    if args.johansen:
        strict &= res["eg_and_johansen_train"]
    res["persisted_strict"] = strict

    rank = res["train_eg_pvalue"].rank(pct=True) + res["test_adf_pvalue"].rank(pct=True)
    if args.wf_step > 0:
        rank += res["wf_adf_pvalue"].rank(pct=True)
    rank += 1 - res["roll_frac_cointegrated"].fillna(0).rank(pct=True)
    res["rank_score"] = rank
    res = res.sort_values(["persisted_strict", "persisted_fdr", "persisted", "rank_score"],
                          ascending=[False, False, False, True])

    res.to_csv(args.out, index=False)
    print(f"\nSaved {len(res)} pairs -> {args.out}")
    print(f"EG cointegrated in train (raw p<{a})   : {int(res.cointegrated_train.sum())}")
    print(f"  ... after FDR correction             : {int(res.cointegrated_train_fdr.sum())}")
    if args.johansen:
        print(f"  ... EG(FDR) AND Johansen             : {int(res.eg_and_johansen_train.sum())}")
    print(f"  ... stationary in frozen test        : {int(res.persisted_fdr.sum())}")
    if args.wf_step > 0:
        print(f"  ... stationary in walk-forward test  : {int(res.persisted_wf_fdr.sum())}")
    print(f"All checks pass (persisted_strict)     : {int(res.persisted_strict.sum())}")
    cols = [c for c in ["y", "x", "hedge_ratio", "train_eg_pvalue_fdr", "train_half_life",
                        "test_adf_pvalue", "wf_adf_pvalue", "wf_beta_std",
                        "roll_frac_cointegrated", "wf_bt_sharpe"] if c in res.columns]
    print("\nTop 10:")
    print(res[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
