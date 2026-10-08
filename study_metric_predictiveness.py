#!/usr/bin/env python3
"""
Panel study: do cointegration / mean-reversion metrics predict forward trading PnL?

For EVERY tested pair at EVERY screen date (no pool, no ranking, no gate - weakly cointegrated pairs
included) the script
    1. computes the screening metrics on the trailing --screen-window days, and
    2. simulates the pair with the full cost model over the next --step-days (default 126 ~ 6 months),
       with the hedge ratio re-fitted on trailing data only (no look-ahead).
Then, per metric:
    * cross-sectional rank IC per period (Spearman, metric vs forward net return) -> mean IC, t-stat across
      periods, % of periods positive, Holm-adjusted p-value across all metrics
    * quintile sorts per period (Q5 = highest metric) -> average forward net/gross return per quintile and the
      Q5 - Q1 spread with its t-stat across periods
    * regressions of forward return on the metric (z-scored within period, period fixed effects, two-way
      clustered standard errors by period and by pair), univariate and joint
    * the existence gate itself: do pairs passing FDR<0.10 / <0.05 earn more than the other pairs of the SAME
      group in the SAME period?
An optional --holdout-start reserves the later periods: they are only reported, never used to pick anything.

Usage:
    python study_metric_predictiveness.py etf_prices.csv --groups-csv groups_etf.csv --holdout-start 2023-01-01
    python study_metric_predictiveness.py raw_data.csv --groups-csv groups_107.csv --screen-window 756
"""
import argparse
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pairs_common import load_panels, pair_indices, screen_universe, simulate_pair

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Parallel forward simulation of one pair
# ----------------------------------------------------------------------------
_S = {}


def _init_sim(D, args):
    _S["D"], _S["args"] = D, args


def _sim_pair(task):
    ty, tx, r, end = task
    d, trs = simulate_pair(ty, tx, _S["D"], r, end, _S["args"])
    out = dict(y=ty, x=tx, next_net_pnl=d["net"].sum(), next_gross_pnl=d["gross"].sum(),
               next_costs=d["costs"].sum(), next_trades=len(trs), next_days_in_pos=int(d["in_pos"].sum()))
    for k in ("target", "stop", "time", "end_of_window"):         # how each trade ended, and what it earned
        sel = [t for t in trs if t["exit_reason"] == k]
        out[f"n_{k}"] = len(sel)
        out[f"pnl_{k}"] = float(sum(t["net"] for t in sel))
        out[f"win_{k}"] = int(sum(t["net"] > 0 for t in sel))
        out[f"days_{k}"] = int(sum(t["days_held"] for t in sel))
    return out


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
METRICS = [  # (column, description) - all signed so that HIGHER = expected better
    ("coint_strength", "EG strength (-t-stat)"),
    ("neglog10_p", "-log10 EG p-value"),
    ("fast_reversion", "-log(half-life)  [faster reversion]"),
    ("crossings_per_year", "mean crossings per year"),
    ("entries_per_year", "entries beyond entry-z per year"),
    ("gross_bps_per_trade", "gross bps per round trip"),
    ("cost_bps_per_trade", "est. cost bps per round trip"),
    ("net_bps_per_trade", "net bps per round trip"),
    ("edge_ratio", "gross / cost edge ratio"),
    ("net_edge_bps_per_year", "est. net edge bps per year"),
    ("stability", "sub-window stability"),
    ("neg_beta_cv", "-hedge-ratio dispersion"),
    ("ret_corr", "daily return correlation"),
    ("johansen_trace_ratio", "Johansen trace / crit95"),
    ("kpss_pvalue", "KPSS p-value (higher = more stationary)"),
]


def t_summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3 or x.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan, T
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(T)
    return m, m / se, 2 * stats.t.sf(abs(m / se), T - 1), T


def period_ics(P, name, outcome, min_n=8):
    out = []
    for _, g in P.groupby("period"):
        gg = g[[name, outcome]].dropna()
        if len(gg) >= min_n and gg[name].nunique() > 1 and gg[outcome].nunique() > 1:
            out.append(gg[name].rank().corr(gg[outcome].rank()))
    return np.array(out)


def quintile_matrix(P, name, outcome, min_n=10):
    rows = []
    for _, g in P.groupby("period"):
        gg = g[[name, outcome]].dropna()
        if len(gg) < min_n or gg[name].nunique() < 2:
            continue
        rk = gg[name].rank(method="average")
        b = np.minimum(((rk - 1) * 5 / len(gg)).astype(int), 4)           # ties share a bucket
        rows.append(gg.groupby(b)[outcome].mean().reindex(range(5)).values)
    return np.array(rows)


def clustered_ols(y, X, period_codes, pair_codes):
    model = sm.OLS(y, X)
    try:                                                                  # two-way: period and pair
        return model.fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([period_codes, pair_codes])})
    except Exception:
        return model.fit(cov_type="cluster", cov_kwds={"groups": period_codes})


def zscore_within(P, cols):
    Z = P[cols].copy()
    g = P["period"]
    for c in cols:
        mu, sd = Z[c].groupby(g).transform("mean"), Z[c].groupby(g).transform("std", ddof=0)
        Z[c] = (Z[c] - mu) / sd.where(sd > 0)
    return Z


def regressions(P, metrics, outcome, joint):
    Z = zscore_within(P, metrics)
    y_all = P[outcome] - P[outcome].groupby(P["period"]).transform("mean")
    rows = []
    for m in metrics:
        d = pd.concat([Z[m].rename("z"), y_all.rename("y"), P["period"], P["pair"]], axis=1).dropna()
        if len(d) < 30 or d["period"].nunique() < 4:
            continue
        r = clustered_ols(d["y"].values, d[["z"]].values, d["period"].values, pd.factorize(d["pair"])[0])
        rows.append(dict(metric=m, spec="univariate", coef_pct_per_1sd=r.params[0], t=r.tvalues[0],
                         p=r.pvalues[0], n=int(r.nobs)))
    jm = [m for m in joint if m in Z.columns and Z[m].notna().sum() > 30]
    if len(jm) >= 2:
        d = pd.concat([Z[jm], y_all.rename("y"), P["period"], P["pair"]], axis=1).dropna()
        if len(d) > 30:
            r = clustered_ols(d["y"].values, d[jm].values, d["period"].values, pd.factorize(d["pair"])[0])
            for i, m in enumerate(jm):
                rows.append(dict(metric=m, spec="joint", coef_pct_per_1sd=r.params[i], t=r.tvalues[i],
                                 p=r.pvalues[i], n=int(r.nobs)))
    return pd.DataFrame(rows)


def gate_test(P, col, outcome):
    """Within-(period, group) difference in forward return between pairs passing the gate and the others."""
    d = P[[col, outcome, "period", "pair", "group"]].dropna().copy()
    key = d["period"].astype(str) + "|" + d["group"].astype(str)
    d = d[d.groupby(key)[col].transform("std") > 0]
    if d.empty:
        return None
    key = d["period"].astype(str) + "|" + d["group"].astype(str)
    x = d[col] - d.groupby(key)[col].transform("mean")
    y = d[outcome] - d.groupby(key)[outcome].transform("mean")
    r = clustered_ols(y.values, x.values.reshape(-1, 1), d["period"].values, pd.factorize(d["pair"])[0])
    return dict(gate=col, outcome=outcome, within_group_diff_pct=r.params[0], t=r.tvalues[0], p=r.pvalues[0],
                n_pass=int((d[col] == 1).sum()), n_fail=int((d[col] == 0).sum()),
                mean_pass=d.loc[d[col] == 1, outcome].mean(), mean_fail=d.loc[d[col] == 0, outcome].mean())


EXITS = ["target", "stop", "time", "end_of_window"]


def assign_quintile(P, name, min_n=10):
    """Within-period quintile (0..4) of a metric for every row; NaN where it can't be formed."""
    q = pd.Series(np.nan, index=P.index)
    for _, g in P.groupby("period"):
        v = g[name].dropna()
        if len(v) < min_n or v.nunique() < 2:
            continue
        rk = v.rank(method="average")
        q.loc[v.index] = np.minimum(((rk - 1) * 5 / len(v)).astype(int), 4)
    return q


def conditional_tables(P, metrics):
    """Same IC / quintile analysis, but only on pair-periods that actually traded (>=1 trade)."""
    T = P[P["next_trades"] > 0]
    rows = []
    for m in metrics:
        for tag, outc in (("net", "net_ret_pct"), ("gross", "gross_ret_pct")):
            ic, ic_t, _, _ = t_summary(period_ics(T, m, outc, min_n=8))
            Q = quintile_matrix(T, m, outc, min_n=10)
            if len(Q) >= 3:
                mean, t, p, _ = t_summary(Q[:, 4] - Q[:, 0])
                rows.append(dict(metric=m, outcome=tag, n_traded_pairperiods=len(T), n_periods=len(Q), ic=ic, ic_t=ic_t,
                                 Q1=np.nanmean(Q[:, 0]), Q2=np.nanmean(Q[:, 1]), Q3=np.nanmean(Q[:, 2]),
                                 Q4=np.nanmean(Q[:, 3]), Q5=np.nanmean(Q[:, 4]), Q5_minus_Q1=mean, t=t, p=p))
    return pd.DataFrame(rows)


def exit_tables(P, metrics):
    """By metric quintile: how often pairs trade, how trades end, and where the PnL comes from."""
    rows = []
    for m in metrics:
        d = P.assign(q=assign_quintile(P, m)).dropna(subset=["q"])
        for k, g in d.groupby("q"):
            nt = g["next_trades"].sum()
            row = dict(metric=m, quintile=int(k) + 1, n_pair_periods=len(g), traded_share=g["traded"].mean(),
                       trades_per_pair=g["next_trades"].mean(),
                       win_rate=g["n_wins"].sum() / nt if nt else np.nan,
                       avg_days_held=g["days_held_total"].sum() / nt if nt else np.nan,
                       avg_net_per_trade_usd=g["next_net_pnl"].sum() / nt if nt else np.nan)
            for e in EXITS:
                row[f"{e}_share"] = g[f"n_{e}"].sum() / nt if nt else np.nan
            for e in EXITS:
                row[f"ret_{e}_pct"] = g[f"ret_{e}_pct"].mean()          # contribution to net_ret_pct
            row["net_ret_pct"] = g["net_ret_pct"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def overall_exit_summary(P):
    tot = sum(P[f"n_{e}"].sum() for e in EXITS)
    rows = []
    for e in EXITS:
        n, pnl = P[f"n_{e}"].sum(), P[f"pnl_{e}"].sum()
        rows.append(dict(exit_reason=e, n_trades=int(n), share_of_trades=n / tot if tot else np.nan,
                         win_rate=P[f"win_{e}"].sum() / n if n else np.nan,
                         avg_days_held=P[f"days_{e}"].sum() / n if n else np.nan,
                         avg_net_per_trade_usd=pnl / n if n else np.nan, total_net_pnl_usd=pnl))
    return pd.DataFrame(rows)


def analyse(P, metrics, joint, label):
    """All tables for one sample (discovery or holdout)."""
    ic_rows, q_rows = [], []
    for m in metrics:
        row = dict(metric=m)
        for tag, outc in (("net", "net_ret_pct"), ("gross", "gross_ret_pct")):
            mean, t, p, T = t_summary(period_ics(P, m, outc))
            row.update({f"ic_{tag}": mean, f"t_{tag}": t, f"p_{tag}": p})
            if tag == "net":
                row["n_periods"] = T
                row["pct_periods_pos"] = float(np.mean(period_ics(P, m, outc) > 0)) if T else np.nan
        ic_rows.append(row)
        for tag, outc in (("net", "net_ret_pct"), ("gross", "gross_ret_pct")):
            Q = quintile_matrix(P, m, outc)
            if len(Q) >= 3:
                spread = Q[:, 4] - Q[:, 0]
                mean, t, p, T = t_summary(spread)
                q_rows.append(dict(metric=m, outcome=tag, Q1=np.nanmean(Q[:, 0]), Q2=np.nanmean(Q[:, 1]),
                                   Q3=np.nanmean(Q[:, 2]), Q4=np.nanmean(Q[:, 3]), Q5=np.nanmean(Q[:, 4]),
                                   Q5_minus_Q1=mean, t=t, p=p))
    ic = pd.DataFrame(ic_rows)
    ok = ic["p_net"].notna()
    if ok.sum() > 1:
        ic.loc[ok, "p_net_holm"] = multipletests(ic.loc[ok, "p_net"], method="holm")[1]
    reg_net = regressions(P, metrics, "net_ret_pct", joint)
    reg_gross = regressions(P, metrics, "gross_ret_pct", joint)
    reg_net.insert(0, "outcome", "net")
    reg_gross.insert(0, "outcome", "gross")
    gates = [g for g in (gate_test(P, c, o) for c in ("fdr10", "fdr05") for o in ("net_ret_pct", "gross_ret_pct")) if g]
    return dict(ic=ic, quint=pd.DataFrame(q_rows), reg=pd.concat([reg_net, reg_gross]), gate=pd.DataFrame(gates),
                cond=conditional_tables(P, metrics), exit_q=exit_tables(P, metrics), exit_all=overall_exit_summary(P))


def show(res, label, detail=None):
    pd.set_option("display.width", 200)
    print(f"\n########## {label} ##########")
    ic = res["ic"].sort_values("t_net", key=lambda s: -s.abs())
    print("\n-- Cross-sectional rank IC with forward 6-month return (mean over periods; t across periods) --")
    print(ic[["metric", "ic_net", "t_net", "p_net", "p_net_holm", "pct_periods_pos", "ic_gross", "t_gross"]]
          .round(3).to_string(index=False))
    q = res["quint"]
    if len(q):
        print("\n-- Quintile sorts: mean forward return, % of capital per 6 months (Q5 = highest metric) --")
        print(q.sort_values(["outcome", "t"], key=lambda s: -s.abs() if s.name == "t" else s)
               .round(3).to_string(index=False))
    if len(res["reg"]):
        print("\n-- Regressions: coefficient = % return per +1 sd of the metric (within period), two-way clustered --")
        print(res["reg"].round(3).to_string(index=False))
    if len(res["gate"]):
        print("\n-- Existence gate: passing pairs vs failing pairs of the SAME group and period --")
        print(res["gate"].round(3).to_string(index=False))
    if len(res["exit_all"]):
        print("\n-- How trades end (all pairs, all periods; PnL in $ per trade, net of costs) --")
        print(res["exit_all"].round(3).to_string(index=False))
    c = res["cond"]
    if len(c):
        cn = c[c["outcome"] == "net"].sort_values("t", key=lambda s: -s.abs())
        print("\n-- CONDITIONAL ON TRADING (pair-periods with >=1 trade): net return by metric quintile, % of capital --")
        print(cn[["metric", "n_traded_pairperiods", "n_periods", "ic", "ic_t", "Q1", "Q2", "Q3", "Q4", "Q5",
                  "Q5_minus_Q1", "t", "p"]].round(3).to_string(index=False))
    eq = res["exit_q"]
    if len(eq) and detail:
        cols = ["metric", "quintile", "n_pair_periods", "traded_share", "trades_per_pair", "win_rate", "avg_days_held",
                "stop_share", "target_share", "time_share", "end_of_window_share", "avg_net_per_trade_usd"]
        print("\n-- Trading behaviour by quintile (Q5 = highest metric): activity and how trades end --")
        print(eq[eq["metric"].isin(detail)][cols].round(3).to_string(index=False))
        cols2 = ["metric", "quintile", "ret_target_pct", "ret_stop_pct", "ret_time_pct", "ret_end_of_window_pct", "net_ret_pct"]
        print("\n-- Where the return comes from, by quintile: contribution of each exit type, % of capital per 6 months --")
        print(eq[eq["metric"].isin(detail)][cols2].round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prices_csv")
    ap.add_argument("--outdir", default="panel_output")
    ap.add_argument("--groups-csv", default=None, help="ticker,group csv; only same-group pairs are tested")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--screen-window", type=int, default=504)
    ap.add_argument("--step-days", type=int, default=126, help="forward horizon = spacing between screens")
    ap.add_argument("--holdout-start", default=None, help="periods starting on/after this date are reported separately")
    ap.add_argument("--min-last-period", type=int, default=20)
    ap.add_argument("--johansen", action="store_true", help="also compute Johansen as a feature")
    ap.add_argument("--kpss", action="store_true", help="also compute KPSS as a feature")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--detail-metrics", default="net_edge_bps_per_year,gross_bps_per_trade,neg_beta_cv,coint_strength,fast_reversion",
                    help="comma-separated metrics shown in the by-quintile trading-behaviour tables (all go to csv)")
    # screening metric parameters
    ap.add_argument("--min-corr", type=float, default=-1.0, help="-1 = keep every pair")
    ap.add_argument("--min-hl", type=float, default=2.0)
    ap.add_argument("--max-hl", type=float, default=60.0)
    ap.add_argument("--n-subwindows", type=int, default=3)
    # strategy
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--wf-step", type=int, default=63)
    ap.add_argument("--wf-lookback", type=int, default=0)
    ap.add_argument("--exec", choices=["open", "close"], default="open")
    # sizing / costs
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--gross-leverage", type=float, default=1.0)
    ap.add_argument("--comm-ps", type=float, default=0.0035)
    ap.add_argument("--min-comm", type=float, default=0.35)
    ap.add_argument("--half-spread-bps", type=float, default=2.0)
    ap.add_argument("--impact-coef", type=float, default=1.0)
    ap.add_argument("--sell-fee-bps", type=float, default=0.3)
    ap.add_argument("--borrow-bps", type=float, default=50.0)
    ap.add_argument("--financing-bps", type=float, default=0.0)
    args = ap.parse_args()
    args.gate_johansen, args.gate_kpss = args.johansen, args.kpss
    args.train_len = args.screen_window
    os.makedirs(args.outdir, exist_ok=True)

    D = load_panels(args.prices_csv)
    close = D["close"]
    dates, n = close.index, len(close)
    logp = np.log(close)
    groups = None
    if args.groups_csv:
        g = pd.read_csv(args.groups_csv)
        groups = dict(zip(g["ticker"], g["group"]))
    n_pairs = len(pair_indices(list(close.columns), groups)[0])
    start = int(dates.searchsorted(pd.Timestamp(args.start_date))) if args.start_date else args.screen_window
    if start < args.screen_window or start >= n - args.min_last_period:
        raise SystemExit("Not enough history: need --screen-window days before the first forward window.")
    print(f"{close.shape[1]} tickers, {n_pairs} pairs per screen{' (within groups)' if groups else ''}, {n} days | "
          f"first forward window {dates[start].date()} | screen window {args.screen_window}d, horizon {args.step_days}d")

    frames = []
    with Pool(args.jobs, initializer=_init_sim, initargs=(D, args)) as sim_pool:
        r, k = start, 0
        while r < n - args.min_last_period:
            end = min(r + args.step_days, n)
            scr = screen_universe(logp, r - args.screen_window, r, args, groups)
            if not scr.empty:
                res = sim_pool.map(_sim_pair, [(a, b, r, end) for a, b in zip(scr["y"], scr["x"])], chunksize=20)
                df = scr.merge(pd.DataFrame(res), on=["y", "x"])
                df.insert(0, "period", k)
                df.insert(1, "window_start", dates[r])
                df.insert(2, "window_end", dates[end - 1])
                frames.append(df)
                print(f"Period {k:>2}: screen {dates[r - args.screen_window].date()} -> {dates[r - 1].date()} | forward "
                      f"{dates[r].date()} -> {dates[end - 1].date()} | {len(df)} pairs, "
                      f"{int((df['eg_pvalue_fdr'] < 0.10).sum())} pass FDR<0.10, "
                      f"mean net {100 * df['next_net_pnl'].mean() / args.capital:+.2f}%")
            r, k = end, k + 1
    P = pd.concat(frames, ignore_index=True)
    P["pair"] = P["y"] + "/" + P["x"]
    P["net_ret_pct"] = 100 * P["next_net_pnl"] / args.capital
    P["gross_ret_pct"] = 100 * P["next_gross_pnl"] / args.capital
    P["coint_strength"] = -P["eg_tstat"]
    P["neglog10_p"] = -np.log10(P["eg_pvalue"].clip(lower=1e-12))
    P["fast_reversion"] = -np.log(P["half_life"])
    P["neg_beta_cv"] = -P["beta_cv"]
    P["fdr10"] = (P["eg_pvalue_fdr"] < 0.10).astype(float)
    P["fdr05"] = (P["eg_pvalue_fdr"] < 0.05).astype(float)
    P["traded"] = (P["next_trades"] > 0).astype(float)
    P["n_wins"] = P[[f"win_{k}" for k in EXITS]].sum(axis=1)
    P["days_held_total"] = P[[f"days_{k}" for k in EXITS]].sum(axis=1)
    for k in EXITS:
        P[f"ret_{k}_pct"] = 100 * P[f"pnl_{k}"] / args.capital
    diff = (P[[f"pnl_{k}" for k in EXITS]].sum(axis=1) - P["next_net_pnl"]).abs().max()
    print(f"Sanity check: exit-reason PnL parts vs total net PnL, max abs difference ${diff:.6f}")
    detail = [x.strip() for x in args.detail_metrics.split(",") if x.strip()]
    P.to_csv(f"{args.outdir}/panel.csv", index=False)

    metrics = [m for m, _ in METRICS if m in P.columns and P[m].notna().sum() > 30 and P[m].nunique() > 2]
    joint = [m for m in ["coint_strength", "fast_reversion", "crossings_per_year", "edge_ratio", "stability"] if m in metrics]

    print(f"\nPanel: {len(P)} pair-periods, {P['period'].nunique()} periods, {P['pair'].nunique()} distinct pairs | "
          f"avg net {P['net_ret_pct'].mean():+.3f}% / gross {P['gross_ret_pct'].mean():+.3f}% of capital per 6 months | "
          f"{100 * (P['next_trades'] > 0).mean():.0f}% of pair-periods traded at all | "
          f"{100 * (P['net_ret_pct'] > 0).mean():.0f}% net-profitable")
    if P["period"].nunique() < 8:
        print("WARNING: fewer than 8 periods - t-statistics across periods are unreliable.")

    if args.holdout_start:
        cut = pd.Timestamp(args.holdout_start)
        disc, hold = P[P["window_start"] < cut], P[P["window_start"] >= cut]
    else:
        disc, hold = P, P.iloc[0:0]
    res_d = analyse(disc, metrics, joint, "discovery")
    show(res_d, f"DISCOVERY sample: {disc['period'].nunique()} periods up to "
                f"{(args.holdout_start or 'end of data')}", detail)
    for name in ("ic", "quint", "reg", "gate", "cond", "exit_q", "exit_all"):
        res_d[name].to_csv(f"{args.outdir}/{name}_discovery.csv", index=False)
    if len(hold) and hold["period"].nunique() >= 3:
        res_h = analyse(hold, metrics, joint, "holdout")
        show(res_h, f"HOLDOUT sample (never used to choose anything): {hold['period'].nunique()} periods from {args.holdout_start}", detail)
        for name in ("ic", "quint", "reg", "gate", "cond", "exit_q", "exit_all"):
            res_h[name].to_csv(f"{args.outdir}/{name}_holdout.csv", index=False)

    print("\nHow to read this: a metric earns a place in the screen only if (a) its mean IC / Q5-Q1 spread is positive with "
          "a Holm-adjusted p < 0.05 in the discovery sample AND (b) the sign holds in the holdout. Net = after costs, "
          "gross = before costs: positive gross but negative net means the edge exists but costs eat it.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        key = [m for m in ["coint_strength", "fast_reversion", "crossings_per_year", "edge_ratio",
                           "net_edge_bps_per_year", "stability"] if m in metrics]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, m in zip(axes.ravel(), key):
            for tag, outc, col in (("net", "net_ret_pct", "tab:blue"), ("gross", "gross_ret_pct", "tab:orange")):
                Q = quintile_matrix(disc, m, outc)
                if len(Q):
                    xs = np.arange(5) + (0.2 if tag == "gross" else -0.2)
                    ax.bar(xs, np.nanmean(Q, axis=0), 0.4, label=tag, color=col)
            ax.axhline(0, color="k", lw=.5); ax.set_title(m); ax.set_xticks(range(5)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
            ax.set_ylabel("fwd return, % of capital / 6m")
        axes.ravel()[0].legend()
        for ax in axes.ravel()[len(key):]:
            ax.axis("off")
        plt.suptitle("Forward return by metric quintile (Q5 = highest metric)  -  discovery sample")
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/panel_quintiles.png", dpi=130)
        print(f"Chart -> {args.outdir}/panel_quintiles.png")
    except ImportError:
        pass
    print(f"Files in {args.outdir}/: panel.csv, ic_*.csv, quint_*.csv, reg_*.csv, gate_*.csv, cond_*.csv, exit_q_*.csv, exit_all_*.csv")


if __name__ == "__main__":
    main()
