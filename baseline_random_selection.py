#!/usr/bin/env python3
"""
Random-selection baseline for the rolling pair-selection backtest.

Question answered: does ranking pairs on last period's performance add value, or would
picking K pairs at random from the same eligible pool do just as well?

Method (a permutation / Monte-Carlo test, no re-simulation needed)
-------------------------------------------------------------------
Each pair is simulated independently with its own capital, so a portfolio's daily PnL is just
the sum of its pairs' daily PnL. For every rebalance period we draw K pairs at random from
the pairs that were ELIGIBLE for selection (same pool, same filters, same K as the real
strategy), assemble the daily portfolio PnL across all periods, and repeat --sims times.
The actual rolling top-K result is then located inside this null distribution.

Also reported:
  * bottom-K by formation rank (if ranking has skill, this should do WORSE than random)
  * average next-period PnL / hit rate by formation-rank quintile (monotonic = real signal)
  * per-period percentile of the actual selection vs random

Usage:
    python backtest_rolling_rescreen.py raw_data.csv ... --outdir rolling_output
    #   (or the legacy: python backtest_rolling_fixed_pool.py prices.csv pairs_results.csv ... --outdir rolling_output)
    python baseline_random_selection.py rolling_output --sims 5000
"""
import argparse
import os

import numpy as np
import pandas as pd


def sharpe_vec(R, cap):
    r = R / cap
    sd = r.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, r.mean(axis=0) / sd * np.sqrt(252), np.nan)


def maxdd_vec(R, cap):
    eq = cap + np.cumsum(R, axis=0)
    return 100 * (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)


def stats_1d(x, cap):
    R = np.asarray(x, float).reshape(-1, 1)
    return dict(net_pnl=R.sum(), return_pct=100 * R.sum() / cap,
                sharpe=float(sharpe_vec(R, cap)[0]), max_drawdown_pct=float(maxdd_vec(R, cap)[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rolling_dir", help="output directory of backtest_rolling_rescreen.py (or backtest_rolling_fixed_pool.py)")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capital", type=float, default=100_000, help="capital per pair slot (as in the backtest)")
    ap.add_argument("--rank-metric", choices=["net_pnl", "sharpe"], default="net_pnl",
                    help="must match the metric used in the rolling backtest")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.rolling_dir, "baseline")
    os.makedirs(outdir, exist_ok=True)

    mat = pd.read_csv(os.path.join(args.rolling_dir, "daily_pair_net_pnl.csv"), index_col=0, parse_dates=True)
    log = pd.read_csv(os.path.join(args.rolling_dir, "selection_log.csv"), parse_dates=["window_start", "window_end"])
    if "eligible" not in log:
        raise SystemExit("selection_log.csv has no 'eligible' column - re-run the (updated) rolling backtest.")
    fcol = "formation_net_pnl" if args.rank_metric == "net_pnl" else "formation_sharpe"
    rng = np.random.default_rng(args.seed)
    S = args.sims
    periods = sorted(log["period"].unique())
    K = int(log[log["selected"]].groupby("period").size().max())
    cap = args.capital * K

    act_parts, bot_parts, rnd_parts, idx_parts, per_rows, bucket_rows = [], [], [], [], [], []
    for p in periods:
        L = log[log["period"] == p].set_index("pair")
        w0, w1 = L["window_start"].iloc[0], L["window_end"].iloc[0]
        M = mat.loc[w0:w1, L.index]                      # days x pairs (same order as L)
        sel = L["selected"].values
        elig = L["eligible"].values.astype(bool)
        kp = int(sel.sum())
        Mv = M.values
        act = Mv[:, sel].sum(axis=1)

        E = np.where(elig)[0]
        m = len(E)
        kp = min(kp, m)
        # random K from the eligible pool, S times (vectorised via argsort of uniforms)
        pick = np.argsort(rng.random((S, m)), axis=1)[:, :kp]
        mask = np.zeros((S, m))
        np.put_along_axis(mask, pick, 1.0, axis=1)
        rnd = Mv[:, E] @ mask.T                           # days x S

        # deterministic bottom-K among eligible by formation metric
        order = L.iloc[E][fcol].sort_values(ascending=True, na_position="first").index[:kp]
        bot = M[list(order)].sum(axis=1).values

        act_parts.append(act); bot_parts.append(bot); rnd_parts.append(rnd); idx_parts.append(M.index)

        rp = rnd.sum(axis=0)
        per_rows.append(dict(period=p, window_start=w0.date(), window_end=w1.date(), n_eligible=m, k=kp,
                             actual_net_pnl=act.sum(), random_mean=rp.mean(), random_p5=np.percentile(rp, 5),
                             random_p95=np.percentile(rp, 95), actual_percentile=100 * (rp < act.sum()).mean(),
                             bottomK_net_pnl=bot.sum()))

        # formation-rank quintiles among eligible pairs (1 = best formation performance)
        Le = L.iloc[E].copy()
        if len(Le) >= 5:
            Le["quintile"] = pd.qcut(Le[fcol].rank(ascending=False, method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
            bucket_rows.append(Le[["quintile", "next_net_pnl"]])

    act_s = np.concatenate(act_parts)
    bot_s = np.concatenate(bot_parts)
    R = np.vstack(rnd_parts)                              # T x S
    dates = idx_parts[0].append(idx_parts[1:]) if len(idx_parts) > 1 else idx_parts[0]

    # ---- distribution statistics ----
    r_pnl = R.sum(axis=0)
    r_sh = sharpe_vec(R, cap)
    r_dd = maxdd_vec(R, cap)
    A, B = stats_1d(act_s, cap), stats_1d(bot_s, cap)

    def pct_row(label, actual, bottom, dist, higher_better=True):
        d = dist[np.isfinite(dist)]
        pctile = 100 * (d < actual).mean()
        p = ((d >= actual).sum() + 1) / (len(d) + 1) if higher_better else ((d <= actual).sum() + 1) / (len(d) + 1)
        return dict(metric=label, actual_top_k=actual, bottom_k=bottom, random_mean=d.mean(), random_std=d.std(),
                    random_p5=np.percentile(d, 5), random_p50=np.percentile(d, 50), random_p95=np.percentile(d, 95),
                    actual_percentile=pctile, p_value_one_sided=p)

    summ = pd.DataFrame([
        pct_row("net_pnl", A["net_pnl"], B["net_pnl"], r_pnl),
        pct_row("sharpe", A["sharpe"], B["sharpe"], r_sh),
        pct_row("max_drawdown_pct", A["max_drawdown_pct"], B["max_drawdown_pct"], r_dd),  # less negative = better
    ])
    per_df = pd.DataFrame(per_rows)
    bk = pd.concat(bucket_rows) if bucket_rows else pd.DataFrame(columns=["quintile", "next_net_pnl"])
    bucket = (bk.groupby("quintile")["next_net_pnl"]
              .agg(n_pair_periods="count", mean_next_pnl="mean", median_next_pnl="median",
                   hit_rate=lambda s: (s > 0).mean()).reset_index()) if len(bk) else pd.DataFrame()

    summ.to_csv(f"{outdir}/baseline_summary.csv", index=False)
    per_df.to_csv(f"{outdir}/baseline_by_period.csv", index=False)
    bucket.to_csv(f"{outdir}/formation_quintile_table.csv", index=False)
    pd.DataFrame({"net_pnl": r_pnl, "sharpe": r_sh, "max_drawdown_pct": r_dd}).to_csv(
        f"{outdir}/random_distribution.csv", index=False)

    # ---- report ----
    print(f"Periods: {len(periods)} | K = {K} | sims = {S} | capital basis = {cap:,.0f}")
    print("\n=== Actual rolling top-K vs random-K from the same eligible pool ===")
    show = summ[["metric", "actual_top_k", "bottom_k", "random_mean", "random_p5", "random_p95",
                 "actual_percentile", "p_value_one_sided"]].copy()
    print(show.round(3).to_string(index=False))
    print("\n(actual_percentile: share of random portfolios the actual one beats; p_value: P[random >= actual];\n"
          " for max_drawdown a HIGHER number (less negative) is better, so read the percentile, not p.)")
    print("\n=== Per period ===")
    print(per_df[["period", "window_start", "window_end", "n_eligible", "k", "actual_net_pnl",
                  "random_mean", "actual_percentile", "bottomK_net_pnl"]].round(1).to_string(index=False))
    if len(bucket):
        print("\n=== Next-period PnL by formation-rank quintile (1 = best formation) ===")
        print(bucket.round(3).to_string(index=False))
    pv = summ.loc[0, "p_value_one_sided"]
    verdict = ("ranking adds statistically meaningful value" if pv < 0.05 else
               "NOT distinguishable from random selection at the 5% level")
    print(f"\nVerdict on net PnL: p = {pv:.3f} -> {verdict}.")

    # ---- charts ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 14))
        ax[0].hist(r_pnl, bins=60, color="tab:blue", alpha=.7)
        ax[0].axvline(A["net_pnl"], color="tab:red", lw=2, label=f"actual top-K ({A['net_pnl']:,.0f})")
        ax[0].axvline(B["net_pnl"], color="k", ls="--", label=f"bottom-K ({B['net_pnl']:,.0f})")
        ax[0].set_title(f"Total net PnL: random-{K} selection ({S} sims) vs actual"); ax[0].legend()
        d = r_sh[np.isfinite(r_sh)]
        ax[1].hist(d, bins=60, color="tab:blue", alpha=.7)
        ax[1].axvline(A["sharpe"], color="tab:red", lw=2, label=f"actual ({A['sharpe']:.2f})")
        ax[1].axvline(B["sharpe"], color="k", ls="--", label=f"bottom-K ({B['sharpe']:.2f})")
        ax[1].set_title("Sharpe ratio: random vs actual"); ax[1].legend()
        cum = np.cumsum(R, axis=0)
        lo, med, hi = np.percentile(cum, [5, 50, 95], axis=1)
        ax[2].fill_between(dates, lo, hi, alpha=.25, label="random 5-95%")
        ax[2].plot(dates, med, color="tab:blue", lw=1, label="random median")
        ax[2].plot(dates, np.cumsum(act_s), color="tab:red", lw=2, label="actual top-K")
        ax[2].plot(dates, np.cumsum(bot_s), color="k", lw=1, ls="--", label="bottom-K")
        ax[2].axhline(0, color="k", lw=.5); ax[2].set_title("Cumulative net PnL"); ax[2].legend(); ax[2].grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/baseline_chart.png", dpi=130)
        print(f"Chart -> {outdir}/baseline_chart.png")
    except ImportError:
        print("matplotlib not installed - skipped chart")
    print(f"Files in {outdir}/: baseline_summary.csv, baseline_by_period.csv, formation_quintile_table.csv, random_distribution.csv")


if __name__ == "__main__":
    main()
