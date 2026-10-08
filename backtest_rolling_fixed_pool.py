#!/usr/bin/env python3
"""
Rolling pair-selection backtest.

LEGACY: superseded by backtest_rolling_rescreen.py, which uses the same simulator (now in
pairs_common.py) and adds periodic re-screening, group universes and gates. Kept only to
reproduce results that used a one-time candidate pool from screen_pairs_train_test.py.

Every `--period-days` (default 126 trading days ~ 6 months), starting at the end of the
train period:
    1. FORMATION : simulate all candidate pairs (~50) over the previous 6 months with the full
                   cost model and rank them by net PnL (or Sharpe).
    2. TRADING   : trade only the top `--select` (default 15) pairs over the next 6 months.
All candidates are also simulated in each trading window (but not traded in the portfolio) so
we can measure whether formation-period performance actually predicts the next period.

Usage:
    python backtest_rolling_fixed_pool.py raw_data.csv pairs_results.csv --candidates 50 --select 15 --train-frac 0.6

Look-ahead control
------------------
* The candidate pool is picked from the screener output using TRAIN-period statistics only.
  --train-frac / --train-end must match the screener run.
* Each ranking uses only data before its rebalance date; the hedge-ratio / z-score parameters
  are refitted on trailing data that ends before the window being simulated.
* Orders are placed at the close and filled at the next open (--exec open).

Costs: commission, half-spread, square-root market impact (from each ticker's own vol/ADV),
sell-side fees, short borrow, optional financing (see flags).
Simplification: all positions are liquidated at each rebalance (also for pairs that stay
selected), which slightly overstates turnover costs.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from pairs_common import load_panels, perf_stats, rank_corr, simulate_pair

warnings.filterwarnings("ignore")


def pick_candidates(res, args):
    """Candidate pool from TRAIN-period statistics only."""
    d = res.copy()
    # if "train_eg_pvalue_fdr" in d:
    #     d = d[d["train_eg_pvalue_fdr"] < args.max_train_p]
    d = d[(d["train_half_life"] >= args.min_hl) & (d["train_half_life"] <= args.max_hl)]
    if args.require_johansen and "train_johansen_cointegrated" in d:
        d = d[d["train_johansen_cointegrated"].astype(bool)]
    score = d["train_eg_pvalue"].rank(pct=True)
    if "train_johansen_trace" in d:
        score = score + (-(d["train_johansen_trace"] / d["train_johansen_trace_crit95"])).rank(pct=True)
    d = d.assign(sel_score=score).sort_values("sel_score")

    out, used = [], {}
    for _, r in d.iterrows():
        if args.max_per_ticker > 0 and (used.get(r["y"], 0) >= args.max_per_ticker
                                        or used.get(r["x"], 0) >= args.max_per_ticker):
            continue
        out.append(r)
        used[r["y"]] = used.get(r["y"], 0) + 1
        used[r["x"]] = used.get(r["x"], 0) + 1
        if len(out) >= args.candidates:
            break
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prices_csv")
    ap.add_argument("results_csv", help="output of screen_pairs_train_test.py")
    ap.add_argument("--outdir", default="pairs_research/rolling_output")
    # period / selection
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--candidates", type=int, default=50, help="size of the candidate pool")
    ap.add_argument("--select", type=int, default=15, help="pairs traded each period")
    ap.add_argument("--period-days", type=int, default=126, help="rebalance every N trading days (~6 months)")
    ap.add_argument("--formation-days", type=int, default=0, help="ranking window (0 = period-days)")
    ap.add_argument("--rank-metric", choices=["net_pnl", "sharpe"], default="net_pnl")
    ap.add_argument("--min-formation-trades", type=int, default=1,
                    help="pair needs at least this many trades in formation window to be eligible")
    ap.add_argument("--require-positive", action="store_true",
                    help="only trade pairs with positive formation net PnL")
    ap.add_argument("--sel-max-per-ticker", type=int, default=0,
                    help="cap on selected pairs sharing a ticker (0 = off)")
    ap.add_argument("--min-last-period", type=int, default=20,
                    help="skip a final trading window shorter than this many days")
    # pool filters (train-only stats)
    ap.add_argument("--max-train-p", type=float, default=0.05)
    ap.add_argument("--min-hl", type=float, default=2.0)
    ap.add_argument("--max-hl", type=float, default=60.0)
    ap.add_argument("--require-johansen", action="store_true")
    ap.add_argument("--max-per-ticker", type=int, default=5, help="pool diversification cap (0 = off)")
    # strategy
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--wf-step", type=int, default=63)
    ap.add_argument("--wf-lookback", type=int, default=0, help="0 = train length")
    ap.add_argument("--exec", choices=["open", "close"], default="open")
    # sizing
    ap.add_argument("--capital", type=float, default=100_000, help="capital per pair slot")
    ap.add_argument("--gross-leverage", type=float, default=1.0)
    # costs
    ap.add_argument("--comm-ps", type=float, default=0.0035)
    ap.add_argument("--min-comm", type=float, default=0.35)
    ap.add_argument("--half-spread-bps", type=float, default=2.0)
    ap.add_argument("--impact-coef", type=float, default=1.0)
    ap.add_argument("--sell-fee-bps", type=float, default=0.3)
    ap.add_argument("--borrow-bps", type=float, default=50.0)
    ap.add_argument("--financing-bps", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    D = load_panels(args.prices_csv)
    close = D["close"]
    n = len(close)
    n_train = (int((close.index <= pd.Timestamp(args.train_end)).sum())
               if args.train_end else int(n * args.train_frac))
    args.train_len = n_train
    formation_days = args.formation_days or args.period_days
    if n_train - formation_days < 60:
        raise SystemExit("Train period too short for the formation window: reduce --period-days / --formation-days.")

    res = pd.read_csv(args.results_csv)
    pool = pick_candidates(res, args)
    pool = pool[pool["y"].isin(close.columns) & pool["x"].isin(close.columns)]
    if pool.empty:
        raise SystemExit("No candidate pairs after filters (try --max-train-p 1 / --min-hl 0).")
    names = [(r["y"], r["x"]) for _, r in pool.iterrows()]
    k = min(args.select, len(names))
    print(f"{close.shape[1]} tickers, {n} days | train ends {close.index[n_train - 1].date()} | "
          f"candidate pool: {len(names)} pairs | trading top {k} every {args.period_days} days")

    if "hedge_ratio" in pool:
        bad = sum(abs(np.polyfit(np.log(close[r["x"]].values[:n_train]),
                                 np.log(close[r["y"]].values[:n_train]), 1)[0] - r["hedge_ratio"]) > 1e-6
                  for _, r in pool.iterrows())
        if bad:
            print(f"WARNING: {bad} pairs have a train hedge ratio that differs from the results file -> "
                  f"train split probably doesn't match the screener run (look-ahead risk).")

    obs, sel_trades, sel_daily, all_daily, period_rows, mats = [], [], [], [], [], []
    r, p_idx = n_train, 0
    while r < n - args.min_last_period:
        end = min(r + args.period_days, n)
        f0 = r - formation_days

        # ---- formation: simulate every candidate on the previous window ----
        form = {}
        for ty, tx in names:
            d, trs = simulate_pair(ty, tx, D, f0, r, args)
            st = perf_stats(d["net"], args.capital)
            form[(ty, tx)] = dict(f_pnl=st["net_pnl"], f_sharpe=st["sharpe"], f_trades=len(trs))
        F = pd.DataFrame(form).T
        metric = "f_pnl" if args.rank_metric == "net_pnl" else "f_sharpe"
        elig = F[F["f_trades"] >= args.min_formation_trades]
        if args.require_positive:
            elig = elig[elig["f_pnl"] > 0]
        elig = elig.sort_values([metric, "f_pnl"], ascending=False, na_position="last")
        elig_set = set(elig.index)
        chosen, used = [], {}
        for (ty, tx) in elig.index:
            if args.sel_max_per_ticker > 0 and (used.get(ty, 0) >= args.sel_max_per_ticker
                                                or used.get(tx, 0) >= args.sel_max_per_ticker):
                continue
            chosen.append((ty, tx))
            used[ty] = used.get(ty, 0) + 1
            used[tx] = used.get(tx, 0) + 1
            if len(chosen) >= k:
                break
        F["f_rank"] = F[metric].rank(ascending=False, method="first")

        # ---- trading window: simulate all candidates, portfolio uses only the selected ----
        tot_sel = tot_all = None
        rows, cols_d = [], {}
        for ty, tx in names:
            d, trs = simulate_pair(ty, tx, D, r, end, args)
            is_sel = (ty, tx) in chosen
            name = f"{ty}/{tx}"
            cols_d[name] = d["net"]
            tot_all = d["net"] if tot_all is None else tot_all + d["net"]
            if is_sel:
                tot_sel = d["net"] if tot_sel is None else tot_sel + d["net"]
                for tr in trs:
                    tr.update(pair=name, period=p_idx)
                sel_trades += trs
            tdf = pd.DataFrame(trs)
            rows.append(dict(
                period=p_idx, window_start=close.index[r].date(), window_end=close.index[end - 1].date(),
                pair=name, selected=is_sel, eligible=(ty, tx) in elig_set, formation_rank=F.loc[(ty, tx), "f_rank"],
                formation_net_pnl=F.loc[(ty, tx), "f_pnl"], formation_sharpe=F.loc[(ty, tx), "f_sharpe"],
                formation_trades=int(F.loc[(ty, tx), "f_trades"]),
                next_net_pnl=d["net"].sum(), next_gross_pnl=d["gross"].sum(), next_costs=d["costs"].sum(),
                next_trades=len(trs),
                next_win_rate=float((tdf["net"] > 0).mean()) if len(tdf) else np.nan))
        obs += rows
        mats.append(pd.DataFrame(cols_d))
        sel_daily.append(tot_sel if tot_sel is not None else pd.Series(0.0, index=close.index[r:end]))
        all_daily.append(tot_all)

        O = pd.DataFrame(rows)
        S_, U_ = O[O["selected"]], O[~O["selected"]]
        period_rows.append(dict(
            period=p_idx, window_start=close.index[r].date(), window_end=close.index[end - 1].date(),
            days=end - r, n_selected=len(S_), selected_pairs=", ".join(S_["pair"]),
            sel_net_pnl=S_["next_net_pnl"].sum(), sel_return_pct=100 * S_["next_net_pnl"].sum() / (args.capital * k),
            all_net_pnl=O["next_net_pnl"].sum(), all_return_pct=100 * O["next_net_pnl"].sum() / (args.capital * len(names)),
            sel_hit_rate=float((S_["next_net_pnl"] > 0).mean()) if len(S_) else np.nan,
            unsel_hit_rate=float((U_["next_net_pnl"] > 0).mean()) if len(U_) else np.nan,
            sel_avg_formation_pnl=S_["formation_net_pnl"].mean(),
            sel_avg_next_pnl=S_["next_net_pnl"].mean(), unsel_avg_next_pnl=U_["next_net_pnl"].mean(),
            formation_vs_next_rank_corr=rank_corr(O["formation_net_pnl"], O["next_net_pnl"])))
        print(f"Period {p_idx}: {close.index[r].date()} -> {close.index[end - 1].date()} | "
              f"selected net PnL {period_rows[-1]['sel_net_pnl']:>10,.0f} "
              f"({period_rows[-1]['sel_return_pct']:+.2f}%) | all-candidates {period_rows[-1]['all_return_pct']:+.2f}% | "
              f"hit rate sel/unsel {period_rows[-1]['sel_hit_rate']:.0%}/{period_rows[-1]['unsel_hit_rate']:.0%}")
        r, p_idx = end, p_idx + 1

    # ---- aggregate ----
    obs_df = pd.DataFrame(obs)
    per_df = pd.DataFrame(period_rows)
    sel_s, all_s = pd.concat(sel_daily), pd.concat(all_daily)
    cap_sel, cap_all = args.capital * k, args.capital * len(names)
    ps, pa = perf_stats(sel_s, cap_sel), perf_stats(all_s, cap_all)
    trades_df = pd.DataFrame(sel_trades)

    # per-pair summary (only the periods in which the pair was selected)
    sel_obs = obs_df[obs_df["selected"]]
    pair_sum = (sel_obs.groupby("pair")
                .agg(periods_selected=("period", "count"), n_trades=("next_trades", "sum"),
                     gross_pnl=("next_gross_pnl", "sum"), total_costs=("next_costs", "sum"),
                     net_pnl=("next_net_pnl", "sum"),
                     periods_profitable=("next_net_pnl", lambda s: int((s > 0).sum())))
                .sort_values("net_pnl", ascending=False).reset_index())

    obs_df.to_csv(f"{args.outdir}/selection_log.csv", index=False)
    pd.concat(mats).to_csv(f"{args.outdir}/daily_pair_net_pnl.csv")  # input for the random-selection baseline
    per_df.to_csv(f"{args.outdir}/period_summary.csv", index=False)
    pair_sum.to_csv(f"{args.outdir}/pair_summary_selected.csv", index=False)
    pd.DataFrame({"SELECTED_net_pnl": sel_s, "ALL_CANDIDATES_net_pnl": all_s,
                  "SELECTED_cum": sel_s.cumsum(), "ALL_CANDIDATES_cum": all_s.cumsum()}).to_csv(
        f"{args.outdir}/daily_portfolio_pnl.csv")
    if len(trades_df):
        cols = ["period", "pair", "entry_date", "exit_date", "side", "exit_reason", "days_held", "entry_z",
                "exit_z", "beta", "shares_y", "shares_x", "entry_notional", "gross", "costs", "net"]
        trades_df[cols].to_csv(f"{args.outdir}/trades_selected.csv", index=False)

    pooled_corr = rank_corr(obs_df["formation_net_pnl"], obs_df["next_net_pnl"])
    print("\n================ RESULTS ================")
    print(f"SELECTED top-{k} (rolling) : net PnL {ps['net_pnl']:>10,.0f} | return {ps['return_pct']:+.2f}% on {cap_sel:,.0f} | "
          f"ann. ret {ps['ann_return_pct']:.2f}% | vol {ps['ann_vol_pct']:.2f}% | Sharpe {ps['sharpe']:.2f} | maxDD {ps['max_drawdown_pct']:.2f}%")
    print(f"ALL {len(names)} candidates (no selection): net PnL {pa['net_pnl']:>10,.0f} | return {pa['return_pct']:+.2f}% on {cap_all:,.0f} | "
          f"ann. ret {pa['ann_return_pct']:.2f}% | vol {pa['ann_vol_pct']:.2f}% | Sharpe {pa['sharpe']:.2f} | maxDD {pa['max_drawdown_pct']:.2f}%")
    print(f"Hit rate (pair-period net PnL > 0): selected {(sel_obs['next_net_pnl'] > 0).mean():.1%} | "
          f"not selected {(obs_df[~obs_df['selected']]['next_net_pnl'] > 0).mean():.1%}")
    print(f"Avg next-period net PnL per pair : selected {sel_obs['next_net_pnl'].mean():,.0f} | "
          f"not selected {obs_df[~obs_df['selected']]['next_net_pnl'].mean():,.0f}")
    print(f"Rank correlation formation vs next-period PnL (all pair-periods): {pooled_corr:+.3f}   "
          f"[~0 => past performance doesn't predict future]")
    print(f"Total costs on selected: {sel_obs['next_costs'].sum():,.0f} "
          f"({100 * sel_obs['next_costs'].sum() / abs(sel_obs['next_gross_pnl'].sum()) if sel_obs['next_gross_pnl'].sum() else np.nan:.1f}% of |gross|)")
    print("\nPer period:\n" + per_df[["period", "window_start", "window_end", "sel_net_pnl", "sel_return_pct",
                                      "all_return_pct", "sel_hit_rate", "formation_vs_next_rank_corr"]].round(3).to_string(index=False))

    # ---- chart ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))
        ax[0].plot(sel_s.index, 100 * sel_s.cumsum() / cap_sel, lw=2, label=f"Rolling top-{k}")
        ax[0].plot(all_s.index, 100 * all_s.cumsum() / cap_all, lw=1.5, ls="--", label=f"All {len(names)} candidates")
        for i, row in per_df.iterrows():
            ax[0].axvline(pd.Timestamp(row["window_start"]), color="grey", lw=.5, ls=":")
        ax[0].axhline(0, color="k", lw=.5); ax[0].set_ylabel("cumulative net return on capital (%)")
        ax[0].set_title("Cumulative net return (dotted lines = rebalances)"); ax[0].legend(); ax[0].grid(alpha=.3)
        x = np.arange(len(per_df)); w = .38
        ax[1].bar(x - w / 2, per_df["sel_return_pct"], w, label=f"Selected top-{k}")
        ax[1].bar(x + w / 2, per_df["all_return_pct"], w, label="All candidates")
        ax[1].set_xticks(x); ax[1].set_xticklabels([str(s) for s in per_df["window_start"]], rotation=30)
        ax[1].set_title("Return by 6-month period (%)"); ax[1].legend(); ax[1].grid(alpha=.3)
        u, s2 = obs_df[~obs_df["selected"]], obs_df[obs_df["selected"]]
        ax[2].scatter(u["formation_net_pnl"], u["next_net_pnl"], s=12, alpha=.5, label="not selected")
        ax[2].scatter(s2["formation_net_pnl"], s2["next_net_pnl"], s=14, alpha=.8, color="tab:red", label="selected")
        ax[2].axhline(0, color="k", lw=.5); ax[2].axvline(0, color="k", lw=.5)
        ax[2].set_xlabel("formation-window net PnL"); ax[2].set_ylabel("next-window net PnL")
        ax[2].set_title(f"Does past performance persist? (rank corr {pooled_corr:+.2f})")
        ax[2].legend(); ax[2].grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/rolling_chart.png", dpi=130)
        print(f"\nChart -> {args.outdir}/rolling_chart.png")
    except ImportError:
        print("matplotlib not installed - skipped chart")
    print(f"Files in {args.outdir}/: period_summary.csv, selection_log.csv, pair_summary_selected.csv, "
          f"daily_portfolio_pnl.csv, daily_pair_net_pnl.csv, trades_selected.csv")


if __name__ == "__main__":
    main()
