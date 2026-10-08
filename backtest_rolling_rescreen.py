#!/usr/bin/env python3
"""
Rolling backtest with periodic RE-SCREENING of cointegration (single price file).
Shared simulator / screening code lives in pairs_common.py.

Timeline (all decisions use only data BEFORE the decision date):
    every --screen-every days (default 252 ~ 1 year):
        RE-SCREEN on the trailing --screen-window days (default 504) in two separate steps:
        1. UNIVERSE : with --groups-csv only pairs inside the same industry group are tested
                      (far fewer tests -> multiple-testing correction is much less punishing,
                      and every pair has an economic reason to be related).
        2. EXISTENCE GATE : a pair must pass Engle-Granger with BH-FDR < --gate-fdr (family = pairs
                      actually tested) and optionally Johansen (--gate-johansen), KPSS (--gate-kpss),
                      sub-window stability (--min-stability), half-life range, and a cost-aware edge
                      ratio (--min-edge-ratio). An empty pool = stay in cash until the next screen.
        3. TRADABILITY RANKING of the survivors (--pool-score): estimated net edge in bps/year after
                      costs, sub-window stability, mean-crossing rate and short half-life; the top
                      --candidates (default 50) form the pool.
    every --period-days (default 126 ~ 6 months):
        FORMATION : simulate the pool over the previous 6 months with the full cost model and
                    rank by net return (--rank-metric net_pnl | sharpe)
        TRADING   : trade the top --select (default 15) pairs over the next 6 months.
All pool pairs are also simulated in each trading window (not traded in the portfolio) so the
random-selection baseline (baseline_random_selection.py) can be run on the output folder.

Usage:
    python backtest_rolling_rescreen.py raw_data.csv --candidates 50 --select 15
    python backtest_rolling_rescreen.py raw_data.csv --start-date 2019-01-01 --screen-window 756 --jobs 8

Needs ONE continuous price file with at least --screen-window days of history before the first
trading date (use --start-date, default = right after the first screen window). Don't stitch two
separately downloaded yfinance files: dividend-adjusted prices jump at the seam.

Costs: commission, half-spread, square-root market impact, sell-side fees, short borrow and
optional financing (see flags). All positions are liquidated at every rebalance.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from pairs_common import load_panels, pair_indices, perf_stats, rank_corr, screen_universe, simulate_pair

warnings.filterwarnings("ignore")


def build_pool(scr, args):
    """Existence gate first, then rank the survivors on tradability."""
    d = scr[(scr["half_life"] >= args.min_hl) & (scr["half_life"] <= args.max_hl)]
    d = d[d["eg_pvalue"] <= args.max_screen_p]
    # ---- 1. existence gate ----
    # if args.gate_fdr > 0:
    #     d = d[d["eg_pvalue_fdr"] < args.gate_fdr]
    if args.gate_johansen:
        d = d[d["johansen_ok"] == True]                                # noqa: E712
    if args.gate_kpss:
        d = d[d["kpss_pvalue"] > 0.05]                                 # KPSS null = stationary
    if args.min_stability > 0:
        d = d[d["stability"] >= args.min_stability]
    if args.min_edge_ratio > 0:
        d = d[d["edge_ratio"] >= args.min_edge_ratio]
    # ---- 2. tradability ranking of the survivors ----
    if len(d):
        if args.pool_score == "tstat":
            score = -d["eg_tstat"]
        elif args.pool_score == "net_edge":
            score = d["net_edge_bps_per_year"]
        else:                                                          # composite: equal-weight percentile ranks
            score = (d["net_edge_bps_per_year"].rank(pct=True) + d["stability"].rank(pct=True)
                     + d["crossings_per_year"].rank(pct=True) + (-d["half_life"]).rank(pct=True)) / 4
        d = d.assign(pool_score=score).sort_values("pool_score", ascending=False)
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
    if not out:
        return pd.DataFrame(columns=["screen_rank"] + list(d.columns))
    pool = pd.DataFrame(out).reset_index(drop=True)
    pool.insert(0, "screen_rank", np.arange(1, len(pool) + 1))
    return pool


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prices_csv")
    ap.add_argument("--outdir", default="rescreen_output")
    ap.add_argument("--start-date", default=None, help="first trading date (needs --screen-window days of history before it)")
    # screening / pool
    ap.add_argument("--screen-window", type=int, default=504, help="trailing days used for each cointegration screen")
    ap.add_argument("--screen-every", type=int, default=252, help="re-screen every N days (~1 year)")
    ap.add_argument("--candidates", type=int, default=50, help="pool size = N most cointegrated pairs")
    ap.add_argument("--min-corr", type=float, default=0.0, help="skip pairs whose daily-return correlation is below this")
    ap.add_argument("--min-hl", type=float, default=2.0)
    ap.add_argument("--max-hl", type=float, default=60.0)
    ap.add_argument("--max-screen-p", type=float, default=1.0, help="optional cap on raw EG p-value (1 = off)")
    ap.add_argument("--max-per-ticker", type=int, default=5, help="pool diversification cap (0 = off)")
    ap.add_argument("--groups-csv", default=None,
                    help="CSV with columns ticker,group. Only pairs inside the same group are tested")
    # existence gate
    ap.add_argument("--gate-fdr", "--require-fdr", dest="gate_fdr", type=float, default=0.10,
                    help="existence gate: BH-FDR-adjusted EG p-value must be below this (0 = off)")
    ap.add_argument("--gate-johansen", action="store_true", help="also require the Johansen test to pass (95%%)")
    ap.add_argument("--gate-kpss", action="store_true", help="also require KPSS not to reject stationarity (p>0.05)")
    ap.add_argument("--min-stability", type=float, default=0.0,
                    help="min share of sub-windows with a valid half-life (e.g. 0.67); 0 = off")
    ap.add_argument("--min-edge-ratio", type=float, default=0.0,
                    help="min (gross bps per round trip) / (estimated cost bps), e.g. 3; 0 = off")
    ap.add_argument("--n-subwindows", type=int, default=3, help="sub-windows for the stability metric")
    # tradability ranking of survivors
    ap.add_argument("--pool-score", choices=["composite", "net_edge", "tstat"], default="composite",
                    help="composite = mean percentile rank of net edge, stability, crossing rate, short half-life")
    ap.add_argument("--jobs", type=int, default=None)
    # selection
    ap.add_argument("--select", type=int, default=15)
    ap.add_argument("--period-days", type=int, default=126)
    ap.add_argument("--formation-days", type=int, default=0, help="0 = period-days")
    ap.add_argument("--rank-metric", choices=["net_pnl", "sharpe"], default="net_pnl")
    ap.add_argument("--min-formation-trades", type=int, default=1)
    ap.add_argument("--require-positive", action="store_true")
    ap.add_argument("--sel-max-per-ticker", type=int, default=0)
    ap.add_argument("--min-last-period", type=int, default=20)
    # strategy
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--wf-step", type=int, default=63)
    ap.add_argument("--wf-lookback", type=int, default=0, help="0 = screen window")
    ap.add_argument("--exec", choices=["open", "close"], default="open")
    # sizing / costs
    ap.add_argument("--capital", type=float, default=100_000, help="capital per pair slot")
    ap.add_argument("--gross-leverage", type=float, default=1.0)
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
    dates = close.index
    n = len(close)
    logp = np.log(close)
    formation_days = args.formation_days or args.period_days
    args.train_len = args.screen_window                      # default hedge-ratio lookback in the engine
    start = int(dates.searchsorted(pd.Timestamp(args.start_date))) if args.start_date else args.screen_window
    if start < args.screen_window:
        raise SystemExit(f"Need >= {args.screen_window} days of history before the first trading date "
                         f"(start index {start}). Use a later --start-date or a smaller --screen-window.")
    if start >= n - args.min_last_period:
        raise SystemExit("Not enough data after the first screen window to trade.")
    groups = None
    if args.groups_csv:
        g = pd.read_csv(args.groups_csv)
        groups = dict(zip(g["ticker"], g["group"]))
        ungrouped = [c for c in close.columns if c not in groups]
        print(f"Groups: {len({groups[c] for c in close.columns if c in groups})} groups from {args.groups_csv}; "
              f"{len(ungrouped)} tickers without a group are not tested {ungrouped[:12]}")
    n_pairs = len(pair_indices(list(close.columns), groups)[0])
    gated = (args.gate_fdr > 0 or args.gate_johansen or args.gate_kpss or args.min_stability > 0
             or args.min_edge_ratio > 0)
    ppk = max(1, int(round(args.screen_every / args.period_days)))       # periods per screen
    K = args.select
    print(f"{close.shape[1]} tickers ({n_pairs} pairs to test{' within groups' if groups else ''}), {n} days | first trade "
          f"{dates[start].date()} | screen every {ppk} period(s) on a {args.screen_window}-day window | "
          f"pool {args.candidates} -> top {K} every {args.period_days} days")

    obs, sel_trades, period_rows, mats, screen_rows = [], [], [], [], []
    sel_ret, all_ret, sel_usd, all_usd = [], [], [], []
    prev, names, meta, screen_id = set(), [], None, -1
    r, p_idx = start, 0
    while r < n - args.min_last_period:
        end = min(r + args.period_days, n)
        f0 = r - formation_days

        # ---- annual re-screen of the whole universe on data strictly before r ----
        if p_idx % ppk == 0:
            screen_id += 1
            print(f"\nScreen {screen_id}: all pairs on {dates[r - args.screen_window].date()} -> {dates[r - 1].date()} ...")
            scr = screen_universe(logp, r - args.screen_window, r, args, groups)
            pool_df = build_pool(scr, args)
            n_fdr = int((scr["eg_pvalue_fdr"] < 0.05).sum())
            if pool_df.empty:
                if not gated:
                    raise SystemExit("No pairs passed the screening filters (try --min-hl 0 / --max-screen-p 1).")
                names, meta, prev = [], None, set()
                print(f"  screened {len(scr)} pairs | FDR<0.05: {n_fdr} | pool EMPTY -> staying in cash until the next screen")
            else:
                names = list(zip(pool_df["y"], pool_df["x"]))
                cur = set(names)
                print(f"  screened {len(scr)} pairs | FDR<0.05: {n_fdr} | "
                      f"pool {len(cur)} (median half-life {pool_df['half_life'].median():.1f}d, "
                      f"max EG p {pool_df['eg_pvalue'].max():.3f}, median stability {pool_df['stability'].median():.2f}, "
                      f"median net edge {pool_df['net_edge_bps_per_year'].median():.0f} bps/yr) | "
                      f"overlap with previous pool: {len(cur & prev)}")
                prev = cur
                meta = pool_df.set_index(["y", "x"])
                sl = pool_df.copy()
                sl.insert(0, "screen_date", dates[r].date())
                sl.insert(0, "screen_id", screen_id)
                screen_rows.append(sl)

        if not names:                                              # empty pool: stay in cash this period
            zero = pd.Series(0.0, index=dates[r:end])
            for lst in (sel_usd, all_usd, sel_ret, all_ret):
                lst.append(zero)
            period_rows.append(dict(period=p_idx, screen_id=screen_id, window_start=dates[r].date(),
                                    window_end=dates[end - 1].date(), days=end - r, pool_size=0, n_selected=0,
                                    selected_pairs="", sel_net_pnl=0.0, sel_return_pct=0.0,
                                    all_net_pnl=0.0, all_return_pct=0.0))
            print(f"Period {p_idx}: {dates[r].date()} -> {dates[end - 1].date()} | no significant pairs -> in cash")
            r, p_idx = end, p_idx + 1
            continue

        # ---- formation: simulate every pool pair on the previous window ----
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
            if len(chosen) >= K:
                break
        F["f_rank"] = F[metric].rank(ascending=False, method="first")

        # ---- trading window ----
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
                period=p_idx, screen_id=screen_id, window_start=dates[r].date(), window_end=dates[end - 1].date(),
                pair=name, selected=is_sel, eligible=(ty, tx) in elig_set,
                screen_rank=int(meta.loc[(ty, tx), "screen_rank"]), screen_tstat=meta.loc[(ty, tx), "eg_tstat"],
                screen_pvalue=meta.loc[(ty, tx), "eg_pvalue"], screen_half_life=meta.loc[(ty, tx), "half_life"],
                screen_pvalue_fdr=meta.loc[(ty, tx), "eg_pvalue_fdr"], screen_group=meta.loc[(ty, tx), "group"],
                screen_score=meta.loc[(ty, tx), "pool_score"], screen_stability=meta.loc[(ty, tx), "stability"],
                screen_crossings_py=meta.loc[(ty, tx), "crossings_per_year"],
                screen_edge_ratio=meta.loc[(ty, tx), "edge_ratio"],
                screen_net_edge_bps_year=meta.loc[(ty, tx), "net_edge_bps_per_year"],
                formation_rank=F.loc[(ty, tx), "f_rank"], formation_net_pnl=F.loc[(ty, tx), "f_pnl"],
                formation_sharpe=F.loc[(ty, tx), "f_sharpe"], formation_trades=int(F.loc[(ty, tx), "f_trades"]),
                next_net_pnl=d["net"].sum(), next_gross_pnl=d["gross"].sum(), next_costs=d["costs"].sum(),
                next_trades=len(trs),
                next_win_rate=float((tdf["net"] > 0).mean()) if len(tdf) else np.nan))
        obs += rows
        mats.append(pd.DataFrame(cols_d))
        if tot_sel is None:
            tot_sel = pd.Series(0.0, index=dates[r:end])
        sel_usd.append(tot_sel); all_usd.append(tot_all)
        sel_ret.append(tot_sel / (args.capital * K))
        all_ret.append(tot_all / (args.capital * len(names)))

        O = pd.DataFrame(rows)
        S_, U_ = O[O["selected"]], O[~O["selected"]]
        period_rows.append(dict(
            period=p_idx, screen_id=screen_id, window_start=dates[r].date(), window_end=dates[end - 1].date(),
            days=end - r, pool_size=len(names), n_selected=len(S_), selected_pairs=", ".join(S_["pair"]),
            sel_net_pnl=S_["next_net_pnl"].sum(), sel_return_pct=100 * S_["next_net_pnl"].sum() / (args.capital * K),
            all_net_pnl=O["next_net_pnl"].sum(),
            all_return_pct=100 * O["next_net_pnl"].sum() / (args.capital * len(names)),
            sel_hit_rate=float((S_["next_net_pnl"] > 0).mean()) if len(S_) else np.nan,
            unsel_hit_rate=float((U_["next_net_pnl"] > 0).mean()) if len(U_) else np.nan,
            sel_avg_formation_pnl=S_["formation_net_pnl"].mean(),
            sel_avg_next_pnl=S_["next_net_pnl"].mean(), unsel_avg_next_pnl=U_["next_net_pnl"].mean(),
            formation_vs_next_rank_corr=rank_corr(O["formation_net_pnl"], O["next_net_pnl"])))
        pr = period_rows[-1]
        print(f"Period {p_idx}: {pr['window_start']} -> {pr['window_end']} | selected net PnL {pr['sel_net_pnl']:>10,.0f} "
              f"({pr['sel_return_pct']:+.2f}%) | whole pool {pr['all_return_pct']:+.2f}% | "
              f"hit rate sel/unsel {pr['sel_hit_rate']:.0%}/{pr['unsel_hit_rate']:.0%}")
        r, p_idx = end, p_idx + 1

    # ---- aggregate ----
    if not obs:
        print("\nNo pair was ever eligible to trade (every pool was empty): the strategy stayed in cash throughout.")
        return
    obs_df, per_df = pd.DataFrame(obs), pd.DataFrame(period_rows)
    sel_r, all_r = pd.concat(sel_ret), pd.concat(all_ret)          # daily returns on capital
    sel_d, all_d = pd.concat(sel_usd), pd.concat(all_usd)
    ps, pa = perf_stats(sel_r, 1.0), perf_stats(all_r, 1.0)
    sel_obs, uns_obs = obs_df[obs_df["selected"]], obs_df[~obs_df["selected"]]
    trades_df = pd.DataFrame(sel_trades)
    pair_sum = (sel_obs.groupby("pair")
                .agg(periods_selected=("period", "count"), n_trades=("next_trades", "sum"),
                     gross_pnl=("next_gross_pnl", "sum"), total_costs=("next_costs", "sum"),
                     net_pnl=("next_net_pnl", "sum"),
                     periods_profitable=("next_net_pnl", lambda s: int((s > 0).sum())))
                .sort_values("net_pnl", ascending=False).reset_index())

    obs_df.to_csv(f"{args.outdir}/selection_log.csv", index=False)
    per_df.to_csv(f"{args.outdir}/period_summary.csv", index=False)
    pair_sum.to_csv(f"{args.outdir}/pair_summary_selected.csv", index=False)
    pd.concat(screen_rows).to_csv(f"{args.outdir}/screen_log.csv", index=False)
    pd.concat(mats).to_csv(f"{args.outdir}/daily_pair_net_pnl.csv")     # input for baseline_random_selection.py
    pd.DataFrame({"SELECTED_net_pnl": sel_d, "POOL_net_pnl": all_d, "SELECTED_return": sel_r, "POOL_return": all_r,
                  "SELECTED_cum_return": sel_r.cumsum(), "POOL_cum_return": all_r.cumsum()}
                 ).to_csv(f"{args.outdir}/daily_portfolio_pnl.csv")
    if len(trades_df):
        cols = ["period", "pair", "entry_date", "exit_date", "side", "exit_reason", "days_held", "entry_z",
                "exit_z", "beta", "shares_y", "shares_x", "entry_notional", "gross", "costs", "net"]
        trades_df[cols].to_csv(f"{args.outdir}/trades_selected.csv", index=False)

    screen_corr = rank_corr(-obs_df["screen_rank"], obs_df["next_net_pnl"])
    pooled_corr = rank_corr(obs_df["formation_net_pnl"], obs_df["next_net_pnl"])
    print("\n================ RESULTS ================")
    print(f"SELECTED top-{K} (rolling)     : net PnL {sel_d.sum():>10,.0f} | return {ps['return_pct']:+.2f}% on {args.capital * K:,.0f} | "
          f"ann. ret {ps['ann_return_pct']:.2f}% | vol {ps['ann_vol_pct']:.2f}% | Sharpe {ps['sharpe']:.2f} | maxDD {ps['max_drawdown_pct']:.2f}%")
    print(f"WHOLE POOL (no 2nd selection): net PnL {all_d.sum():>10,.0f} | return {pa['return_pct']:+.2f}% | "
          f"ann. ret {pa['ann_return_pct']:.2f}% | vol {pa['ann_vol_pct']:.2f}% | Sharpe {pa['sharpe']:.2f} | maxDD {pa['max_drawdown_pct']:.2f}%")
    print(f"Hit rate (pair-period net PnL > 0): selected {(sel_obs['next_net_pnl'] > 0).mean():.1%} | "
          f"not selected {(uns_obs['next_net_pnl'] > 0).mean():.1%}")
    print("Rank correlation with NEXT-period net PnL (all pair-periods; ~0 => no predictive power):")
    for col, label in [("screen_score", "pool score (what selects the pool)"),
                       ("screen_tstat", "EG t-stat (more negative = stronger)"),
                       ("screen_half_life", "half-life (days)"),
                       ("screen_crossings_py", "mean crossings per year"),
                       ("screen_net_edge_bps_year", "estimated net edge, bps/yr"),
                       ("screen_edge_ratio", "gross / cost edge ratio"),
                       ("screen_stability", "sub-window stability"),
                       ("formation_net_pnl", "formation-window PnL (the 6-month ranking)")]:
        print(f"   {label:<46s}{rank_corr(obs_df[col], obs_df['next_net_pnl']):+.3f}")
    gsum = sel_obs["next_gross_pnl"].sum()
    print(f"Total costs on selected: {sel_obs['next_costs'].sum():,.0f} "
          f"({100 * sel_obs['next_costs'].sum() / abs(gsum) if gsum else np.nan:.1f}% of |gross|)")
    print("\nPer period:\n" + per_df[["period", "screen_id", "window_start", "window_end", "sel_net_pnl", "sel_return_pct",
                                      "all_return_pct", "sel_hit_rate", "formation_vs_next_rank_corr"]].round(3).to_string(index=False))
    print("\nNext: python baseline_random_selection.py " + args.outdir + " --sims 5000   (random-selection test)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))
        ax[0].plot(sel_r.index, 100 * sel_r.cumsum(), lw=2, label=f"Rolling top-{K}")
        ax[0].plot(all_r.index, 100 * all_r.cumsum(), lw=1.5, ls="--", label="Whole pool (equal weight)")
        for _, row in per_df.iterrows():
            ax[0].axvline(pd.Timestamp(row["window_start"]), color="grey", lw=.5, ls=":")
        for sid, g in per_df.groupby("screen_id"):
            ax[0].axvline(pd.Timestamp(g["window_start"].iloc[0]), color="tab:red", lw=1, ls="--", alpha=.6)
        ax[0].axhline(0, color="k", lw=.5); ax[0].set_ylabel("cumulative net return on capital (%)")
        ax[0].set_title("Cumulative net return (grey = rebalance, red = re-screen)"); ax[0].legend(); ax[0].grid(alpha=.3)
        x = np.arange(len(per_df)); w = .38
        ax[1].bar(x - w / 2, per_df["sel_return_pct"], w, label=f"Selected top-{K}")
        ax[1].bar(x + w / 2, per_df["all_return_pct"], w, label="Whole pool")
        ax[1].set_xticks(x); ax[1].set_xticklabels([str(s) for s in per_df["window_start"]], rotation=30)
        ax[1].set_title("Return by period (%)"); ax[1].legend(); ax[1].grid(alpha=.3)
        ax[2].scatter(uns_obs["formation_net_pnl"], uns_obs["next_net_pnl"], s=12, alpha=.5, label="not selected")
        ax[2].scatter(sel_obs["formation_net_pnl"], sel_obs["next_net_pnl"], s=14, alpha=.8, color="tab:red", label="selected")
        ax[2].axhline(0, color="k", lw=.5); ax[2].axvline(0, color="k", lw=.5)
        ax[2].set_xlabel("formation-window net PnL"); ax[2].set_ylabel("next-window net PnL")
        ax[2].set_title(f"Does past performance persist? (rank corr {pooled_corr:+.2f})"); ax[2].legend(); ax[2].grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/rescreen_chart.png", dpi=130)
        print(f"Chart -> {args.outdir}/rescreen_chart.png")
    except ImportError:
        print("matplotlib not installed - skipped chart")
    print(f"Files in {args.outdir}/: period_summary.csv, selection_log.csv, screen_log.csv, pair_summary_selected.csv, "
          f"daily_portfolio_pnl.csv, daily_pair_net_pnl.csv, trades_selected.csv")


if __name__ == "__main__":
    main()
