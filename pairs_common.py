#!/usr/bin/env python3
"""
Shared building blocks for the pairs-trading research scripts.

Previously these functions were copy-pasted into the rolling backtests and the panel
study. They now live here once, so a fix is made in one place.

Contents
--------
Data loading
    load_panels          yfinance 3-row-header CSV -> {"open","close","adv","sig"} panels

Pair simulation (full cost model: commission, half-spread, sqrt impact, sell fee, borrow)
    live_params          per-day walk-forward hedge ratio / spread mean / spread std (no look-ahead)
    simulate_pair        trade one pair over [start, end) -> (daily PnL frame, trade list)

Performance helpers
    perf_stats           net PnL, return, Sharpe, max drawdown
    rank_corr            Spearman rank correlation that tolerates NaNs

Cointegration screening of a universe on a trailing window
    half_life, zero_crossings         spread statistics
    pair_indices                      all pairs, or only same-group pairs
    _init_screen, _screen_pair        multiprocessing worker pair (module-level so they pickle)
    screen_universe                   Engle-Granger (+ optional Johansen / KPSS) with BH-FDR

Used by:
    backtest_rolling_rescreen.py, backtest_rolling_fixed_pool.py (legacy),
    study_metric_predictiveness.py
"""
import warnings
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_panels(path, max_missing=0.05):
    """Same cleaning as the screener so row indices / train split line up."""
    df = pd.read_csv(path, header=[0, 1], index_col=0, skiprows=[2])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    def panel(name):
        return df.xs(name, axis=1, level=1).apply(pd.to_numeric, errors="coerce")

    close = panel("Close")
    close = close.loc[:, close.isna().mean() <= max_missing]
    close = close.ffill(limit=3).dropna(how="any")
    close = close.loc[:, (close > 0).all()]

    opn = panel("Open").reindex(index=close.index, columns=close.columns).ffill(limit=3)
    opn = opn.where(opn > 0).fillna(close)
    vol = panel("Volume").reindex(index=close.index, columns=close.columns).fillna(0)

    sig = np.log(close).diff().rolling(20).std().shift(1)
    sig = sig.fillna(sig.median())
    adv = vol.rolling(20).mean().shift(1)
    return {"open": opn, "close": close, "adv": adv, "sig": sig}


# ----------------------------------------------------------------------------
# Single-pair simulation
# ----------------------------------------------------------------------------
def live_params(ly, lx, start, end, step, lookback):
    """Per-day (alpha, beta, mu, sd) for days [start, end); each fit uses only data BEFORE its block."""
    n = len(ly)
    a, b, mu, sd = (np.full(n, np.nan) for _ in range(4))
    s = start
    while s < end:
        e = min(s + step, end) if step > 0 else end
        lo = max(0, s - lookback)
        b0, a0 = np.polyfit(lx[lo:s], ly[lo:s], 1)
        sp = ly[lo:s] - (a0 + b0 * lx[lo:s])
        a[s:e], b[s:e] = a0, b0
        mu[s:e], sd[s:e] = sp.mean(), sp.std(ddof=1)
        s = e
    return a, b, mu, sd


def simulate_pair(ty, tx, D, start, end, args):
    dates = D["close"].index
    n = len(dates)
    g = lambda k, t: D[k][t].values
    oy, cy, ox, cx = g("open", ty), g("close", ty), g("open", tx), g("close", tx)
    ady, sgy, adx, sgx = g("adv", ty), g("sig", ty), g("adv", tx), g("sig", tx)
    ly, lx = np.log(cy), np.log(cx)

    lookback = args.wf_lookback if args.wf_lookback > 0 else args.train_len
    a, b, mu, sd = live_params(ly, lx, start, end, args.wf_step, lookback)

    cols = ["gross", "commission", "spread", "impact", "fees", "borrow", "financing"]
    A = {c: np.zeros(n) for c in cols}
    in_pos = np.zeros(n, dtype=bool)
    trades = []
    S = dict(pos=0, sy=0, sx=0, ry=0.0, rx=0.0, tr=None, hold=0)
    pending = None
    exec_open = args.exec == "open"

    def leg_costs(shares, px, adv, sig, opening):
        sell = (shares < 0) if opening else (shares > 0)
        q = abs(shares)
        notional = q * px
        comm = max(args.min_comm, args.comm_ps * q)
        spread = notional * args.half_spread_bps / 1e4
        part = 1.0 if not (np.isfinite(adv) and adv > 0) else min(1.0, q / adv)
        impact = notional * args.impact_coef * sig * np.sqrt(part)
        fees = notional * args.sell_fee_bps / 1e4 if sell else 0.0
        return comm, spread, impact, fees

    def book_costs(t, fy, fx, opening):
        tot = 0.0
        for sh, px, adv, sig in ((S["sy"], fy, ady[t], sgy[t]), (S["sx"], fx, adx[t], sgx[t])):
            c, s_, i_, f_ = leg_costs(sh, px, adv, sig, opening)
            A["commission"][t] += c; A["spread"][t] += s_; A["impact"][t] += i_; A["fees"][t] += f_
            tot += c + s_ + i_ + f_
        S["tr"]["costs"] += tot

    def close_position(t, fy, fx, reason, z_exit, add_gross):
        if add_gross:
            gp = S["sy"] * (fy - cy[t - 1]) + S["sx"] * (fx - cx[t - 1])
            A["gross"][t] += gp
            S["tr"]["gross"] += gp
        book_costs(t, fy, fx, opening=False)
        tr = S["tr"]
        tr.update(exit_date=dates[t], exit_reason=reason, exit_z=z_exit, days_held=S["hold"],
                  net=tr["gross"] - tr["costs"])
        trades.append(tr)
        S.update(pos=0, sy=0, sx=0, tr=None, hold=0)

    for t in range(start, end):
        last = t == end - 1
        entered_today = False

        if pending is not None:  # 1) execute yesterday's order
            fy, fx = (oy[t], ox[t]) if exec_open else (cy[t], cx[t])
            if pending[0] == "exit" and S["pos"] != 0:
                close_position(t, fy, fx, pending[1], pending[2], add_gross=True)
            elif pending[0] == "enter" and S["pos"] == 0:
                _, side, beta, z_in = pending
                ny = args.capital * args.gross_leverage / (1 + abs(beta))
                sy = int(np.round(side * ny / fy))
                sx = int(np.round(-side * beta * ny / fx))
                if sy != 0 and sx != 0:
                    S.update(pos=side, sy=sy, sx=sx, ry=fy, rx=fx, hold=0,
                             tr=dict(y=ty, x=tx, entry_date=dates[t],
                                     side="long_spread" if side > 0 else "short_spread",
                                     entry_z=z_in, beta=beta, shares_y=sy, shares_x=sx,
                                     entry_notional=abs(sy) * fy + abs(sx) * fx, gross=0.0, costs=0.0))
                    book_costs(t, fy, fx, opening=True)
                    entered_today = True
            pending = None

        if S["pos"] != 0:  # 2) mark to close; borrow / financing for overnight hold
            gp = S["sy"] * (cy[t] - S["ry"]) + S["sx"] * (cx[t] - S["rx"])
            A["gross"][t] += gp
            S["tr"]["gross"] += gp
            if not entered_today:
                days = (dates[t] - dates[t - 1]).days
                legs = ((S["sy"], cy[t - 1]), (S["sx"], cx[t - 1]))
                short_v = sum(abs(s) * p for s, p in legs if s < 0)
                long_v = sum(abs(s) * p for s, p in legs if s > 0)
                bc = short_v * args.borrow_bps / 1e4 * days / 365
                fc = long_v * args.financing_bps / 1e4 * days / 365
                A["borrow"][t] += bc; A["financing"][t] += fc
                S["tr"]["costs"] += bc + fc
            S["ry"], S["rx"] = cy[t], cx[t]
            S["hold"] += 1

        # 3) signal at today's close -> order for tomorrow; liquidate on the window's last day
        z = (ly[t] - a[t] - b[t] * lx[t] - mu[t]) / sd[t] if sd[t] > 0 else np.nan
        if last:
            if S["pos"] != 0:
                close_position(t, cy[t], cx[t], "end_of_window", z, add_gross=False)
        elif np.isfinite(z):
            if S["pos"] == 0:
                if t < end - 3 and args.entry_z < abs(z) < args.stop_z:
                    pending = ("enter", -1 if z > 0 else 1, b[t], z)
            else:
                if abs(z) > args.stop_z:
                    pending = ("exit", "stop", z)
                elif abs(z) < args.exit_z:
                    pending = ("exit", "target", z)
                elif S["hold"] >= args.max_hold:
                    pending = ("exit", "time", z)
        in_pos[t] = S["pos"] != 0

    daily = pd.DataFrame(A, index=dates)
    daily["in_pos"] = in_pos
    daily = daily.iloc[start:end].copy()
    daily["costs"] = daily[["commission", "spread", "impact", "fees", "borrow", "financing"]].sum(axis=1)
    daily["net"] = daily["gross"] - daily["costs"]
    return daily, trades


# ----------------------------------------------------------------------------
# Performance helpers
# ----------------------------------------------------------------------------
def perf_stats(net, capital):
    r = net / capital
    eq = capital + net.cumsum()
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    sd = r.std()
    return dict(net_pnl=net.sum(), return_pct=100 * net.sum() / capital,
                ann_return_pct=100 * r.mean() * 252, ann_vol_pct=100 * sd * np.sqrt(252),
                sharpe=r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
                max_drawdown_pct=100 * dd)


def rank_corr(a, b):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return np.nan
    return a[m].rank().corr(b[m].rank())


# ----------------------------------------------------------------------------
# Cointegration screening of the whole universe on a trailing window
# ----------------------------------------------------------------------------
_G = {}

SCREEN_COLS = ["y", "x", "group", "eg_tstat", "eg_pvalue", "eg_pvalue_fdr", "half_life", "hedge_ratio", "ret_corr",
               "crossings_per_year", "entries_per_year", "gross_bps_per_trade", "cost_bps_per_trade",
               "net_bps_per_trade", "edge_ratio", "net_edge_bps_per_year", "stability", "beta_cv",
               "johansen_ok", "johansen_trace_ratio", "kpss_pvalue"]


def _init_screen(W, P):
    _G["W"], _G["P"] = W, P


def half_life(s):
    s = np.asarray(s)
    k = np.polyfit(s[:-1], np.diff(s), 1)[0]
    phi = 1.0 + k
    return -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan


def zero_crossings(z):
    z = np.asarray(z)
    return int(np.sum(np.signbit(z[1:]) != np.signbit(z[:-1])))


def _screen_pair(ij):
    """Existence tests + tradability metrics for one pair on the screening window."""
    i, j = ij
    W, P = _G["W"], _G["P"]
    y, x = W[:, i], W[:, j]
    n = len(y)
    corr = np.corrcoef(np.diff(y), np.diff(x))[0, 1]
    if not np.isfinite(corr) or corr < P["min_corr"]:
        return None
    try:
        t, p, _ = coint(y, x, autolag="aic")                     # Engle-Granger existence test
    except Exception:
        return None
    b, a = np.polyfit(x, y, 1)
    s = y - (a + b * x)
    sd = s.std(ddof=1)
    if not sd > 0:
        return None
    z = (s - s.mean()) / sd
    hl = half_life(s)

    # ---- tradability: how often does it trade and is each trade worth more than it costs? ----
    ez, xz = P["entry_z"], P["exit_z"]
    az = np.abs(z)
    entries_py = float(np.sum((az[1:] > ez) & (az[:-1] <= ez))) * 252.0 / n
    cross_py = zero_crossings(z) * 252.0 / n
    wy, wx = 1.0 / (1 + abs(b)), abs(b) / (1 + abs(b))          # leg weights in gross notional
    gross_bps = 1e4 * (ez - xz) * sd / (1 + abs(b))             # profit if z goes entry -> exit
    py, px = np.exp(y[-1]), np.exp(x[-1])
    cost_bps = (2 * P["half_spread_bps"]                        # 4 fills = 2x gross traded
                + P["sell_fee_bps"]                             # sells ~ 1x gross
                + 2e4 * P["comm_ps"] * (wy / py + wx / px)      # commission
                + (P["borrow_bps"] * hl / 252.0 if np.isfinite(hl) else np.nan))   # ~2*HL hold, half short
    net_trade = gross_bps - cost_bps
    # (market impact is NOT in this estimate; the full simulation includes it)

    # ---- stability across sub-windows (full-window hedge ratio) ----
    k = P["n_sub"]
    edges = np.linspace(0, n, k + 1).astype(int)
    ok, betas = [], []
    for c in range(k):
        sl = slice(edges[c], edges[c + 1])
        hc = half_life(s[sl])
        ok.append(bool(np.isfinite(hc) and P["min_hl"] <= hc <= P["max_hl"]))
        betas.append(np.polyfit(x[sl], y[sl], 1)[0])
    stability = float(np.mean(ok))
    beta_cv = float(np.std(betas) / abs(b)) if b != 0 else np.nan

    joh_ok, joh_ratio = np.nan, np.nan
    if P["want_johansen"]:
        try:
            r = coint_johansen(np.column_stack([y, x]), 0, 1)
            joh_ratio = float(r.lr1[0] / r.cvt[0, 1])
            joh_ok = bool(r.lr1[0] > r.cvt[0, 1] and r.lr2[0] > r.cvm[0, 1])
        except Exception:
            joh_ok = False
    kpss_p = np.nan
    if P["want_kpss"]:
        try:
            kpss_p = float(kpss(s, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = np.nan

    return dict(i=i, j=j, eg_tstat=t, eg_pvalue=p, half_life=hl, hedge_ratio=b, ret_corr=corr,
                crossings_per_year=cross_py, entries_per_year=entries_py, gross_bps_per_trade=gross_bps,
                cost_bps_per_trade=cost_bps, net_bps_per_trade=net_trade,
                edge_ratio=gross_bps / cost_bps if cost_bps and cost_bps > 0 else np.nan,
                net_edge_bps_per_year=entries_py * net_trade if np.isfinite(net_trade) else np.nan,
                stability=stability, beta_cv=beta_cv, johansen_ok=joh_ok,
                johansen_trace_ratio=joh_ratio, kpss_pvalue=kpss_p)


def pair_indices(cols, groups):
    """Pairs to test: every pair, or only pairs inside the same group."""
    if groups is None:
        return list(combinations(range(len(cols)), 2)), None
    gmap = {c: groups[c] for c in cols if c in groups}
    by = {}
    for i, c in enumerate(cols):
        if c in gmap:
            by.setdefault(gmap[c], []).append(i)
    return [pr for ii in by.values() for pr in combinations(ii, 2)], gmap


def screen_universe(logp, lo, hi, args, groups=None):
    """Existence tests + tradability metrics for the candidate pairs on log prices in rows [lo, hi)."""
    cols = list(logp.columns)
    W = logp.iloc[lo:hi].values
    idx, gmap = pair_indices(cols, groups)
    P = dict(entry_z=args.entry_z, exit_z=args.exit_z, half_spread_bps=args.half_spread_bps,
             sell_fee_bps=args.sell_fee_bps, comm_ps=args.comm_ps, borrow_bps=args.borrow_bps,
             n_sub=args.n_subwindows, min_hl=args.min_hl, max_hl=args.max_hl, min_corr=args.min_corr,
             want_johansen=args.gate_johansen, want_kpss=args.gate_kpss)
    with Pool(args.jobs, initializer=_init_screen, initargs=(W, P)) as pool:
        rows = [r for r in pool.imap_unordered(_screen_pair, idx, chunksize=50) if r]
    if not rows:
        return pd.DataFrame(columns=SCREEN_COLS)
    df = pd.DataFrame(rows)
    df["y"] = [cols[i] for i in df["i"]]
    df["x"] = [cols[j] for j in df["j"]]
    df["group"] = [gmap[cols[i]] for i in df["i"]] if gmap else "all"
    df["eg_pvalue_fdr"] = multipletests(df["eg_pvalue"].fillna(1), method="fdr_bh")[1]   # family = pairs tested
    return df.drop(columns=["i", "j"])
