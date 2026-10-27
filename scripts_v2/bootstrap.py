"""Phase 6 Step 5c — block bootstrap confidence intervals.

Weekly blocks, resampled with replacement, so diurnal and synoptic structure are
preserved inside each block. No dispatch is re-run: every controller's hourly
`soc / charge / discharge / diesel / unmet / curtail` is persisted, which is
exactly what a cost re-evaluation needs.

Genset *starts* need care. Counting them within each resampled block adds a
start wherever a block boundary falls mid-run, and that bias does **not** cancel
in the ratios: it adds roughly the same absolute number of starts to every
controller, which raises the cheaper controller's relative saving. Measured, it
moved MILP-vs-baseline from a 14.34% point estimate to a 16.21% bootstrap mean.

The fix is to mark starts **once, on the continuous series**, and then sum that
per-hour indicator over whatever hours a resample selects. A start is then
attributed to the hour it actually happened in, and block boundaries introduce
none of their own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cost_model as cm
from . import paths as P
from .run_ladder import manifest

CONTROLLERS = ("baseline", "heuristic", "oracle", "roll24")


def load_dispatch(npz_path, controllers=CONTROLLERS):
    """Hourly series per controller, from a walk-forward fold or a phase-5 dir."""
    z = np.load(npz_path)
    keys = set(z.files)
    out = {}
    for c in controllers:
        if f"{c}_diesel" in keys:                       # fold_*.npz layout
            out[c] = {f: z[f"{c}_{f}"] for f in
                      ("diesel", "charge", "discharge", "unmet", "curtail")}
            out[c]["is_start"] = mark_starts(out[c]["diesel"])
    return out


def load_dispatch_dir(d, controllers=CONTROLLERS):
    """Phase-5 style: one npz per controller."""
    out = {}
    for c in controllers:
        p = Path(d) / f"{c}_dispatch.npz"
        if p.exists():
            z = np.load(p)
            out[c] = {f: z[f] for f in
                      ("diesel", "charge", "discharge", "unmet", "curtail")}
            out[c]["is_start"] = mark_starts(out[c]["diesel"])
    return out


def mark_starts(diesel):
    """Per-hour start indicator, computed once on the continuous series."""
    run = diesel > 1e-9
    st = np.zeros(len(run), dtype=float)
    st[0] = float(run[0])
    st[1:] = (run[1:] & ~run[:-1]).astype(float)
    return st


def block_cost(series, idx, cost):
    """Total cost of the hours selected by `idx` (a list of block slices)."""
    g = np.concatenate([series["diesel"][s] for s in idx])
    ch = np.concatenate([series["charge"][s] for s in idx])
    di = np.concatenate([series["discharge"][s] for s in idx])
    un = np.concatenate([series["unmet"][s] for s in idx])
    cu = np.concatenate([series["curtail"][s] for s in idx])

    run = g > 1e-9
    litres = (cost.genset.no_load_l_per_h * run.sum()
              + cost.genset.b_l_per_kwh * g.sum())
    fuel = litres * cost.genset.fuel_price_ghs_per_l
    # Starts come from the pre-marked continuous series, so a block boundary
    # falling mid-run does not invent one.
    starts = float(np.concatenate([series["is_start"][s] for s in idx]).sum())
    thru = (ch.sum() + di.sum()) / 2.0
    return (fuel + starts * cost.genset.start_cost_ghs
            + thru * cost.battery_ghs_per_kwh_throughput
            + un.sum() * cost.w_unmet + cu.sum() * cost.w_curtail)


def bootstrap(dispatch, cost=None, n_boot=2000, block_h=168, seed=0):
    cost = cost or cm.CORRECTED
    n = len(next(iter(dispatch.values()))["diesel"])
    starts = list(range(0, n - block_h + 1, block_h))
    n_blocks = len(starts)
    rng = np.random.default_rng(seed)

    stats = {k: [] for k in ("milp_vs_baseline_pct", "milp_vs_heuristic_pct",
                             "heur_vs_baseline_pct", "milp_premium_vs_oracle_pct")}
    for _ in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        idx = [slice(starts[i], starts[i] + block_h) for i in pick]
        tot = {c: block_cost(s, idx, cost) for c, s in dispatch.items()}
        stats["milp_vs_baseline_pct"].append(
            (1 - tot["roll24"] / tot["baseline"]) * 100)
        stats["milp_vs_heuristic_pct"].append(
            (1 - tot["roll24"] / tot["heuristic"]) * 100)
        stats["heur_vs_baseline_pct"].append(
            (1 - tot["heuristic"] / tot["baseline"]) * 100)
        stats["milp_premium_vs_oracle_pct"].append(
            (tot["roll24"] / tot["oracle"] - 1) * 100)

    rows = []
    for k, v in stats.items():
        a = np.asarray(v)
        rows.append(dict(metric=k, mean=a.mean(), std=a.std(ddof=1),
                         ci_lo_95=np.percentile(a, 2.5),
                         ci_hi_95=np.percentile(a, 97.5),
                         n_boot=n_boot, block_hours=block_h,
                         n_blocks=n_blocks))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fold",
                    choices=["fold", "phase5"])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--start-cost", type=float, default=None,
                    help="GHS per start. Default is the cost-model 150.")
    ap.add_argument("--fold-subdir", default="walkforward")
    ap.add_argument("--out-name", default="bootstrap_ci.csv")
    a = ap.parse_args(argv)

    if a.source == "fold":
        p = P.results_subdir(a.fold_subdir) / f"fold_{a.year}_dispatch.npz"
        disp = load_dispatch(p)
        label = f"walk-forward fold {a.year} ({a.fold_subdir})"
    else:
        disp = load_dispatch_dir(P.results_subdir("phase5_sourced_mean"))
        label = "phase 5, load_v2, sourced_mean"

    missing = [c for c in CONTROLLERS if c not in disp]
    if missing:
        raise SystemExit(f"missing controllers: {missing}")
    if a.start_cost is None:
        cost = cm.CORRECTED
    else:
        cost = cm.corrected("sourced_mean", start_cost_ghs=a.start_cost)
    print(f"source: {label}  |  hours: {len(disp['roll24']['diesel'])}  |  "
          f"start cost {cost.genset.start_cost_ghs:g} GHS  |  "
          f"persisted fields: {sorted(disp['roll24'])}")

    df = bootstrap(disp, cost=cost, n_boot=a.n_boot)
    out = P.results_subdir(a.out_name)
    df.to_csv(out, index=False)
    manifest_name = Path(a.out_name).stem + "_manifest.json"
    (P.results_subdir(manifest_name)).write_text(
        json.dumps(manifest({"source": label, "n_boot": a.n_boot,
                             "start_cost_ghs": cost.genset.start_cost_ghs,
                             "fold_subdir": a.fold_subdir}), indent=2))
    print("\n95% BLOCK-BOOTSTRAP INTERVALS (weekly blocks)")
    print(f"{'metric':<32}{'mean':>9}{'std':>8}{'95% CI':>22}")
    for _, r in df.iterrows():
        print(f"{r.metric:<32}{r['mean']:>8.2f}%{r['std']:>7.2f}"
              f"   [{r.ci_lo_95:>6.2f}%, {r.ci_hi_95:>6.2f}%]")
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    main()
