"""Phase 6 Step 5 — one-at-a-time sensitivity.

Two classes of parameter, handled differently:

**Linear in the objective** — battery throughput cost `c_b` and start cost `c_s`
enter total cost as `c_b · throughput` and `c_s · starts`, and neither changes
what the MILP *decides* unless it changes the optimum. Their sensitivity is
therefore available in closed form from stored dispatches, with no re-run. The
closed form is verified against one full re-run before being relied on.

**Decision-changing** — fuel curve `A` and `B`, minimum load ratio, minimum
up/down times, and the demand-model parameters all change the optimal dispatch,
so each level needs a real run.

Full-year rolling MILP is ~25 min per setting, so decision-changing parameters
are evaluated on a **stratified 3-month window** (February, June, October — one
per season), validated against the full year at the nominal setting so the offset
is known and reported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import controllers as ct
from . import cost_model as cm
from . import direct_forecast as D
from . import dispatch_core as dc
from . import heuristic as H
from . import milp_dispatch as mp
from . import paths as P
from .run_ladder import manifest
from .walkforward import soc_at_year_start

TEST_YEAR = 2025
WINDOW_MONTHS = (2, 6, 10)          # one per season: harmattan, rainy, rainy-late
# V4.2: the side studies use the walk-forward grid and its year-1 selection rule.
from .walkforward import THETAS, PSIS  # noqa: E402


def out_dir():
    d = P.results_subdir("sensitivity")
    d.mkdir(parents=True, exist_ok=True)
    return d


def window(full=False, pred=None, scale_pv=1.0):
    """Actuals on the evaluation window, and a direct trajectory for each hour.

    Forecasts are looked up by the hour's index in the full series. Each plan
    is cut at the end of its contiguous month block, so a February decision
    cannot see June. ``scale_pv`` scales the PV forecast the way the sizing
    sweep scales PV output.
    """
    if pred is None:
        frame = D.observed_frame()
        pred_pv, pred_ld = D.ensure_direct(
            frame, TEST_YEAR, D.cache_path(TEST_YEAR), seed=42)
    else:
        frame, pred_pv, pred_ld = pred
    idx = D.year_index(frame, TEST_YEAR)
    yr = frame.iloc[idx].reset_index(drop=True)
    if full:
        keep = np.ones(len(yr), dtype=bool)
    else:
        keep = yr.datetime.dt.month.isin(WINDOW_MONTHS).to_numpy()
    abs_of_local = idx[keep]
    block_end = D.contiguous_block_ends(abs_of_local)
    traj = D.trajectory_fn(pred_pv, pred_ld, abs_of_local, block_end,
                           scale_pv=scale_pv)
    soc0 = soc_at_year_start(TEST_YEAR)
    return (P.smoke_trim(yr.PV_power_kW.to_numpy(dtype=float)[keep]),
            P.smoke_trim(yr.total_load_kW.to_numpy(dtype=float)[keep]), soc0, traj)


# ------------------------------------------------------- linear parameters
def linear_sensitivity(dispatch_dir, cost=None, c_b_levels=(0.05, 0.632, 1.00, 2.00),
                       c_s_levels=(0, 33.75, 50.625, 75, 150, 300, 600)):
    """Closed-form sweep of the two linear cost parameters.

    total(c_b, c_s) = fixed + c_b · throughput + c_s · starts
    where `fixed` is everything else, recovered from the stored evaluation.
    """
    cost = cost or cm.CORRECTED
    cb0 = cost.battery_ghs_per_kwh_throughput
    cs0 = cost.genset.start_cost_ghs
    comps = {}
    for c in ("baseline", "heuristic", "oracle", "roll24"):
        j = json.loads((Path(dispatch_dir) / f"{c}.json").read_text())
        thr = j["battery_cost_GHS"] / cb0
        starts = j["genset_starts"]
        fixed = j["total_cost_GHS"] - cb0 * thr - cs0 * starts
        comps[c] = dict(fixed=fixed, thr=thr, starts=starts)

    rows = []
    for cb in c_b_levels:
        tot = {k: v["fixed"] + cb * v["thr"] + cs0 * v["starts"]
               for k, v in comps.items()}
        rows.append(dict(parameter="battery throughput cost c_b", value=cb,
                         nominal=cb0, **_ratios(tot)))
    for cs in c_s_levels:
        tot = {k: v["fixed"] + cb0 * v["thr"] + cs * v["starts"]
               for k, v in comps.items()}
        rows.append(dict(parameter="start cost c_s", value=cs, nominal=cs0,
                         **_ratios(tot)))
    return pd.DataFrame(rows)


def _ratios(tot):
    return dict(
        milp_vs_baseline_pct=(1 - tot["roll24"] / tot["baseline"]) * 100,
        milp_vs_heuristic_pct=(1 - tot["roll24"] / tot["heuristic"]) * 100,
        heur_vs_baseline_pct=(1 - tot["heuristic"] / tot["baseline"]) * 100,
        milp_premium_vs_oracle_pct=(tot["roll24"] / tot["oracle"] - 1) * 100)


# -------------------------------------------------- decision-changing runs
def tune_series(frame=None, scale_pv=1.0, soc_prev=None):
    """Year-1 (TEST_YEAR - 1, full year) actuals for the heuristic selection,
    exactly as walkforward.select_heuristic uses them."""
    from .walkforward import full_series, soc_at_year_start, soc_at_year_start_of
    d = full_series() if frame is None else frame.reset_index(drop=True)
    prev = d[d.datetime.dt.year == TEST_YEAR - 1]
    if soc_prev is None:
        soc_prev = (soc_at_year_start(TEST_YEAR - 1) if frame is None
                    else soc_at_year_start_of(d, TEST_YEAR - 1))
    return (prev.PV_power_kW.to_numpy(dtype=float) * scale_pv,
            prev.total_load_kW.to_numpy(dtype=float), float(soc_prev))


def evaluate(cost, PVa, LDa, soc0, *, trajectory_at, tune_on, label=""):
    """baseline / heuristic (selected on year-1) / ceiling / rolling MILP.

    ``tune_on`` = (pv_prev, load_prev, soc_prev) from ``tune_series``. The
    heuristic is never tuned on the evaluation window itself.
    """
    from .walkforward import select_heuristic_on
    t0 = time.time()
    binfo = {}
    base = dc.baseline(PVa, LDa, soc_init=soc0, genset=cost.genset, info=binfo)
    theta, psi = select_heuristic_on(*tune_on, cost)
    heur, _ = H.commitment_heuristic(PVa, LDa, soc_init=soc0, threshold=theta,
                                     bank_to_soc=psi, genset=cost.genset)
    orc, _ = mp.solve_year_blocked(PVa, LDa, soc0, cost, block_h=744,
                                   overlap_h=48, time_limit=300.0, mip_gap=1e-3,
                                   verbose=False)
    nan = np.full(len(PVa), np.nan)
    milp, _ = ct.run_rolling_milp(PVa, LDa, nan, nan, soc_init=soc0, horizon=24,
                                  commit=1, cost=cost, hierarchical=True,
                                  time_limit=10.0, mip_gap=1e-3,
                                  trajectory_at=trajectory_at)
    tot = {k: cost.evaluate(v)["total_cost_GHS"] for k, v in
           (("baseline", base), ("heuristic", heur), ("oracle", orc),
            ("roll24", milp))}
    rec = dict(label=label, theta=theta, psi=psi,
               heuristic_selected_on=f"{TEST_YEAR - 1} full year (walk-forward rule)",
               start_cost_ghs=cost.genset.start_cost_ghs,
               seconds=round(time.time() - t0, 1),
               **{f"cost_{k}": v for k, v in tot.items()}, **_ratios(tot),
               milp_run_h=cost.evaluate(milp)["genset_run_hours"],
               milp_LF_pct=cost.evaluate(milp)["mean_load_factor_pct"],
               milp_LPSP_pct=milp.metrics()["LPSP_pct"],
               uc_parity=dc.uc_parity_default(),
               baseline_min_down_overrides=binfo["min_down_overrides"])
    for k in ("baseline", "heuristic"):
        # A causal rule beating the perfect-foresight bound means the two are
        # not playing by the same rules: flag it, never crash.
        rec[f"{k}_below_oracle"] = bool(tot[k] < tot["oracle"] - 1e-6)
        if rec[f"{k}_below_oracle"]:
            print(f"  WARNING {label}: {k} {tot[k]:.0f} < oracle "
                  f"{tot['oracle']:.0f} GHS", flush=True)
    print(f"  {label:<34} MILP v heur {rec['milp_vs_heuristic_pct']:+7.2f}%  "
          f"v base {rec['milp_vs_baseline_pct']:+7.2f}%  [{rec['seconds']:.0f}s]",
          flush=True)
    return rec


def genset_variants():
    """Low / nominal / high for each decision-changing genset parameter."""
    base = cm.CORRECTED.genset
    out = {}
    for a in (0.0300, 0.0454, 0.0613):
        out[f"fuel A = {a:.4f}"] = cm.corrected("sourced_mean",
                                               a_l_per_kw_rated_h=a)
    for b in (0.270, 0.3005, 0.330):
        out[f"fuel B = {b:.4f}"] = cm.corrected("sourced_mean", b_l_per_kwh=b)
    for r in (0.20, 0.30, 0.40):
        out[f"min load ratio = {r:.2f}"] = cm.corrected("sourced_mean",
                                                        min_load_ratio=r)
    for u in (1, 2, 4):
        out[f"min up/down = {u} h"] = cm.corrected("sourced_mean", min_up_h=u,
                                                   min_down_h=u)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["linear", "validate-window", "genset"])
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)

    if a.stage == "linear":
        df = linear_sensitivity(P.results_subdir("phase5_sourced_mean"))
        df.to_csv(out_dir() / "linear_sensitivity.csv", index=False)
        print(df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        return df

    PVa, LDa, soc0, traj = window(full=a.full)
    tag = "full year" if a.full else f"{len(PVa)} h window"
    print(f"evaluation window: {tag}\n")

    if a.stage == "validate-window":
        rec = evaluate(cm.CORRECTED, PVa, LDa, soc0, trajectory_at=traj,
                       tune_on=tune_series(), label=f"nominal ({tag})")
        p = out_dir() / f"window_check_{'full' if a.full else 'sub'}.json"
        p.write_text(json.dumps(rec, indent=2, default=float))
        return rec

    rows = []
    prev = tune_series()
    items = P.smoke_groups(genset_variants().items(), key=lambda it: it[0].split("=")[0])
    for label, cost in items:
        rows.append(evaluate(cost, PVa, LDa, soc0, trajectory_at=traj,
                             tune_on=prev, label=label))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / "genset_sensitivity.csv", index=False)
    (out_dir() / "sensitivity_manifest.json").write_text(
        json.dumps(manifest({"window_months": list(WINDOW_MONTHS),
                             "start_cost_ghs": cm.CORRECTED.genset.start_cost_ghs,
                             "hours": int(len(PVa))}), indent=2))
    return df


if __name__ == "__main__":
    main()
