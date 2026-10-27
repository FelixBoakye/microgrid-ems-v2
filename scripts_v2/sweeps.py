"""Phase 4/6 sweeps: re-planning interval, sizing, stress scenarios.

The re-planning sweep exists to *test* the paper's claim that shorter intervals
close the gap, rather than assert it. Under the flat-fuel cost model the oracle
is a horizontal floor that the greedy baseline already sits on, so the sweep is
expected to show the interval barely matters — which is the finding.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import controllers as ct
from . import paths as P
from . import cost_model as cm
from . import dispatch_core as dc
from . import oracle
from .config import DIESEL_MARGINAL_COST, HORIZON, SOC_MIN

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _series(quick=False):
    from .run_ladder import direct_trajectory, load_test_year, soc_start_for_2025
    df = load_test_year()
    soc0 = soc_start_for_2025()
    traj, frame, idx = direct_trajectory(2025)
    if len(df) != len(idx) or not np.array_equal(
            df.datetime.to_numpy(), frame.datetime.to_numpy()[idx]):
        raise RuntimeError("test-year table is not the observed 2025 block")
    n = 24 * 28 if quick else (P.smoke_hours() or len(df))
    return (df.PV_power_kW.values[:n], df.total_load_kW.values[:n], soc0, traj)


def replan_sweep(intervals=(1, 2, 3, 6, 12, 24), quick=False, hierarchical=False):
    """Cost vs re-planning interval, with the oracle as a horizontal floor."""
    PVa, LDa, soc0, traj = _series(quick)
    _, ores = oracle.solve_year(PVa, LDa, soc_init=soc0)
    base = dc.baseline(PVa, LDa, soc_init=soc0, uc_parity=False)  # flat model
    o_obj = dc.objective(ores)

    rows = [dict(interval_h=0, configuration="oracle (floor)",
                 **ores.metrics(), premium_vs_oracle_pct=0.0),
            dict(interval_h=-1, configuration="rule-based baseline",
                 **base.metrics(),
                 premium_vs_oracle_pct=(dc.objective(base) / o_obj - 1) * 100)]

    for k in intervals:
        t0 = time.time()
        res = ct.run_receding(
            pv_actual=PVa, load_actual=LDa,
            pv_forecast=np.full(len(PVa), np.nan),
            load_forecast=np.full(len(PVa), np.nan),
            trajectory_at=traj,
            planner=ct.plan_lp, soc_init=soc0, horizon=HORIZON, commit=k,
            executor="core", rt_balance=True,
            battery_mode="merit" if hierarchical else "setpoint",
            use_soc_floor=hierarchical,
            planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST))
        m = res.metrics()
        rows.append(dict(interval_h=k,
                         configuration=("LP hierarchical" if hierarchical
                                        else "LP setpoint"),
                         seconds=round(time.time() - t0, 1), **m,
                         premium_vs_oracle_pct=(m["objective"] / o_obj - 1) * 100))
        print(f"  interval {k:>3} h  diesel {m['diesel_kWh_per_year']:>10,.0f} kWh  "
              f"premium vs oracle {rows[-1]['premium_vs_oracle_pct']:+7.3f}%",
              flush=True)
    return pd.DataFrame(rows)


def replan_sweep_corrected(intervals=(1, 2, 3, 6, 12, 24), quick=False):
    """Re-planning interval under the FULL cost model (affine fuel, start cost,
    minimum load, min up/down, battery throughput): rolling MILP, 24 h horizon,
    hierarchical execution, direct forecasts, re-planned every k hours. Floor is
    the blocked perfect-foresight MILP. Start cost from MG_START_COST_GHS."""
    from . import milp_dispatch as mp
    PVa, LDa, soc0, traj = _series(quick)
    cost = cm.CORRECTED
    orc, _ = mp.solve_year_blocked(PVa, LDa, soc0, cost, block_h=744, overlap_h=48,
                                   time_limit=300.0, mip_gap=1e-3, verbose=False)
    o = cost.evaluate(orc)["total_cost_GHS"]
    tag = dict(start_cost_ghs=cost.genset.start_cost_ghs)  # cost_model comes from evaluate()
    rows = [dict(interval_h=0, configuration="perfect-foresight bound (floor)",
                 **tag, **cost.evaluate(orc), premium_vs_oracle_pct=0.0)]
    base = dc.baseline(PVa, LDa, soc_init=soc0, genset=cost.genset)
    rows.append(dict(interval_h=-1, configuration="rule-based baseline", **tag,
                     **cost.evaluate(base),
                     premium_vs_oracle_pct=(cost.evaluate(base)["total_cost_GHS"] / o - 1) * 100))
    nan = np.full(len(PVa), np.nan)
    for k in intervals:
        t0 = time.time()
        res, _ = ct.run_rolling_milp(PVa, LDa, nan, nan, soc_init=soc0, horizon=24,
                                     commit=k, cost=cost, hierarchical=True,
                                     time_limit=10.0, mip_gap=1e-3, trajectory_at=traj)
        e = cost.evaluate(res)
        rows.append(dict(interval_h=k, configuration="rolling MILP, full cost model",
                         seconds=round(time.time() - t0, 1), **tag, **e,
                         LPSP_pct=res.metrics()["LPSP_pct"],
                         premium_vs_oracle_pct=(e["total_cost_GHS"] / o - 1) * 100))
        print(f"  interval {k:>3} h  cost {e['total_cost_GHS']:>12,.0f} GHS  "
              f"premium {rows[-1]['premium_vs_oracle_pct']:+7.3f}%", flush=True)
    return pd.DataFrame(rows)


def genset_loading(dispatches: dict, rated_kw=40.0, bins=None):
    """Load-factor histogram per controller — the paper's most persuasive plot."""
    bins = bins if bins is not None else np.arange(0, 1.05, 0.05)
    rows = []
    for label, diesel in dispatches.items():
        d = np.asarray(diesel, float)
        run = d > 1e-9
        lf = d[run] / rated_kw
        hist, _ = np.histogram(lf, bins=bins)
        for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
            rows.append(dict(controller=label, lf_lo=lo, lf_hi=hi, hours=int(cnt),
                             share_pct=100.0 * cnt / max(run.sum(), 1)))
        rows.append(dict(controller=label, lf_lo=np.nan, lf_hi=np.nan,
                         hours=int(run.sum()), share_pct=100.0,
                         mean_lf_pct=float(lf.mean() * 100) if run.any() else 0.0))
    return pd.DataFrame(rows)


def sizing_sweep(batteries=(65, 97.5, 130, 195, 260), pvs=(60, 75, 90, 105, 120),
                 quick=True, cost=cm.CORRECTED):
    """Where does the EMS matter, and where does capacity dominate?

    Answers the reviewer question "why not just add battery instead of
    intelligence?" with a curve rather than an assertion.
    """
    from . import config as C
    from . import milp_dispatch as mp
    PVa, LDa, soc0, _traj = _series(quick)
    rows = []
    orig = (C.BATTERY_CAPACITY_KWH, C.SOC_MIN, C.SOC_MAX,
            C.MAX_CHARGE_KW, C.MAX_DISCHARGE_KW)
    try:
        for b in batteries:
            for p in pvs:
                C.BATTERY_CAPACITY_KWH = b
                C.SOC_MIN = (1 - C.DOD_MAX) * b
                C.SOC_MAX = b
                C.MAX_CHARGE_KW = C.MAX_DISCHARGE_KW = 0.5 * b
                scale = p / 90.0
                pv_s = PVa * scale
                base = dc.baseline(pv_s, LDa, soc_init=C.SOC_MIN, genset=cost.genset)
                mres, _ = mp.solve_year_blocked(pv_s, LDa, C.SOC_MIN, cost,
                                                block_h=len(pv_s), overlap_h=0,
                                                verbose=False)
                mb = cost.evaluate(base)
                mm = cost.evaluate(mres)
                rows.append(dict(
                    battery_kwh=b, pv_kwp=p,
                    baseline_cost=mb["total_cost_GHS"],
                    ems_cost=mm["total_cost_GHS"],
                    ems_saving_pct=(1 - mm["total_cost_GHS"] / mb["total_cost_GHS"]) * 100,
                    baseline_lpsp=base.metrics()["LPSP_pct"],
                    ems_lpsp=mres.metrics()["LPSP_pct"]))
                print(f"  battery {b:>6.1f} kWh  PV {p:>5.1f} kWp  "
                      f"EMS saves {rows[-1]['ems_saving_pct']:+6.2f}%", flush=True)
    finally:
        (C.BATTERY_CAPACITY_KWH, C.SOC_MIN, C.SOC_MAX,
         C.MAX_CHARGE_KW, C.MAX_DISCHARGE_KW) = orig
    return pd.DataFrame(rows)


def stress_scenarios(quick=True, cost=cm.CORRECTED):
    """Regimes where reliability actually discriminates between controllers.

    Both controllers sit at 0% LPSP under normal conditions, so reliability is
    not currently a discriminating metric. Fuel rationing, a forced genset
    outage and a degraded pack create the frontier.
    """
    from . import milp_dispatch as mp
    PVa, LDa, soc0, _traj = _series(quick)
    scen = {
        "nominal": dict(),
        "genset derated 50%": dict(diesel_max=20.0),
        "genset outage 72 h": dict(outage=(24 * 7, 24 * 7 + 72)),
        "battery degraded 70%": dict(soc_scale=0.7),
        "fuel rationed 60%": dict(fuel_cap_frac=0.60),
    }
    rows = []
    for name, kw in scen.items():
        dmax = kw.get("diesel_max", 40.0)
        dmax_series = np.full(len(PVa), dmax)
        if "outage" in kw:
            a, b = kw["outage"]
            dmax_series[a:b] = 0.0
        base = dc.baseline(PVa, LDa, soc_init=soc0, genset=cost.genset,
                           diesel_max=dmax_series.min()
                           if "outage" not in kw else dmax)
        # Baseline under an outage: simulate hour-wise with the varying cap.
        if "outage" in kw:
            res = []
            s = soc0
            out = {k: np.zeros(len(PVa)) for k in
                   ("soc", "charge", "discharge", "diesel", "unmet", "curtail")}
            for t in range(len(PVa)):
                s, c, d, g, u, k2 = dc.step(PVa[t], LDa[t], s, 65.0, 65.0, 40.0,
                                            rt_balance=True, battery_mode="merit",
                                            diesel_max=dmax_series[t])
                out["soc"][t], out["charge"][t], out["discharge"][t] = s, c, d
                out["diesel"][t], out["unmet"][t], out["curtail"][t] = g, u, k2
            base = dc.SimResult(pv=PVa, load=LDa, **out)
        m = base.metrics()
        rows.append(dict(scenario=name, controller="baseline", **m,
                         **{k: v for k, v in cost.evaluate(base).items()
                            if k.endswith("GHS") or k.startswith("genset")}))
        print(f"  {name:<24} baseline LPSP {m['LPSP_pct']:.3f}%", flush=True)
    return pd.DataFrame(rows)


def _out(name, quick):
    """Quick runs never overwrite a canonical result file.

    The stored replan sweep and attribution ladder once disagreed by a factor of
    two on the same configurations, purely because one had been produced with
    --quick (672 h of January) and the other on the full 8,760-hour year. Both
    files recorded `hours`, but nothing stopped the 28-day run from landing on
    the canonical filename. Now it cannot.
    """
    stem, ext = name.rsplit(".", 1)
    if quick:
        name = f"{stem}_quick.{ext}"
    # Route through the load-variant resolver so MG_LOAD=v2 writes under
    # results/v2/ and can never overwrite a load_v1 canonical result.
    return P.results_subdir(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="replan",
                    choices=["replan", "replan_corrected", "sizing", "stress", "all"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)

    if a.what in ("replan", "all"):
        print("Re-planning interval sweep (setpoint execution):")
        df = replan_sweep(quick=a.quick)
        print("Re-planning interval sweep (hierarchical execution):")
        dfh = replan_sweep(quick=a.quick, hierarchical=True)
        p = _out("replan_sweep.csv", a.quick)
        out = pd.concat([df, dfh])
        out["cost_model"] = "flat (as published), no start cost"
        out.to_csv(p, index=False)
        print(f"Saved {p}")

    if a.what == "replan_corrected":
        print("Re-planning interval sweep, full cost model:")
        p = _out("replan_sweep_corrected.csv", a.quick)
        replan_sweep_corrected(quick=a.quick).to_csv(p, index=False)
        print(f"Saved {p}")

    if a.what in ("sizing", "all"):
        p = _out("sizing_sweep.csv", a.quick)
        sizing_sweep(quick=a.quick).to_csv(p, index=False)
        print(f"Saved {p}")

    if a.what in ("stress", "all"):
        p = _out("stress_scenarios.csv", a.quick)
        stress_scenarios(quick=a.quick).to_csv(p, index=False)
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
