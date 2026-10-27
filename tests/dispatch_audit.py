"""Helpers for test_dispatch_rules.py: rule checks on executed hourly dispatch.

Analysis only. Nothing here changes controller behaviour.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

RATED = 40.0
RHO_MIN = 0.30
T_UP = T_DN = 2
EPS = 1e-9
BASE = Path("/workspace/dispatch_fix_2026-10-18")
LOCKED = BASE / "locked"        # Build's locked 2025 dispatches (511 starts)
FOLDS = BASE / "folds"          # box walk-forward folds 2022-2025 (33.75 GHS, output only)


def blocks(on):
    """(start_index, length) of every run of True."""
    on = np.asarray(on, bool)
    d = np.diff(np.r_[0, on.astype(int), 0])
    s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1)
    return list(zip(s, e - s))


def short_on_blocks(on, t_up=T_UP):
    """On-blocks shorter than t_up. Blocks cut by the start or end of the series
    are excluded (their true length is unknown)."""
    n = len(on)
    return [(s, L) for s, L in blocks(on) if L < t_up and s > 0 and s + L < n]


def short_off_gaps(on, t_dn=T_DN):
    """Off gaps strictly between two on-blocks, shorter than t_dn."""
    on = np.asarray(on, bool)
    b = blocks(~on)
    n = len(on)
    return [(s, L) for s, L in b if s > 0 and s + L < n and L < t_dn]


def below_min_load(diesel, rho=RHO_MIN, rated=RATED):
    diesel = np.asarray(diesel, float)
    return np.flatnonzero((diesel > EPS) & (diesel < rho * rated - 1e-6))


def out_without_flag(diesel, flag):
    return np.flatnonzero((np.asarray(diesel) > EPS) & ~np.asarray(flag, bool))


def starts(on):
    on = np.asarray(on, bool)
    return int((on & ~np.r_[False, on[:-1]]).sum())


def load_locked(name):
    z = np.load(LOCKED / f"{name}_dispatch.npz")
    return {k: z[k] for k in z.files}


def load_fold(year, ctrl):
    z = np.load(FOLDS / f"fold_{year}_dispatch.npz", allow_pickle=True)
    return {k[len(ctrl) + 1:]: z[k] for k in z.files if k.startswith(ctrl + "_")}, z["datetime"]


# ------------------------------------------------------------ instrumented run
WINDOW_START = 72          # 2025-01-04 00:00, generator off in the hour before
WINDOW_HOURS = 504         # three weeks; densest violation window in the locked run


def run_window(start=WINDOW_START, hours=WINDOW_HOURS):
    """Re-run the locked rolling-MILP configuration (24 h, hourly, hierarchical,
    33.75 GHS, sourced-mean curve, direct forecasts) on a window of 2025 and log,
    for every executed hour, the plan the MILP produced and what was executed.

    Instrumentation wraps milp_dispatch.solve and dispatch_core.feasibility_reserve;
    controller logic is untouched.
    """
    os.environ["MG_LOAD"] = "v2"
    import pandas as pd
    from scripts_v2 import milp_dispatch as mp, dispatch_core as dc
    from scripts_v2 import controllers as ct, cost_model as cm
    from scripts_v2 import walkforward as W
    from scripts_v2.fc_compat import year_trajectory
    d = W.full_series()
    df = d[d.datetime.dt.year == 2025].reset_index(drop=True)
    sl = slice(start, start + hours)
    traj = year_trajectory(2025, sl)
    nan = np.full(hours, np.nan)
    pv = df.PV_power_kW.values[sl]; ld = df.total_load_kW.values[sl]
    # Starting SOC: the baseline's state at the hour before the window, from
    # the SOC carried into 1 January (no dependence on any stored result).
    from scripts_v2 import walkforward as W
    b = dc.baseline(df.PV_power_kW.values[:start], df.total_load_kW.values[:start],
                    uc_parity=False,  # year-start SOC: legacy rule
                    soc_init=W.soc_at_year_start(2025))
    soc0 = float(b.soc[-1])
    cost = cm.corrected("sourced_mean", start_cost_ghs=33.75)
    log = {"z_init": [], "plan_on": [], "plan_g": [], "floor": []}
    ctl_log = {}
    orig_solve, orig_res = mp.solve, dc.feasibility_reserve

    def solve(*a, **k):
        sol = orig_solve(*a, **k)
        log["z_init"].append(int(k.get("z_init", 0)))
        log["plan_on"].append(bool(sol.running[0])); log["plan_g"].append(float(sol.diesel[0]))
        return sol

    def res(*a, **k):
        f = orig_res(*a, **k); log["floor"].append(f); return f
    mp.solve, dc.feasibility_reserve = solve, res
    try:
        out, committed = ct.run_rolling_milp(
            pv, ld, nan, nan, trajectory_at=traj, soc_init=soc0, horizon=24, commit=1, cost=cost,
            hierarchical=True, time_limit=10.0, mip_gap=1e-3, log=ctl_log)
    finally:
        mp.solve, dc.feasibility_reserve = orig_solve, orig_res
    frame = pd.DataFrame({
        "time": df.datetime.values[sl], "pv": pv, "load": ld,
        "deficit": np.maximum(ld - pv, 0.0),
        "z_init": log["z_init"], "plan_on": log["plan_on"], "plan_g": log["plan_g"],
        "floor": log["floor"], "flag": committed,
        "plan_flag": ctl_log["plan_running"], "status": ctl_log["solver_status"], "g": out.diesel,
        "dis": out.discharge, "ch": out.charge, "soc": out.soc})
    return frame


# ------------------------------------------------------- live heuristic / rules
def fold_heuristic(year):
    """Heuristic for test year `year` with (theta, psi) re-selected on year-1."""
    os.environ["MG_LOAD"] = "v2"
    from scripts_v2 import walkforward as W, heuristic as H, cost_model as cm
    cost = cm.corrected("sourced_mean", start_cost_ghs=33.75)
    th, psi = W.select_heuristic(year, cost)
    d = W.full_series()
    yr = d[d.datetime.dt.year == year]
    res, run = H.commitment_heuristic(yr.PV_power_kW.values, yr.total_load_kW.values,
                                      soc_init=W.soc_at_year_start(year), threshold=th,
                                      bank_to_soc=psi, genset=cost.genset)
    return res, run, th, psi


def comparator(mode, year=2025, setpoint=0.8):
    os.environ["MG_LOAD"] = "v2"
    from scripts_v2 import walkforward as W, comparators as CP, cost_model as cm
    cost = cm.corrected("sourced_mean", start_cost_ghs=33.75)
    d = W.full_series()
    yr = d[d.datetime.dt.year == year]
    return CP.rule_dispatch(yr.PV_power_kW.values, yr.total_load_kW.values,
                            soc_init=W.soc_at_year_start(year), mode=mode,
                            genset=cost.genset, setpoint=setpoint)


def v42_dir():
    """Full-year V4.2 outputs to check (dir holding roll24.npz etc.), or None."""
    p = os.environ.get("V42_DISPATCH_DIR")
    if p:
        return Path(p)
    p = BASE / "box_validation_2025" / "2025_cs33p75"
    return p if (p / "roll24.npz").exists() else None


def deficit(year):
    os.environ["MG_LOAD"] = "v2"
    from scripts_v2 import walkforward as W
    d = W.full_series()
    yr = d[d.datetime.dt.year == year]
    return np.maximum(yr.total_load_kW.values - yr.PV_power_kW.values, 0.0)
