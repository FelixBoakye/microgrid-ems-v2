"""Per-plan wall time of the 2025 hourly rolling MILP (24 h and/or 48 h horizon).

Times every mp.solve call inside controllers.run_rolling_milp (10 s, gap 1e-3,
hierarchical, direct forecasts), so the mean, p95 and longest re-plan come from a
real year, not a sample. Writes <MG_RESULTS_ROOT>/replan_times_h<H>.csv and a summary json.

    python tools/v42b_replan_times.py --horizon 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts_v2 import controllers as ct, cost_model as cm, milp_dispatch as mp, paths as P  # noqa: E402
from scripts_v2.run_ladder import direct_trajectory, soc_start_for_2025  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    a = ap.parse_args(argv)
    out = Path(os.environ["MG_RESULTS_ROOT"])
    traj, frame, idx = direct_trajectory(2025)
    yr = frame.iloc[idx].reset_index(drop=True)
    n = P.smoke_hours() or len(yr)
    PVa, LDa = yr.PV_power_kW.values[:n], yr.total_load_kW.values[:n]
    soc0 = soc_start_for_2025()
    times, status = [], []
    real = mp.solve

    def timed(*args, **kw):
        t0 = time.perf_counter()
        sol = real(*args, **kw)
        times.append(time.perf_counter() - t0)
        status.append(str(getattr(sol, "status", "")))
        return sol

    mp.solve = timed
    try:
        t0 = time.perf_counter()
        res, _ = ct.run_rolling_milp(PVa, LDa, np.full(n, np.nan), np.full(n, np.nan),
                                     soc_init=soc0, horizon=a.horizon, commit=1,
                                     cost=cm.CORRECTED, hierarchical=True,
                                     time_limit=10.0, mip_gap=1e-3, trajectory_at=traj)
        wall = time.perf_counter() - t0
    finally:
        mp.solve = real
    t = np.asarray(times)
    pd.DataFrame({"plan": np.arange(len(t)), "seconds": t, "status": status}).to_csv(
        out / f"replan_times_h{a.horizon}.csv", index=False)
    s = {"horizon_h": a.horizon, "hours": int(n), "n_solves": int(len(t)),
         "mean_s": float(t.mean()), "median_s": float(np.median(t)),
         "p95_s": float(np.percentile(t, 95)), "max_s": float(t.max()),
         "n_at_time_limit_ge_9.9s": int(np.sum(t >= 9.9)),
         "sum_solve_s": float(t.sum()), "run_wall_s": wall,
         "total_cost_GHS": cm.CORRECTED.evaluate(res)["total_cost_GHS"],
         "start_cost_ghs": cm.CORRECTED.genset.start_cost_ghs}
    (out / f"replan_times_h{a.horizon}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
