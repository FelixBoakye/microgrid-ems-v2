"""The three structural invariants — Phase 6, Step 2 gate.

These must survive a change of demand series. If they do not, the finding was a
property of the old load series rather than of the cost structure, which is a
different and much weaker paper.

  1. flat-model oracle ties the merit-order baseline exactly (objective 0.000e+00)
  2. hierarchical execution ties the oracle at every re-planning interval
  3. no controller comes in below the oracle, under either cost model

Run with MG_LOAD=v2 to check the load_v2 series.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import controllers as ct
from . import cost_model as cm
from . import dispatch_core as dc
from . import oracle
from . import paths as P
from .config import DIESEL_MARGINAL_COST, HORIZON
from .run_ladder import direct_trajectory, load_test_year, manifest, soc_start_for_2025

INTERVALS = (1, 2, 3, 6, 12, 24)


def main():
    df = load_test_year()
    soc0 = soc_start_for_2025()
    PVa, LDa = df.PV_power_kW.values, df.total_load_kW.values
    traj, frame, idx = direct_trajectory(2025)
    if len(df) != len(idx) or not np.array_equal(
            df.datetime.to_numpy(), frame.datetime.to_numpy()[idx]):
        raise RuntimeError("test-year table is not the observed 2025 block")
    print(f"load variant: {P.variant()} | {len(df)} h | start SOC {soc0:.3f} kWh\n")

    out = {"variant": P.variant(), "hours": int(len(df)), "soc_init": soc0}
    ok = True

    # ---- 1. flat-model oracle ties the greedy baseline exactly ----
    base = dc.baseline(PVa, LDa, soc_init=soc0, uc_parity=False)  # flat-model tie
    _, orc = oracle.solve_year(PVa, LDa, soc_init=soc0)
    d_obj = dc.objective(orc) - dc.objective(base)
    d_diesel = orc.diesel.sum() - base.diesel.sum()
    inv1 = abs(d_obj) < 1e-6
    ok &= inv1
    print("INVARIANT 1 — flat-model oracle vs merit-order baseline")
    print(f"  baseline objective {dc.objective(base):,.10f}")
    print(f"  oracle   objective {dc.objective(orc):,.10f}")
    print(f"  difference {d_obj:+.3e}   diesel difference {d_diesel:+.3e} kWh")
    print(f"  -> {'PASS' if inv1 else '*** FAIL ***'}\n")
    out["invariant_1_objective_difference"] = float(d_obj)
    out["invariant_1_pass"] = bool(inv1)

    # ---- 2. hierarchical execution ties the oracle at every interval ----
    o_obj = dc.objective(orc)
    print("INVARIANT 2 — hierarchical execution vs oracle, by re-planning interval")
    prem = {}
    for k in INTERVALS:
        res = ct.run_receding(
            pv_actual=PVa, load_actual=LDa,
            pv_forecast=np.full(len(PVa), np.nan),
            load_forecast=np.full(len(PVa), np.nan),
            trajectory_at=traj,
            planner=ct.plan_lp, soc_init=soc0, horizon=HORIZON, commit=k,
            executor="core", rt_balance=True, battery_mode="merit",
            use_soc_floor=True,
            planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST))
        p = (dc.objective(res) / o_obj - 1.0) * 100.0
        prem[k] = float(p)
        print(f"  {k:>3} h  premium {p:+.6f}%  diesel {res.diesel.sum():,.2f} kWh")
    inv2 = all(abs(v) < 1e-6 for v in prem.values())
    ok &= inv2
    print(f"  -> {'PASS' if inv2 else '*** FAIL ***'}\n")
    out["invariant_2_premium_by_interval_pct"] = prem
    out["invariant_2_pass"] = bool(inv2)

    # ---- 3. nothing below the oracle, under either cost model ----
    print("INVARIANT 3 — no controller below the oracle")
    controllers = {
        "baseline": base,
        "idle battery": dc.simulate(PVa, LDa, soc_init=soc0,
                                    d_cmd=np.zeros(len(PVa)), rt_balance=True,
                                    battery_mode="setpoint"),
        "reserve floor 60 kWh": dc.simulate(PVa, LDa, soc_init=soc0,
                                            rt_balance=True, battery_mode="merit",
                                            soc_floor=60.0),
        "LP 6 h setpoint": ct.run_receding(
            pv_actual=PVa, load_actual=LDa,
            pv_forecast=np.full(len(PVa), np.nan),
            load_forecast=np.full(len(PVa), np.nan),
            trajectory_at=traj,
            planner=ct.plan_lp, soc_init=soc0, horizon=HORIZON, commit=6,
            executor="core", rt_balance=True, battery_mode="setpoint",
            planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST)),
    }
    worst = 0.0
    for label, res in controllers.items():
        p = (dc.objective(res) / o_obj - 1.0) * 100.0
        worst = min(worst, p)
        print(f"  {label:<22} premium vs oracle {p:+9.4f}%")
    inv3 = worst >= -1e-6
    ok &= inv3
    print(f"  -> {'PASS' if inv3 else '*** FAIL ***'}\n")
    out["invariant_3_min_premium_pct"] = float(worst)
    out["invariant_3_pass"] = bool(inv3)

    out["all_pass"] = bool(ok)
    path = P.results_subdir(f"invariants{P.tag()}.json")
    path.write_text(json.dumps(manifest(out), indent=2))
    print(f"{'ALL THREE INVARIANTS HOLD' if ok else '*** INVARIANT FAILURE ***'}")
    print(f"Saved {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
