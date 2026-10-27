"""Phase 4 — the attribution ladder.

Every configuration is run on the held-out 2025 test year and scored against the
perfect-foresight oracle, not against the rule-based baseline. The premium
column is what the paper's central table reports.

  row 9 - row 8   true cost of forecast error
  row 8 - row 0   cost of receding-horizon commitment
  row 2 - row 7   defect cost

Usage:  python -m scripts_v2.run_ladder [--quick] [--rows 0,1,7,8]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import controllers as ct
from . import direct_forecast as D
from . import dispatch_core as dc
from . import lp_dispatch, oracle
from . import paths as P
from .config import (DIESEL_MARGINAL_COST, HORIZON, SOC_MIN)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_test_year():
    """Actual PV/load and v2 full-year forecasts for 2025."""
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    act = pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime")
    fc = pd.read_csv(P.data("forecasts.csv"), parse_dates=["datetime"])
    df = act.merge(fc, on="datetime", how="inner").sort_values("datetime")
    df = df.reset_index(drop=True)
    assert len(df) == 8760, f"expected 8760 test hours, got {len(df)}"
    return df


def soc_start_for_2025():
    """The SOC the baseline carries into 2025-01-01, so every configuration
    starts from the same physical state.

    For load_v1 this is read from the frozen baseline_simulation.csv so the
    published runs are untouched. For another load variant the same quantity is
    recomputed by running the baseline over that variant's full series, since
    the frozen file describes a different demand series.
    """
    if P.variant() == "v1":
        b = pd.read_csv(DATA / "baseline_simulation.csv", parse_dates=["datetime"])
        b = b.sort_values("datetime").reset_index(drop=True)
        i = b.index[b.datetime.dt.year == 2025][0]
        return float(b.battery_SOC_kWh.iloc[i - 1])

    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    d = pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime").sort_values("datetime")
    d = d.reset_index(drop=True)
    from .config import SOC_INIT
    res = dc.baseline(d.PV_power_kW.values, d.total_load_kW.values,
                      soc_init=SOC_INIT, uc_parity=False)  # year-start SOC: legacy rule so MILP rows are unchanged
    i = int(d.index[d.datetime.dt.year == 2025][0])
    return float(res.soc[i - 1])


# ---------------------------------------------------------------- ladder
def direct_trajectory(year=2025, seed=42):
    """Direct plan for the active load variant, indexed on the full series."""
    frame = D.observed_frame()
    idx = D.year_index(frame, year)
    pred_pv, pred_ld = D.ensure_direct(frame, year, D.cache_path(year, seed), seed=seed)
    return D.trajectory_fn(pred_pv, pred_ld, idx), frame, idx


def build_configs(df, soc0, quick=False, trajectory_at=None):
    PVa = df.PV_power_kW.values
    LDa = df.total_load_kW.values

    if quick:
        n = 24 * 28
        PVa, LDa = PVa[:n], LDa[:n]

    if trajectory_at is None:
        raise RuntimeError("build_configs requires the direct-forecast trajectory")

    base_kw = dict(pv_actual=PVa, load_actual=LDa, soc_init=soc0, horizon=HORIZON)
    fc = dict(pv_forecast=np.full(len(PVa), np.nan),
              load_forecast=np.full(len(PVa), np.nan),
              trajectory_at=trajectory_at)
    # Rows 9 and 10p plan on the realised future. The flag is the same
    # escape used by roll24p. Row 9 is the perfect-forecast setpoint row.
    perfect = dict(pv_forecast=PVa, load_forecast=LDa, perfect_foresight=True)

    cfgs = []

    cfgs.append(("0", "Perfect-foresight oracle (global optimum)",
                 lambda: oracle.solve_year(PVa, LDa, soc_init=soc0)[1]))

    cfgs.append(("1", "Rule-based baseline",
                 lambda: dc.baseline(PVa, LDa, soc_init=soc0, uc_parity=False)))  # flat ladder

    # Row 2: the published configuration, on full-year data (D5 corrected).
    cfgs.append(("2", "Published pipeline (GA, 6 h block, D1+D2+D4 present)",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_ga, commit=6,
                     executor="legacy", conserve_energy=False, rt_balance=False,
                     planner_kw=dict())))

    cfgs.append(("3", "+ fix D1 (energy conservation)",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_ga, commit=6,
                     executor="legacy", conserve_energy=True, rt_balance=False,
                     planner_kw=dict(charge_from_surplus=True))))

    cfgs.append(("4", "+ fix D2 (diesel capped at residual)",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_ga, commit=6,
                     executor="legacy", conserve_energy=True, rt_balance=False,
                     planner_kw=dict(charge_from_surplus=True, clip_diesel=True))))

    cfgs.append(("5", "+ fix D4 (real-time balancing layer)",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_ga, commit=6,
                     executor="core", rt_balance=True, battery_mode="setpoint",
                     planner_kw=dict(charge_from_surplus=True, clip_diesel=True))))

    cfgs.append(("6", "+ terminal SOC value",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_ga, commit=6,
                     executor="core", rt_balance=True, battery_mode="setpoint",
                     planner_kw=dict(charge_from_surplus=True, clip_diesel=True,
                                     terminal_soc_value=DIESEL_MARGINAL_COST))))

    cfgs.append(("7", "LP planner replacing GA, 6 h commitment",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_lp, commit=6,
                     executor="core", rt_balance=True, battery_mode="setpoint",
                     planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST))))

    cfgs.append(("8", "LP planner, 1 h receding horizon",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_lp, commit=1,
                     executor="core", rt_balance=True, battery_mode="setpoint",
                     planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST))))

    cfgs.append(("9", "LP planner, 1 h, perfect forecasts",
                 lambda: ct.run_receding(
                     **base_kw, **perfect, planner=ct.plan_lp, commit=1,
                     executor="core", rt_balance=True, battery_mode="setpoint",
                     planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST))))

    cfgs.append(("10", "LP planner, 1 h, hierarchical balancing (SOC envelope)",
                 lambda: ct.run_receding(
                     **base_kw, **fc, planner=ct.plan_lp, commit=1,
                     executor="core", rt_balance=True, battery_mode="merit",
                     use_soc_floor=True,
                     planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST))))

    cfgs.append(("10p", "LP, 1 h, perfect forecasts, full balancing (ties oracle)",
                 lambda: ct.run_receding(
                     **base_kw, **perfect, planner=ct.plan_lp, commit=1,
                     executor="core", rt_balance=True, battery_mode="merit",
                     use_soc_floor=True,
                     planner_kw=dict(terminal_soc_value=DIESEL_MARGINAL_COST))))

    return cfgs, (PVa, LDa)


def manifest(extra=None):
    import scipy, sklearn, xgboost
    m = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        m["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        m["git_commit"] = None
    m.update(extra or {})
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="28-day slice")
    ap.add_argument("--rows", default=None, help="comma-separated row ids")
    ap.add_argument("--out", default="attribution_ladder.csv")
    a = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    df = load_test_year()
    soc0 = soc_start_for_2025()
    traj, frame, idx = direct_trajectory(2025)
    if len(df) != len(idx) or not np.array_equal(
            df.datetime.to_numpy(), frame.datetime.to_numpy()[idx]):
        raise RuntimeError("test-year table is not the observed 2025 block")
    print(f"Test year 2025: {len(df)} h | start SOC {soc0:.2f} kWh")

    cfgs, (PVa, LDa) = build_configs(df, soc0, quick=a.quick, trajectory_at=traj)
    if a.rows:
        want = set(a.rows.split(","))
        cfgs = [c for c in cfgs if c[0] in want]

    oracle_res = None
    rows, sims = [], {}
    for rid, label, fn in cfgs:
        t0 = time.time()
        res = fn()
        dt = time.time() - t0
        m = res.metrics()
        sims[rid] = res
        if rid == "0":
            oracle_res = res
        rows.append(dict(row=rid, configuration=label, seconds=round(dt, 2), **m))
        print(f"  [{rid:>3}] {label[:52]:<52} {dt:7.1f}s  "
              f"diesel {m['diesel_kWh_per_year']:>10,.0f} kWh  "
              f"LPSP {m['LPSP_pct']:.4f}%")

    out = pd.DataFrame(rows)
    if oracle_res is not None:
        o = dc.objective(oracle_res)
        od = oracle_res.diesel.sum()
        out["premium_vs_oracle_pct"] = (out.objective / o - 1.0) * 100.0
        out["diesel_premium_vs_oracle_pct"] = (
            out.diesel_kWh_per_year / out.loc[out.row == "0",
                                              "diesel_kWh_per_year"].iloc[0] - 1.0) * 100.0
        # Any controller below the oracle means information has leaked.
        bad = out[out.premium_vs_oracle_pct < -1e-6]
        if len(bad):
            print("\n*** A CONTROLLER BEAT THE ORACLE — INFORMATION LEAK ***")
            print(bad[["row", "configuration", "premium_vs_oracle_pct"]])

    # A --quick run must never land on a canonical result filename: the stored
    # ladder and replan sweep once disagreed by a factor of two on identical
    # configurations because one of them was a 28-day slice.
    name = a.out
    if a.quick and not Path(name).stem.endswith("_quick"):
        stem, ext = name.rsplit(".", 1)
        name = f"{stem}_quick.{ext}"
    path = P.results_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    path.with_name(path.stem + "_manifest.json").write_text(
        json.dumps(manifest({"rows": [c[0] for c in cfgs], "quick": a.quick,
                             "soc_init": soc0, "hours": len(PVa),
                             "forecast": "direct"}), indent=2))
    print(f"\nSaved {path}")
    np.savez_compressed(path.with_name(path.stem + "_dispatch.npz"),
                        **{f"{k}_{f}": getattr(v, f)
                           for k, v in sims.items()
                           for f in ("diesel", "soc", "charge", "discharge",
                                     "unmet", "curtail")})
    return out


if __name__ == "__main__":
    main()
