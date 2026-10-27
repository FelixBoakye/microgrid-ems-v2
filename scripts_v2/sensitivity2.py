"""Phase 6 Step 5b — demand-model sensitivity, sizing sweep, seed variance.

Three sweeps that the genset sensitivity in `sensitivity.py` does not cover:

* **demand model** — the AR(1) coefficient phi, the productive day-of-week
  factors and the household temperature response. All three are assumptions
  (see results/load_validation.md section 6-7) and each needs a fresh demand
  series, dataset and forecaster, not just a cost-model override.
* **sizing** — battery 65/130/195/260 kWh against PV 60/90/120 kWp, the
  "why not just buy more battery instead of intelligence?" question.
* **forecast seeds** — ten seeds on the nominal setting, to separate genuine
  effects from forecaster noise.

Everything runs on the same stratified 3-month window as `sensitivity.py`, whose
offset against the full year is known and reported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import controllers as ct
from . import cost_model as cm
from . import direct_forecast as D
from . import dispatch_core as dc
from . import heuristic as H
from . import load_model as LM
from . import milp_dispatch as mp
from .forecast import metrics
from . import paths as P
from .sensitivity import WINDOW_MONTHS, TEST_YEAR, evaluate, out_dir, tune_series
from .walkforward import soc_at_year_start

DATA = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------- demand
def _slug(label):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label)[:80]


def _window_from_frame(df, pred_pv, pred_ld):
    idx = D.year_index(df, TEST_YEAR)
    yr = df.iloc[idx]
    keep = yr.datetime.dt.month.isin(WINDOW_MONTHS).to_numpy()
    abs_of_local = idx[keep]
    block_end = D.contiguous_block_ends(abs_of_local)
    traj = D.trajectory_fn(pred_pv, pred_ld, abs_of_local, block_end)
    return (P.smoke_trim(yr.PV_power_kW.to_numpy(dtype=float)[keep]),
            P.smoke_trim(yr.total_load_kW.to_numpy(dtype=float)[keep]), traj, idx)


def make_series(tag, seed=42, **gen_kw):
    """Generate a demand variant and fit its own direct per-horizon models."""
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    orig = pd.read_csv(DATA / "synthetic_load.csv", parse_dates=["datetime"])
    v1a = orig.total_load_kW.sum() / (len(orig) / 8760)
    ld = LM.generate_v2(pv.datetime, pv.T2M.values, v1_annual_kwh=v1a, **gen_kw)

    df = pv.merge(ld[["datetime", "total_load_kW"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    path = out_dir() / "direct" / f"{_slug(tag)}_seed{seed}.npz"
    pred_pv, pred_ld = D.ensure_direct(df, TEST_YEAR, path, seed=seed)
    PVa, LDa, traj, idx = _window_from_frame(df, pred_pv, pred_ld)
    y_pv = df.PV_power_kW.to_numpy(dtype=float)[idx]
    y_ld = df.total_load_kW.to_numpy(dtype=float)[idx]
    mets = {"xgb_PV": metrics(y_pv, pred_pv[1, idx]),
            "xgb_load": metrics(y_ld, pred_ld[1, idx])}
    return PVa, LDa, traj, mets, df


def demand_sweep():
    rows = []
    variants = {}
    for phi in (0.50, 0.70, 0.85):
        variants[f"phi = {phi:.2f}"] = dict(params=LM.LoadV2Params(phi=phi))
    for dow, lbl in (((1,) * 5 + (1.0, 1.0), "flat (no weekend effect)"),
                     ((1,) * 5 + (0.85, 0.45), "nominal Sat .85 / Sun .45"),
                     ((1,) * 5 + (0.70, 0.25), "strong Sat .70 / Sun .25")):
        variants[f"day-of-week: {lbl}"] = dict(
            params=LM.LoadV2Params(dow_productive=tuple(float(x) for x in dow)))
    for b in (0.000, 0.020, 0.040):
        variants[f"temp response beta = {b:.3f}"] = dict(
            params=LM.LoadV2Params(beta_temp=b))

    soc0 = soc_at_year_start(TEST_YEAR)
    items = P.smoke_groups(variants.items(), key=lambda it: it[0].split(":")[0].split("=")[0])
    for label, kw in items:
        PVa, LDa, traj, mets, frame = make_series(label, **kw)
        # Year-1 selection on the variant's own demand series.
        rec = evaluate(cm.CORRECTED, PVa, LDa, soc0, trajectory_at=traj,
                       tune_on=tune_series(frame=frame), label=label)
        rec["load_R2"] = mets["xgb_load"]["r2"]
        rec["load_RMSE"] = mets["xgb_load"]["rmse"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / "demand_sensitivity.csv", index=False)
    return df


# --------------------------------------------------------------- sizing
def sizing_sweep(batteries=(65, 130, 195, 260), pvs=(60, 90, 120)):
    """Scale the battery and PV, holding demand fixed."""
    from .sensitivity import window
    PVa, LDa, _soc0, traj = window()
    orig = (C.BATTERY_CAPACITY_KWH, C.SOC_MIN, C.SOC_MAX,
            C.MAX_CHARGE_KW, C.MAX_DISCHARGE_KW)
    rows = []
    batteries, pvs = P.smoke_groups(batteries, key=lambda _: 0), P.smoke_groups(pvs, key=lambda _: 0)
    try:
        for b in batteries:
            for p in pvs:
                C.BATTERY_CAPACITY_KWH = b
                C.SOC_MIN = (1 - C.DOD_MAX) * b
                C.SOC_MAX = b
                C.MAX_CHARGE_KW = C.MAX_DISCHARGE_KW = 0.5 * b
                for mod in (dc, ct, H, mp):
                    for nm in ("SOC_MIN", "SOC_MAX", "MAX_CHARGE_KW",
                               "MAX_DISCHARGE_KW"):
                        if hasattr(mod, nm):
                            setattr(mod, nm, getattr(C, nm))
                s = p / 90.0

                def scaled(i, _s=s, _traj=traj):
                    pv_fc, ld_fc = _traj(i)
                    return pv_fc * _s, ld_fc

                # Year-1 selection on the same resized system (PV scaled, pack
                # patched), starting from SOC_MIN like the evaluation.
                rec = evaluate(cm.CORRECTED, PVa * s, LDa, C.SOC_MIN,
                               trajectory_at=scaled,
                               tune_on=tune_series(scale_pv=s, soc_prev=C.SOC_MIN),
                               label=f"battery {b} kWh / PV {p} kWp")
                rec.update(battery_kwh=b, pv_kwp=p)
                rows.append(rec)
    finally:
        (C.BATTERY_CAPACITY_KWH, C.SOC_MIN, C.SOC_MAX,
         C.MAX_CHARGE_KW, C.MAX_DISCHARGE_KW) = orig
        for mod in (dc, ct, H, mp):
            for nm in ("SOC_MIN", "SOC_MAX", "MAX_CHARGE_KW", "MAX_DISCHARGE_KW"):
                if hasattr(mod, nm):
                    setattr(mod, nm, getattr(C, nm))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / "sizing_sweep.csv", index=False)
    return df


# ---------------------------------------------------------------- seeds
def seed_sweep(n=10):
    """Forecaster seeds on the nominal demand model. ``n`` is the full count."""
    df = D.observed_frame()
    soc0 = soc_at_year_start(TEST_YEAR)
    prev = tune_series()
    rows = []
    for seed in range(1 if P.smoke_hours() else n):
        path = out_dir() / "direct" / f"nominal_seed{seed}.npz"
        pred_pv, pred_ld = D.ensure_direct(df, TEST_YEAR, path, seed=seed)
        PVa, LDa, traj, idx = _window_from_frame(df, pred_pv, pred_ld)
        rec = evaluate(cm.CORRECTED, PVa, LDa, soc0, trajectory_at=traj,
                       tune_on=prev, label=f"seed {seed}")
        y_pv = df.PV_power_kW.to_numpy(dtype=float)[idx]
        y_ld = df.total_load_kW.to_numpy(dtype=float)[idx]
        rec["load_R2"] = metrics(y_ld, pred_ld[1, idx])["r2"]
        rec["PV_R2"] = metrics(y_pv, pred_pv[1, idx])["r2"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / "seed_variance.csv", index=False)
    print("\nSEED VARIANCE (mean +- std over %d seeds)" % n)
    for k in ("milp_vs_heuristic_pct", "milp_vs_baseline_pct",
              "milp_premium_vs_oracle_pct", "load_R2"):
        print(f"  {k:<30} {df[k].mean():>9.3f} +- {df[k].std(ddof=1):.3f}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["demand", "sizing", "seeds"])
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)
    t0 = time.time()
    df = {"demand": demand_sweep, "sizing": sizing_sweep,
          "seeds": lambda: seed_sweep(a.n)}[a.stage]()
    print(f"\n[{a.stage} done in {(time.time() - t0) / 60:.1f} min]")
    return df


if __name__ == "__main__":
    main()
