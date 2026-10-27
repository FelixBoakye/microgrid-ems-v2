"""Hourly dispatch trace for one representative week — results/week_trace.csv.

Week selection is by a stated, reproducible criterion, not by eye:

    among all complete Monday-to-Sunday weeks of the test year, choose the one
    minimising the sum of relative absolute deviations of (a) its mean daily PV
    energy and (b) its mean daily peak load from the *annual medians* of those
    two daily quantities

Both terms are normalised by the annual median so neither unit dominates. The
criterion, the resulting date range and the deviations are written to
results/week_trace_manifest.json.

Traces are sliced out of the annual dispatches already stored in
results/phase5/*_dispatch.npz rather than re-simulated. That keeps the week
exactly consistent with the annual numbers, including each controller's inherited
SOC and genset state at the start of the week, which a fresh short simulation
would get wrong.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import cost_model as cm
from . import paths as P
from .run_ladder import load_test_year, manifest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
P5 = RESULTS / "phase5"

CONTROLLERS = {
    "oracle": "Perfect-foresight unit-commitment ceiling",
    "baseline": "Rule-based merit-order baseline",
    "heuristic": "Reactive high-loading heuristic (70% target, bank to 50%)",
    "roll24": "Rolling MILP, 24 h horizon, 1 h re-plan, XGBoost forecasts",
}

CRITERION = (
    "Among all complete Mon-Sun weeks of the test year, the week minimising "
    "|mean daily PV energy - annual median daily PV energy| / annual median + "
    "|mean daily peak load - annual median daily peak load| / annual median."
)


def choose_week(df):
    """Return (start_idx, end_idx, diagnostics) for the representative week."""
    d = df[["datetime", "PV_power_kW", "total_load_kW"]].copy()
    d["date"] = d.datetime.dt.date

    daily = d.groupby("date").agg(pv_kwh=("PV_power_kW", "sum"),
                                  peak_kw=("total_load_kW", "max"))
    med_pv = float(daily.pv_kwh.median())
    med_peak = float(daily.peak_kw.median())

    # Complete Monday-start weeks only.
    first_monday = d.datetime.dt.dayofweek.values.argmin() \
        if d.datetime.dt.dayofweek.iloc[0] != 0 else 0
    starts = [i for i in range(len(d))
              if d.datetime.iloc[i].dayofweek == 0
              and d.datetime.iloc[i].hour == 0
              and i + 168 <= len(d)]

    best = None
    for i in starts:
        w = d.iloc[i:i + 168]
        wd = w.groupby("date").agg(pv_kwh=("PV_power_kW", "sum"),
                                   peak_kw=("total_load_kW", "max"))
        if len(wd) != 7:
            continue
        dev_pv = abs(wd.pv_kwh.mean() - med_pv) / med_pv
        dev_peak = abs(wd.peak_kw.mean() - med_peak) / med_peak
        score = dev_pv + dev_peak
        if best is None or score < best["score"]:
            best = dict(start=i, score=float(score), dev_pv=float(dev_pv),
                        dev_peak=float(dev_peak),
                        week_mean_daily_pv_kwh=float(wd.pv_kwh.mean()),
                        week_mean_daily_peak_kw=float(wd.peak_kw.mean()))
    if best is None:
        raise SystemExit("no complete week found")
    best.update(annual_median_daily_pv_kwh=med_pv,
                annual_median_daily_peak_kw=med_peak,
                n_candidate_weeks=len(starts))
    return best["start"], best["start"] + 168, best


def _locations():
    """Canonical week trace stays in results/. A rerun writes beside its ladder."""
    if os.environ.get("MG_RESULTS_ROOT", "").strip():
        root = P.results_root()
        return root / "phase5_sourced_mean", root
    return P5, RESULTS


def build(controllers=None):
    controllers = controllers or list(CONTROLLERS)
    dispatch_dir, out_dir = _locations()
    df = P.smoke_trim(load_test_year())
    i0, i1, diag = choose_week(df)
    sl = slice(i0, i1)
    rated = cm.CORRECTED.genset.rated_kw

    week = df.iloc[sl]
    frames, summary = [], []

    for name in controllers:
        f = dispatch_dir / f"{name}_dispatch.npz"
        if not f.exists():
            print(f"  (skipping {name}: {f.name} not present)")
            continue
        z = np.load(f)
        g = z["diesel"][sl]
        on = z["running"][sl].astype(bool) if "running" in z else (g > 1e-9)
        lf = np.where(on, g / rated, np.nan)

        frames.append(pd.DataFrame({
            "datetime": week.datetime.values,
            "controller": name,
            "controller_label": CONTROLLERS.get(name, name),
            "pv_available_kW": week.PV_power_kW.values,
            "load_kW": week.total_load_kW.values,
            "battery_charge_kW": z["charge"][sl],
            "battery_discharge_kW": z["discharge"][sl],
            "soc_kWh": z["soc"][sl],
            "genset_kW": g,
            "genset_on": on.astype(int),
            "genset_load_factor": lf,
            "unmet_load_kW": z["unmet"][sl],
            "curtailed_pv_kW": z["curtail"][sl],
        }))

        # Week-level figures, so the trace can be checked against the annual run.
        ann = json.loads((dispatch_dir / f"{name}.json").read_text())
        summary.append(dict(
            controller=name,
            week_genset_run_hours=int(on.sum()),
            week_mean_load_factor_pct=float(np.nanmean(lf) * 100) if on.any() else 0.0,
            week_fuel_litres=float(cm.CORRECTED.fuel_cost(g, on)[1]),
            week_diesel_kWh=float(g.sum()),
            week_unmet_kWh=float(z["unmet"][sl].sum()),
            week_curtailed_kWh=float(z["curtail"][sl].sum()),
            annual_genset_run_hours=ann["genset_run_hours"],
            annual_mean_load_factor_pct=ann["mean_load_factor_pct"],
        ))

    trace = pd.concat(frames, ignore_index=True)
    trace.to_csv(out_dir / "week_trace.csv", index=False)

    summ = pd.DataFrame(summary)
    summ.to_csv(out_dir / "week_trace_summary.csv", index=False)

    man = manifest({
        "selection_criterion": CRITERION,
        "week_start": str(week.datetime.iloc[0]),
        "week_end": str(week.datetime.iloc[-1]),
        "week_start_index": int(i0),
        "hours": 168,
        "controllers": controllers,
        **{k: v for k, v in diag.items() if k != "start"},
    })
    (out_dir / "week_trace_manifest.json").write_text(json.dumps(man, indent=2))

    print(f"Representative week: {week.datetime.iloc[0]} to {week.datetime.iloc[-1]}")
    print(f"  criterion: {CRITERION}")
    print(f"  week mean daily PV {diag['week_mean_daily_pv_kwh']:.1f} kWh "
          f"vs annual median {diag['annual_median_daily_pv_kwh']:.1f} "
          f"({diag['dev_pv'] * 100:.2f}% off)")
    print(f"  week mean daily peak load {diag['week_mean_daily_peak_kw']:.2f} kW "
          f"vs annual median {diag['annual_median_daily_peak_kw']:.2f} "
          f"({diag['dev_peak'] * 100:.2f}% off)")
    print(f"  chosen from {diag['n_candidate_weeks']} complete weeks\n")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\nSaved {out_dir / 'week_trace.csv'} ({len(trace)} rows)")
    return trace, summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", default=None,
                    help="comma-separated subset")
    a = ap.parse_args(argv)
    build(a.controllers.split(",") if a.controllers else None)


if __name__ == "__main__":
    main()
