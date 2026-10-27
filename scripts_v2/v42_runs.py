"""V4.2 runs: fixed rolling MILP, heuristic, comparators, ablations.

One job per process so jobs can run in parallel; each writes
<out>/<year>_cs<cost>/<job>.npz (+ .json metrics). ``collect`` builds a table.

Jobs
  rules        baseline, heuristic (V4.2 grid and V4.0 grid, tuned on year-1),
               constrained merit order, load following, cycle charging
  oracle       perfect-foresight annual optimum (monthly blocks)
  roll24       fixed rolling MILP (24 h, hourly re-plan, direct forecasts)
  abl_cs0      ablation: plans with start cost 0 (evaluated at the real cost)
  abl_heurz    ablation: commitment fixed to the heuristic's executed on/off,
               dispatch optimised on forecasts (needs rules)
  abl_oraclez  ablation: commitment fixed to the oracle's on/off (perfect-
               foresight timing), dispatch on forecasts (needs oracle)
  q_rules      quiet-hours scenario (unit off 23:00-05:00): heuristic, (theta,
               psi) re-selected on year-1 under the same scenario
  q_oracle     quiet-hours God mode
  q_roll24     quiet-hours rolling MILP
  abl_nobank   ablation: generator may not charge the battery beyond the
               surplus forced by minimum load

Usage (MG_LOAD=v2):
  python -m scripts_v2.v42_runs --job roll24 --year 2025 --start-cost 33.75 --out DIR
  python -m scripts_v2.v42_runs --job collect --year 2025 --start-cost 33.75 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import comparators as CP
from . import controllers as ct
from . import cost_model as cm
from . import dispatch_core as dc
from . import heuristic as H
from . import milp_dispatch as mp
from . import paths as P
from . import walkforward as W

FIELDS = ("soc", "charge", "discharge", "diesel", "unmet", "curtail")


def _save(d, name, res, running, extra=None):
    """Atomic: write <name>.tmp.npz / .tmp.json, then rename (json last)."""
    from .progress import atomic_write_text
    arr = {f: getattr(res, f) for f in FIELDS}
    arr["running"] = np.asarray(running, bool)
    for k, v in (extra or {}).items():
        if isinstance(v, np.ndarray):
            arr[k] = v
    tmp = d / f"{name}.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arr)
    os.replace(tmp, d / f"{name}.npz")
    meta = {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
            for k, v in (extra or {}).items() if not isinstance(v, np.ndarray)}
    atomic_write_text(d / f"{name}.json", json.dumps(meta, indent=2))


def _load(d, name):
    z = np.load(d / f"{name}.npz")
    return {k: z[k] for k in z.files}


def _data(year):
    d = W.full_series()
    yr = d[d.datetime.dt.year == year].reset_index(drop=True)
    prev = d[d.datetime.dt.year == year - 1].reset_index(drop=True)
    return yr, prev


QUIET = (23, 0, 1, 2, 3, 4)


def quiet_mask(dt):
    return pd.to_datetime(pd.Series(dt)).dt.hour.isin(QUIET).to_numpy()


def run(job, year, cs, out):
    cost = cm.corrected("sourced_mean", start_cost_ghs=cs)
    yr, prev = _data(year)
    PVa, LDa = yr.PV_power_kW.values, yr.total_load_kW.values
    soc0 = W.soc_at_year_start(year)
    t = time.time()
    qm = quiet_mask(yr.datetime)
    if job == "q_rules":
        qp = quiet_mask(prev.datetime)
        soc_p = W.soc_at_year_start(year - 1)
        best = None
        for th in W.THETAS:
            for psi in W.PSIS:
                r, _ = H.commitment_heuristic(prev.PV_power_kW.values,
                                              prev.total_load_kW.values, soc_init=soc_p,
                                              threshold=th, bank_to_soc=psi,
                                              genset=cost.genset, quiet_hours=qp)
                c = cost.evaluate(r)["total_cost_GHS"]
                if best is None or c < best[0]:
                    best = (c, th, psi)
        _, th, psi = best
        info = {}
        res, run_ = H.commitment_heuristic(PVa, LDa, soc_init=soc0, threshold=th,
                                           bank_to_soc=psi, genset=cost.genset,
                                           quiet_hours=qm, info=info)
        _save(out, "q_heuristic", res, run_, dict(info, theta=th, psi=psi))
    elif job == "q_oracle":
        # God mode has no backstop, so quiet hours before the first non-quiet
        # hour of the year (00:00-04:00 on 1 January, entered at whatever SOC
        # the year starts with and no chance to charge first) are exempt;
        # otherwise a low starting SOC forces load shedding there.
        qo = qm.copy()
        qo[:int(np.argmax(~qm))] = False
        res, run_ = mp.solve_year_blocked(PVa, LDa, soc0, cost, block_h=744,
                                          overlap_h=48, time_limit=300.0,
                                          mip_gap=1e-3, verbose=False, quiet=qo)
        _save(out, "q_oracle", res, run_, dict(
            quiet_run_hours=int((res.diesel[qm] > 1e-9).sum()),
            quiet_unmet_kwh=float(res.unmet[qm].sum())))
    elif job == "rules":
        _save(out, "baseline", dc.baseline(PVa, LDa, soc_init=soc0, genset=cost.genset),
              np.zeros(len(PVa), bool))
        soc_p = W.soc_at_year_start(year - 1)
        for tag, grid in (("heuristic", W.THETAS), ("heuristic_v40grid", W.THETAS_V40)):
            sw = H.sweep_threshold(prev.PV_power_kW.values, prev.total_load_kW.values,
                                   soc_init=soc_p, thresholds=grid,
                                   bank_ceilings=W.PSIS, cost=cost)
            b = sw.loc[sw.total_cost_GHS.idxmin()]
            th, psi = float(b.threshold_pct) / 100, float(b.bank_to_soc_pct) / 100
            res, run_ = H.commitment_heuristic(PVa, LDa, soc_init=soc0, threshold=th,
                                               bank_to_soc=psi, genset=cost.genset)
            _save(out, tag, res, run_, dict(theta=th, psi=psi))
        sp = CP.select_cc_setpoint(prev.PV_power_kW.values, prev.total_load_kW.values,
                                   soc_p, cost)
        for mode in CP.MODES:
            res, run_, info = CP.rule_dispatch(PVa, LDa, soc_init=soc0, mode=mode,
                                               genset=cost.genset, setpoint=sp)
            _save(out, mode, res, run_, dict(info, setpoint=sp if mode ==
                                             "cycle_charging" else None))
    elif job == "oracle":
        res, run_ = mp.solve_year_blocked(PVa, LDa, soc0, cost, block_h=744,
                                          overlap_h=48, time_limit=300.0,
                                          mip_gap=1e-3, verbose=False)
        _save(out, "oracle", res, run_)
    else:
        from .fc_compat import year_trajectory
        traj = year_trajectory(year)
        nan = np.full(len(PVa), np.nan)
        kw = {}
        if job == "abl_cs0":
            kw["plan_cost"] = cm.corrected("sourced_mean", start_cost_ghs=0.0)
        elif job == "abl_heurz":
            kw["fixed_commitment"] = _load(out, "heuristic")["running"]
        elif job == "abl_oraclez":
            kw["fixed_commitment"] = _load(out, "oracle")["running"]
        elif job == "q_roll24":
            kw["quiet_hours"] = qm
        elif job == "abl_nobank":
            kw["allow_genset_charging"] = False
        else:
            assert job == "roll24", job
        log = {}
        res, run_ = ct.run_rolling_milp(PVa, LDa, nan, nan, soc_init=soc0, horizon=24,
                                        commit=1, cost=cost, hierarchical=True,
                                        time_limit=10.0, mip_gap=1e-3,
                                        progress_every=2000, log=log,
                                        trajectory_at=traj, **kw)
        st = log["solver_status"]
        gp = log["mip_gap"]
        extra = dict(plan_running=log["plan_running"], solver_status=st, mip_gap=gp,
                     n_replans=int(len(st)), n_status_nonzero=int((st != 0).sum()),
                     max_mip_gap=float(np.nanmax(gp)), mean_mip_gap=float(np.nanmean(gp)),
                     backstop_hours=log["backstop_hours"],
                     holds_relaxed=log["holds_relaxed"],
                     **{k: log[k] for k in ("quiet_emergency_hours",
                                            "quiet_emergency_kwh", "quiet_unmet_kwh")
                        if k in log},
                     below_min_load_unabsorbed=log["below_min_load_unabsorbed"])
        _save(out, job, res, run_, extra)
    print(f"{job} {year} cs={cs}: {time.time() - t:.0f}s", flush=True)


def collect(year, cs, out):
    cost = cm.corrected("sourced_mean", start_cost_ghs=cs)
    yr, _ = _data(year)
    PVa, LDa = yr.PV_power_kW.values, yr.total_load_kW.values
    rows = []
    for f in sorted(p for p in out.glob("*.npz") if ".tmp" not in p.name):
        a = _load(out, f.stem)
        res = dc.SimResult(pv=PVa, load=LDa, **{k: a[k] for k in FIELDS})
        e = cost.evaluate(res)
        g = a["diesel"] > 1e-9
        meta = json.loads((out / f"{f.stem}.json").read_text())
        rows.append(dict(config=f.stem, total_cost_GHS=e["total_cost_GHS"],
                         starts=int((g & ~np.r_[False, g[:-1]]).sum()),
                         run_hours=int(g.sum()),
                         unmet_kWh=float(a["unmet"].sum()),
                         curtail_kWh=float(a["curtail"].sum()),
                         diesel_kWh=float(a["diesel"].sum()),
                         **{k: v for k, v in meta.items()}))
    df = pd.DataFrame(rows)
    ref = df.set_index("config").total_cost_GHS
    for k in ("baseline", "heuristic", "oracle"):
        if k in ref:
            df[f"saving_vs_{k}_pct"] = (1 - df.total_cost_GHS / ref[k]) * 100
    df.to_csv(out / "summary.csv", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--start-cost", type=float, default=33.75)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out) / f"{a.year}_cs{a.start_cost:g}".replace(".", "p")
    out.mkdir(parents=True, exist_ok=True)
    if a.job == "collect":
        collect(a.year, a.start_cost, out)
    else:
        run(a.job, a.year, a.start_cost, out)


if __name__ == "__main__":
    main()
