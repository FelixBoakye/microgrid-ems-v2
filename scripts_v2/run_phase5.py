"""Phase 5 — the ladder under the corrected cost model.

Every configuration is scored under `cost_model.CORRECTED` (affine fuel curve,
start-up cost, minimum load ratio, minimum up/down times, battery throughput at
its amortised replacement cost) and compared against:

  * the merit-order rule-based baseline (what the paper compares to)
  * the reactive high-loading heuristic (the mandatory competent comparator)
  * a perfect-foresight unit-commitment ceiling

The heuristic exists so the MILP cannot take credit for fixing a loading policy
that costs nothing to fix. Any margin the MILP keeps over it is attributable to
foresight.

Usage:
    python -m scripts_v2.run_phase5 --config baseline,heuristic
    python -m scripts_v2.run_phase5 --config roll24
    python -m scripts_v2.run_phase5 --config oracle
    python -m scripts_v2.run_phase5 --collect
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import os

from . import controllers as ct
from . import cost_model as cm
from . import dispatch_core as dc
from . import heuristic as H
from . import milp_dispatch as mp
from . import paths as P
from .config import SOC_MIN
from .run_ladder import direct_trajectory, load_test_year, manifest, soc_start_for_2025

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
P5 = RESULTS / "phase5"
COST = cm.CORRECTED          # rebound by --curve / --outdir
TAG = ""                     # rebound by --tag

# V4.2: selected on year-1 (2024) by walkforward.select_heuristic under this
# ladder's cost model and load series; set in main(). --theta/--psi override it
# only for an explicitly labelled diagnostic.
HEURISTIC_BEST = None
HEURISTIC_SOURCE = "unset"


def series(quick=0):
    df = load_test_year()
    soc0 = soc_start_for_2025()
    traj, frame, idx = direct_trajectory(2025)
    if len(df) != len(idx) or not np.array_equal(
            df.datetime.to_numpy(), frame.datetime.to_numpy()[idx]):
        raise RuntimeError("test-year table is not the observed 2025 block")
    n = quick or P.smoke_hours() or len(df)
    return (df.PV_power_kW.values[:n], df.total_load_kW.values[:n], soc0, traj)


def record(name, label, res, running=None, seconds=None, extra=None):
    P5.mkdir(parents=True, exist_ok=True)
    m = res.metrics()
    e = COST.evaluate(res)
    flat = cm.FLAT.evaluate(res)
    row = dict(config=name, label=label, seconds=seconds, **m,
               **{k: v for k, v in e.items() if k != "cost_model"},
               flat_cost_GHS=flat["total_cost_GHS"], **(extra or {}))
    (P5 / f"{name}.json").write_text(json.dumps(row, indent=2, default=float))
    np.savez_compressed(P5 / f"{name}_dispatch.npz",
                        soc=res.soc, charge=res.charge, discharge=res.discharge,
                        diesel=res.diesel, unmet=res.unmet, curtail=res.curtail,
                        running=(running if running is not None
                                 else res.diesel > 1e-9))
    print(f"  -> {name}: cost {e['total_cost_GHS']:,.0f} GHS | fuel "
          f"{e['fuel_litres']:,.0f} L | run {e['genset_run_hours']:,} h | "
          f"LF {e['mean_load_factor_pct']:.1f}% | LPSP {m['LPSP_pct']:.4f}%",
          flush=True)
    return row


def run_config(name, quick=0):
    PVa, LDa, soc0, traj = series(quick)
    t0 = time.time()

    if name == "baseline":
        info = {}
        res = dc.baseline(PVa, LDa, soc_init=soc0, genset=COST.genset, info=info)
        return record(name, "Rule-based merit-order baseline", res,
                      seconds=time.time() - t0,
                      extra=dict(uc_parity=dc.uc_parity_default(),
                                 min_down_overrides=info["min_down_overrides"]))

    if name == "heuristic":
        res, run = H.commitment_heuristic(PVa, LDa, soc_init=soc0,
                                          genset=COST.genset, **HEURISTIC_BEST)
        return record(name, "Reactive high-loading heuristic "
                            f"({HEURISTIC_BEST['threshold']:.0%} target, "
                            f"bank to {HEURISTIC_BEST['bank_to_soc']:.0%})",
                      res, run, time.time() - t0,
                      extra=dict(theta=HEURISTIC_BEST["threshold"],
                                 psi=HEURISTIC_BEST["bank_to_soc"],
                                 heuristic_selected_on=HEURISTIC_SOURCE))

    if name == "oracle":
        res, run = mp.solve_year_blocked(PVa, LDa, soc0, COST,
                                         block_h=744, overlap_h=48,
                                         time_limit=300.0, mip_gap=1e-3)
        return record(name, "Perfect-foresight unit-commitment ceiling",
                      res, run, time.time() - t0)

    if name.startswith("roll"):
        # roll24 / roll48 / roll24p / roll48p  (p = perfect forecasts)
        perfect = name.endswith("p")
        horizon = int(name[4:6])
        nan = np.full(len(PVa), np.nan)
        res, run = ct.run_rolling_milp(
            PVa, LDa, PVa if perfect else nan, LDa if perfect else nan,
            soc_init=soc0, horizon=horizon, commit=1,
            cost=COST, hierarchical=True, time_limit=10.0,
            mip_gap=1e-3, progress_every=1000,
            trajectory_at=None if perfect else traj,
            perfect_foresight=perfect)
        return record(name, f"Rolling MILP, {horizon} h horizon, 1 h re-plan, "
                            f"{'perfect' if perfect else 'XGBoost'} forecasts",
                      res, run, time.time() - t0)

    raise SystemExit(f"unknown config {name}")


def collect():
    rows = [json.loads(p.read_text()) for p in sorted(P5.glob("*.json"))
            if not p.name.endswith("_manifest.json")]
    if not rows:
        raise SystemExit("no phase 5 results yet")
    df = pd.DataFrame(rows)
    order = ["oracle", "baseline", "heuristic", "roll24", "roll48",
             "roll24p", "roll48p"]
    df["ord"] = df.config.map({c: i for i, c in enumerate(order)}).fillna(99)
    df = df.sort_values("ord").drop(columns="ord")

    base = df[df.config == "baseline"]
    heur = df[df.config == "heuristic"]
    orc = df[df.config == "oracle"]
    if len(base):
        b = base.total_cost_GHS.iloc[0]
        df["saving_vs_baseline_pct"] = (1 - df.total_cost_GHS / b) * 100
    if len(heur):
        h = heur.total_cost_GHS.iloc[0]
        df["saving_vs_heuristic_pct"] = (1 - df.total_cost_GHS / h) * 100
    if len(orc):
        o = orc.total_cost_GHS.iloc[0]
        df["premium_vs_oracle_pct"] = (df.total_cost_GHS / o - 1) * 100
        bad = df[(df.premium_vs_oracle_pct < -1e-6) & (df.config != "oracle")]
        if len(bad):
            print("\n*** A CONTROLLER CAME IN BELOW THE PERFECT-FORESIGHT "
                  "CEILING ***")
            print(bad[["config", "total_cost_GHS", "premium_vs_oracle_pct"]]
                  .to_string(index=False))
            print("The ceiling is a blocked MILP solved to a 1e-3 gap, so it is "
                  "an achievable upper bound rather than a proven optimum; a "
                  "controller edging below it means the blocking, not the "
                  "controller, is suboptimal.")

    dest = P.results_root() if os.environ.get("MG_RESULTS_ROOT", "").strip() else RESULTS
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / f"phase5_ladder{TAG}.csv", index=False)
    (dest / f"phase5_ladder{TAG}_manifest.json").write_text(
        json.dumps(manifest({"configs": list(df.config), "cost_model": COST.name,
                             "start_cost_ghs": COST.genset.start_cost_ghs,
                             "heuristic": HEURISTIC_BEST,
                             "heuristic_selected_on": HEURISTIC_SOURCE,
                             "fuel_A": COST.genset.a_l_per_kw_rated_h,
                             "fuel_B": COST.genset.b_l_per_kwh}), indent=2))

    show = ["config", "total_cost_GHS", "fuel_litres", "genset_run_hours",
            "genset_starts", "mean_load_factor_pct", "LPSP_pct",
            "curtailed_kWh_per_year"]
    show += [c for c in ("saving_vs_baseline_pct", "saving_vs_heuristic_pct",
                         "premium_vs_oracle_pct") if c in df.columns]
    print("\n" + df[show].to_string(index=False,
                                    float_format=lambda v: f"{v:,.2f}"))
    print(f"\nSaved {dest / f'phase5_ladder{TAG}.csv'}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="comma-separated: baseline,heuristic,oracle,roll24,"
                         "roll48,roll24p,roll48p")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--quick", type=int, default=0, help="first N hours only")
    ap.add_argument("--curve", default=cm.ADOPTED_CURVE,
                    choices=sorted(cm.FUEL_CURVES), help="named fuel curve")
    ap.add_argument("--outdir", default=None, help="subdir of results/ to write")
    ap.add_argument("--tag", default="", help="suffix for the collected csv")
    ap.add_argument("--theta", type=float, default=None,
                    help="heuristic loading target (re-tune per load series)")
    ap.add_argument("--psi", type=float, default=None,
                    help="heuristic banking ceiling")
    ap.add_argument("--start-cost", type=float, default=None,
                    help="GHS per start; default MG_START_COST_GHS, else 150")
    a = ap.parse_args(argv)

    global COST, P5, TAG, HEURISTIC_BEST, HEURISTIC_SOURCE
    TAG = a.tag
    COST = (cm.corrected(a.curve) if a.start_cost is None
            else cm.corrected(a.curve, start_cost_ghs=a.start_cost))
    if a.theta is not None or a.psi is not None:
        if a.theta is None or a.psi is None:
            raise SystemExit("--theta and --psi go together")
        HEURISTIC_BEST = dict(threshold=a.theta, bank_to_soc=a.psi)
        HEURISTIC_SOURCE = "fixed on the command line (diagnostic only)"
    elif a.config and "heuristic" in a.config.split(","):
        from .walkforward import select_heuristic
        th, ps = select_heuristic(2025, COST)
        HEURISTIC_BEST = dict(threshold=th, bank_to_soc=ps)
        HEURISTIC_SOURCE = "2024 full year (walk-forward rule)"
    print(f'heuristic: {HEURISTIC_BEST} ({HEURISTIC_SOURCE})', flush=True)
    if a.outdir:
        P5 = P.results_file(a.outdir)
    print(f'cost model: {COST.name}  c_s={COST.genset.start_cost_ghs:g} GHS  '
          f'A={COST.genset.a_l_per_kw_rated_h:.4f} '
          f'B={COST.genset.b_l_per_kwh:.4f}  -> {P5}', flush=True)

    if a.config:
        for name in a.config.split(","):
            print(f"[{name}] running...", flush=True)
            run_config(name.strip(), quick=a.quick)
    if a.collect or not a.config:
        collect()


if __name__ == "__main__":
    main()
