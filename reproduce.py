"""Single entry point: regenerate every table in the manuscript from raw inputs.

    python reproduce.py --list              show the stages and their runtimes
    python reproduce.py --stage data        stages 1-3, ~2 minutes
    python reproduce.py --all               everything, ~6 hours
    python reproduce.py --verify            checks only, ~2 minutes

Nothing here reads a stored result to produce another result: every stage runs
from the raw NASA POWER CSV forward. The one exception is documented — the
bootstrap re-costs persisted hourly dispatch rather than re-running dispatch,
which is the point of persisting it.

The original study's headline table could not be regenerated from its own code,
which is why this file exists and why `tools/check_no_hardcoded_results.py` runs
as part of `--verify`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

#  name          env            command                                    minutes
STAGES = [
    ("pv",        {},            ["repro/run_original.py", "01"],              1,
     "PV model from the raw NASA POWER CSV"),
    ("load_v1",   {},            ["repro/run_original.py", "02", "03"],        1,
     "published demand series + rule-based baseline (frozen originals)"),
    ("dataset_v1", {},           ["-m", "scripts_v2.build_dataset"],           1,
     "D5-corrected feature matrix, load_v1"),
    ("forecast_v1", {},          ["-m", "scripts_v2.forecast"],                2,
     "XGBoost + RF + quantile forecasters, load_v1"),
    ("load_v2",   {},            ["-m", "scripts_v2.make_load_v2"],            1,
     "AR(1) demand series with weekday and temperature structure"),
    ("dataset_v2", {"MG_LOAD": "v2"}, ["-m", "scripts_v2.build_dataset"],      1,
     "feature matrix, load_v2"),
    ("forecast_v2", {"MG_LOAD": "v2"}, ["-m", "scripts_v2.forecast"],          2,
     "forecasters, load_v2"),
    ("invariants", {"MG_LOAD": "v2"}, ["-m", "scripts_v2.check_invariants"],   15,
     "the three structural invariants, on direct forecasts"),
    ("ladder_flat", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.run_ladder", "--rows", "0,1,2,3,4,5,6,7,8,9,10,10p",
      "--out", "v2/attribution_ladder_loadv2.csv"],                           90,
     "flat-model attribution ladder, rows 0-10 and 10p, direct forecasts"),
    ("replan",    {"MG_LOAD": "v2"}, ["-m", "scripts_v2.sweeps", "--what", "replan"], 20,
     "re-planning interval sweep, FLAT cost model (as published), direct forecasts"),
    ("replan_corrected", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.sweeps", "--what", "replan_corrected"],                 90,
     "re-planning interval sweep, FULL cost model at MG_START_COST_GHS"),
    ("phase5",    {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.run_phase5", "--curve", "sourced_mean",
      "--outdir", "v2/phase5_sourced_mean", "--tag", "_lv2_sourced_mean",
      "--config", "baseline,heuristic,oracle,roll24,roll24p", "--collect"],  90,
     "corrected-cost ladder, load_v2, direct forecasts"),
    ("phase5_generac", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.run_phase5", "--curve", "generac_sd040",
      "--outdir", "v2/phase5_generac_sd040", "--tag", "_lv2_generac",
      "--config", "baseline,heuristic,oracle,roll24", "--collect"],          40,
     "Generac ladder, load_v2"),
    ("phase5_cummins", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.run_phase5", "--curve", "cummins_c40d6",
      "--outdir", "v2/phase5_cummins_c40d6", "--tag", "_lv2_cummins",
      "--config", "baseline,heuristic,oracle,roll24", "--collect"],          40,
     "Cummins ladder, load_v2"),
    ("phase5_kohler", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.run_phase5", "--curve", "kohler_40reozk",
      "--outdir", "v2/phase5_kohler_40reozk", "--tag", "_lv2_kohler",
      "--config", "baseline,heuristic,oracle,roll24", "--collect"],          40,
     "Kohler ladder, load_v2"),
    ("phase5_mix12", {"MG_LOAD": "mix12"},
     ["-m", "scripts_v2.run_phase5", "--curve", "sourced_mean",
      "--outdir", "phase5_sourced_mean", "--tag", "_mix12",
      "--config", "baseline,heuristic,oracle,roll24", "--collect"],          40,
     "mix12 ladder, heuristic selected on 2024, direct forecasts"),
    ("phase5_loadv1", {"MG_LOAD": "v1"},
     ["-m", "scripts_v2.run_phase5", "--curve", "sourced_mean",
      "--outdir", "phase5_sourced_mean", "--tag", "_loadv1",
      "--config", "baseline,heuristic,oracle,roll24", "--collect"],          40,
     "load_v1 ladder, direct forecasts"),
    ("forecast_direct", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "forecast_direct"],          60,
     "direct per-horizon forecasters, 2022-2025"),
    ("skill", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "skill"],                     2,
     "nRMSE by lead 1-24 versus persistence and seasonal-naive"),
    ("walkforward_fc", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "forecast"],                  4,
     "1-step walk-forward forecasters, kept only as a skill table"),
    ("walkforward",  {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "dispatch",
      "--start-cost", "33.75"],                                              240,
     "walk-forward at 33.75 GHS/start, direct and seasonal-naive, four years"),
    ("walkforward_collect", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "collect"],                   1,
     "walk-forward summary at 33.75 GHS/start"),
    ("seasonal",  {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "seasonal",
      "--start-cost", "33.75"],                                                1,
     "seasonal disaggregation at 33.75 GHS/start, all four years"),
    ("walkforward_50", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "dispatch",
      "--start-cost", "50.625", "--out-subdir", "walkforward_cs50p625"],    240,
     "walk-forward at 50.625 GHS/start"),
    ("walkforward_50_collect", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "collect",
      "--out-subdir", "walkforward_cs50p625"],                                 1,
     "walk-forward summary at 50.625 GHS/start"),
    ("seasonal_50", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.walkforward", "--stage", "seasonal",
      "--start-cost", "50.625", "--out-subdir", "walkforward_cs50p625"],       1,
     "seasonal disaggregation at 50.625 GHS/start"),
    ("sens_linear", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.sensitivity", "--stage", "linear"],                    1,
     "closed-form sensitivity, battery and start cost"),
    ("sens_window", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.sensitivity", "--stage", "validate-window"],          20,
     "nominal 3-month window on direct forecasts"),
    ("sens_genset", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.sensitivity", "--stage", "genset"],                   80,
     "fuel curve, minimum load, minimum up/down"),
    ("sens_demand", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.sensitivity2", "--stage", "demand"],                 180,
     "phi, day-of-week, temperature response, direct models"),
    ("sizing",    {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.sensitivity2", "--stage", "sizing"],                  80,
     "battery 65-260 kWh x PV 60-120 kWp"),
    ("seeds",     {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.sensitivity2", "--stage", "seeds"],                  240,
     "ten forecaster seeds, full count, direct models"),
    ("bootstrap", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.bootstrap", "--source", "fold", "--year", "2025",
      "--start-cost", "33.75"],                                                2,
     "block-bootstrap 95% interval at 33.75 GHS/start"),
    ("bootstrap_50", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.bootstrap", "--source", "fold", "--year", "2025",
      "--start-cost", "50.625", "--fold-subdir", "walkforward_cs50p625",
      "--out-name", "bootstrap_ci_cs50p625.csv"],                               2,
     "block-bootstrap 95% interval at 50.625 GHS/start"),
    ("tables", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.full_rerun_tables"],                                    1,
     "Table IX structure and the forecast/design split"),
    ("week_trace", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.week_trace"],                                           1,
     "representative week trace from the phase-5 dispatches"),
    ("figures", {"MG_LOAD": "v2"},
     ["-m", "scripts_v2.make_figures"],                                         2,
     "figures 1-6 from the rerun"),
]

# Inputs the leak audit says not to regenerate. They also write data/.
SKIP_ON_ALL = {"pv", "load_v1", "dataset_v1", "forecast_v1",
               "load_v2", "dataset_v2", "forecast_v2", "walkforward_fc"}
RERUN_ROOT = ROOT / "results" / "v2" / "full_rerun_2026-10-16"
GATE = ["-m", "pytest", "tests/test_no_lookahead.py",
        "tests/test_direct_forecast.py", "-q", "--tb=short"]

DATA_STAGES = {"pv", "load_v1", "dataset_v1"}


def run(stage):
    name, env, cmd, mins, desc = stage
    import os
    e = dict(os.environ, **env)
    full = [PY] + cmd
    print(f"\n=== {name} ({desc}) — expect ~{mins} min ===", flush=True)
    t0 = time.time()
    r = subprocess.run(full, cwd=ROOT, env=e)
    dt = (time.time() - t0) / 60
    print(f"=== {name} {'OK' if r.returncode == 0 else 'FAILED'} in {dt:.1f} min ===",
          flush=True)
    return r.returncode


def verify():
    print("=== verification ===")
    rc = subprocess.run([PY, "tools/check_no_hardcoded_results.py"], cwd=ROOT).returncode
    rc |= subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT).returncode
    return rc


def gate():
    import os
    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT)
    e.setdefault("MG_LOAD", "v2")
    print("=== test gate: test_no_lookahead.py and test_direct_forecast.py ===",
          flush=True)
    r = subprocess.run([PY] + GATE, cwd=ROOT, env=e)
    print(f"=== test gate {'OK' if r.returncode == 0 else 'FAILED'} ===", flush=True)
    return r.returncode


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stage", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.list:
        total = sum(s[3] for s in STAGES)
        print(f"{'stage':<22}{'min':>5}  description")
        for n, _, _, m, d in STAGES:
            print(f"{n:<22}{m:>5}  {d}")
        print(f"{'TOTAL':<22}{total:>5}  (~{total / 60:.1f} hours)")
        return 0
    if a.verify:
        return verify()

    # Every stage launched from here writes under the dated rerun tree.
    # A caller-supplied MG_RESULTS_ROOT is kept so this rerun does not rewrite
    # results/v2/full_rerun_2026-10-16.
    if not os.environ.get("MG_RESULTS_ROOT", "").strip():
        os.environ["MG_RESULTS_ROOT"] = str(RERUN_ROOT)
    target = Path(os.environ["MG_RESULTS_ROOT"]).resolve()
    for old in ("full_rerun_2026-10-16", "v42_rerun_2026-10-18c",
                "nominal_33p75_2026-10-16", "nominal_33p75_direct_2026-10-16"):
        if old in target.parts:
            print(f"refusing to write the older result folder {target}")
            return 1
    target.mkdir(parents=True, exist_ok=True)

    if a.all:
        if gate():
            print("\nstopping: test gate failed")
            return 1
        todo = [s for s in STAGES if s[0] not in SKIP_ON_ALL]
        print("skipping data and 1-step forecast stages. "
              "Those inputs stay as they are.", flush=True)
    elif a.stage:
        todo = [s for s in STAGES if s[0] in set(a.stage)]
    else:
        todo = [s for s in STAGES if s[0] in DATA_STAGES]
    if not todo:
        print("nothing to do; try --list")
        return 1
    for s in todo:
        if run(s):
            print(f"\nstopping: stage {s[0]} failed")
            return 1
    return verify() if a.all else 0


if __name__ == "__main__":
    sys.exit(main())
