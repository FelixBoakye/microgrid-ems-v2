"""Wall-clock timing for every planner — results/timing.csv.

Reported per configuration: mean, median and p95 solve time for a single
planning window, the number of windows a full 8,760-hour year requires at that
configuration's re-planning interval, and the implied annual wall-clock.

Two honesty rules are enforced in the output itself:

  * `annual_wallclock_source` says whether the annual figure was *measured* end
    to end or *extrapolated* from per-window samples. The original GA is far too
    slow to run a full year for a timing number alone (roughly two hours), so it
    is sampled and extrapolated, and the column says so.
  * `machine_contended` records whether other heavy jobs were running. A timing
    run on a loaded machine is not comparable with one on a quiet machine, and
    the numbers in this repository were produced under both conditions.

Usage:  python -m scripts_v2.timing [--ga-windows 50] [--contended]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cost_model as cm
from . import paths as P
from . import ga as ga2
from . import lp_dispatch, milp_dispatch as mp
from .config import DIESEL_MARGINAL_COST, HORIZON, SOC_MIN
from .run_ladder import load_test_year, soc_start_for_2025

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HOURS_PER_YEAR = 8760


def _load_original_ga():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "orig_ga_timing", ROOT / "scripts" / "09_ga_optimizer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig_ga_timing"] = mod
    spec.loader.exec_module(mod)
    return mod


def _env():
    import scipy
    import sklearn
    import xgboost
    try:
        cpu = subprocess.check_output(
            ["wmic", "cpu", "get", "name"], text=True,
            stderr=subprocess.DEVNULL).splitlines()
        cpu = next((l.strip() for l in cpu[1:] if l.strip()), platform.processor())
    except Exception:
        cpu = platform.processor() or "unknown"
    import os
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solver": f"HiGHS via scipy {scipy.__version__}",
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _stats(samples):
    a = np.asarray(samples, float)
    return dict(n_windows_timed=len(a),
                mean_s=float(a.mean()),
                median_s=float(np.median(a)),
                p95_s=float(np.percentile(a, 95)),
                min_s=float(a.min()),
                max_s=float(a.max()))


def benchmark(ga_windows=50, lp_windows=50, milp_windows=50, seed=0):
    """Sample ~50 representative planning windows per configuration.

    Nothing is run over the full year purely to produce a timing number. The one
    exception is the full-year oracle LP, which *is* a single solve and so is
    timed directly rather than extrapolated.
    """
    df = load_test_year()
    PV, LD = df.PV_power_kW.values, df.total_load_kW.values
    PVf, LDf = df.xgb_PV_pred.values, df.xgb_load_pred.values
    rng = np.random.default_rng(seed)
    rows = []

    def sample_starts(n, horizon):
        return rng.integers(0, len(PV) - horizon, n)

    # ---- original GA (sampled, extrapolated) ----
    orig = _load_original_ga() if (ROOT / "scripts" / "09_ga_optimizer.py").exists() else None
    ts = []
    for s in (sample_starts(ga_windows, HORIZON) if orig else []):
        t0 = time.perf_counter()
        orig.optimize(PVf[s:s + HORIZON], LDf[s:s + HORIZON], 65.0, seed=int(s))
        ts.append(time.perf_counter() - t0)
    if orig: rows.append(dict(planner="GA (original, 09_ga_optimizer.py)",
                     replan_interval_h=6, horizon_h=HORIZON,
                     windows_per_year=HOURS_PER_YEAR // 6,
                     annual_wallclock_source="extrapolated", **_stats(ts)))

    # ---- vectorised GA ----
    ts = []
    for s in sample_starts(ga_windows, HORIZON):
        t0 = time.perf_counter()
        ga2.optimize(PVf[s:s + HORIZON], LDf[s:s + HORIZON], 65.0, seed=int(s))
        ts.append(time.perf_counter() - t0)
    rows.append(dict(planner="GA (vectorised, scripts_v2/ga.py)",
                     replan_interval_h=6, horizon_h=HORIZON,
                     windows_per_year=HOURS_PER_YEAR // 6,
                     annual_wallclock_source="extrapolated", **_stats(ts)))

    # ---- LP, 6 h and 1 h ----
    for interval in (6, 1):
        ts = []
        for s in sample_starts(lp_windows, HORIZON):
            t0 = time.perf_counter()
            lp_dispatch.solve(PVf[s:s + HORIZON], LDf[s:s + HORIZON], 65.0,
                              terminal_soc_value=DIESEL_MARGINAL_COST)
            ts.append(time.perf_counter() - t0)
        rows.append(dict(planner="LP (scipy linprog / HiGHS)",
                         replan_interval_h=interval, horizon_h=HORIZON,
                         windows_per_year=HOURS_PER_YEAR // interval,
                         annual_wallclock_source="extrapolated", **_stats(ts)))

    # ---- rolling MILP, 24 h and 48 h ----
    for horizon in (24, 48):
        ts = []
        for s in sample_starts(milp_windows, horizon):
            t0 = time.perf_counter()
            mp.solve(PVf[s:s + horizon], LDf[s:s + horizon], 65.0, cm.CORRECTED,
                     time_limit=10.0, mip_gap=1e-3)
            ts.append(time.perf_counter() - t0)
        rows.append(dict(planner="MILP unit commitment (scipy milp / HiGHS)",
                         replan_interval_h=1, horizon_h=horizon,
                         windows_per_year=HOURS_PER_YEAR,
                         annual_wallclock_source="extrapolated", **_stats(ts)))

    # ---- full-year oracle LP: a single solve, measured ----
    t0 = time.perf_counter()
    lp_dispatch.solve(PV, LD, SOC_MIN)
    dt = time.perf_counter() - t0
    rows.append(dict(planner="LP oracle (single 8,760 h solve)",
                     replan_interval_h=np.nan, horizon_h=HOURS_PER_YEAR,
                     windows_per_year=1,
                     annual_wallclock_source="measured (single direct solve)",
                     **_stats([dt])))

    out = pd.DataFrame(rows)
    out["extrapolated_annual_s"] = out.mean_s * out.windows_per_year
    out["extrapolated_annual_min"] = out.extrapolated_annual_s / 60.0
    return out


def attach_measured_totals(df):
    """Fold in the end-to-end wall clock actually observed for the Phase 5 runs.

    Those runs were executed five-at-a-time, so they are contended and are
    labelled as such; they are not comparable with the quiet per-window samples.
    """
    p5 = (P.results_root() / "phase5_sourced_mean"
          if os.environ.get("MG_RESULTS_ROOT", "").strip()
          else ROOT / "results" / "phase5")
    rows = []
    for name, horizon in (("roll24", 24), ("roll48", 48),
                          ("roll24p", 24), ("roll48p", 48)):
        f = p5 / f"{name}.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        rows.append(dict(
            planner=f"rolling MILP end-to-end ({name})",
            replan_interval_h=1, horizon_h=horizon, windows_per_year=HOURS_PER_YEAR,
            annual_wallclock_source="measured (contended: 5 jobs in parallel)",
            n_windows_timed=HOURS_PER_YEAR,
            mean_s=j["seconds"] / HOURS_PER_YEAR,
            median_s=np.nan, p95_s=np.nan, min_s=np.nan, max_s=np.nan,
            extrapolated_annual_s=j["seconds"],
            extrapolated_annual_min=j["seconds"] / 60.0))
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if rows else df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga-windows", type=int, default=50)
    ap.add_argument("--lp-windows", type=int, default=50)
    ap.add_argument("--milp-windows", type=int, default=50)
    ap.add_argument("--contended", action="store_true",
                    help="record that other heavy jobs were running")
    a = ap.parse_args(argv)

    out = P.results_root() if os.environ.get("MG_RESULTS_ROOT", "").strip() else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    df = benchmark(a.ga_windows, a.lp_windows, a.milp_windows)
    df = attach_measured_totals(df)
    df.insert(0, "machine_contended", a.contended)
    env = _env()
    for k, v in env.items():
        df[k] = v
    df.to_csv(out / "timing.csv", index=False)

    from .run_ladder import manifest as _man
    (out / "timing_manifest.json").write_text(json.dumps(_man({
        "hardware": {"cpu": env["cpu"], "logical_cores": env["logical_cores"],
                     "platform": env["platform"]},
        "solvers": {"HiGHS": f"via scipy {env['scipy']}",
                    "scipy": env["scipy"], "numpy": env["numpy"],
                    "xgboost": env["xgboost"],
                    "scikit_learn": env["scikit_learn"],
                    "python": env["python"]},
        "sampling": {"ga_windows": a.ga_windows, "lp_windows": a.lp_windows,
                     "milp_windows": a.milp_windows,
                     "note": "Per-window samples; annual figures are "
                             "extrapolated as mean_s * windows_per_year. "
                             "Nothing was run over the full year purely to "
                             "measure time. The full-year oracle LP is a single "
                             "solve and is timed directly."},
        "machine_contended": a.contended,
    }), indent=2))

    show = ["planner", "horizon_h", "replan_interval_h", "n_windows_timed",
            "mean_s", "median_s", "p95_s", "windows_per_year",
            "extrapolated_annual_min", "annual_wallclock_source"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\n{env['cpu']} | {env['logical_cores']} logical cores | "
          f"{env['solver']}")
    print(f"machine_contended={a.contended}")
    print(f"Saved {RESULTS / 'timing.csv'}")
    return df


if __name__ == "__main__":
    main()
