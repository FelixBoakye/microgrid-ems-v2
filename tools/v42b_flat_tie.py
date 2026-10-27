"""Flat-model tie: full-year LP oracle vs the flat merit-order rule, 2025.

Recomputes the Section V-A numbers under the current code: diesel energy of both,
their flat objectives, and in how many of the 8,760 hours the two dispatches differ.
Active load variant from MG_LOAD (v1 or v2). Writes <MG_RESULTS_ROOT>/flat_tie_load_<v>.json.

    python tools/v42b_flat_tie.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts_v2 import dispatch_core as dc, oracle, paths as P  # noqa: E402
from scripts_v2.run_ladder import soc_start_for_2025  # noqa: E402

TOLS = (1e-9, 1e-6, 1e-3)


def main():
    out = Path(os.environ["MG_RESULTS_ROOT"])
    v = P.variant()
    pv = pd.read_csv(P.DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    d = pv[["datetime", "PV_power_kW"]].merge(ld[["datetime", "total_load_kW"]], on="datetime")
    d = d[d.datetime.dt.year == 2025].sort_values("datetime").reset_index(drop=True)
    n = P.smoke_hours() or len(d)
    PVa, LDa = d.PV_power_kW.values[:n], d.total_load_kW.values[:n]
    soc0 = soc_start_for_2025()
    t0 = time.perf_counter()
    _, orc = oracle.solve_year(PVa, LDa, soc_init=soc0)
    t_lp = time.perf_counter() - t0
    base = dc.baseline(PVa, LDa, soc_init=soc0, uc_parity=False)  # flat model, as in sweeps.replan_sweep
    jo, jb = dc.objective(orc), dc.objective(base)
    dg = np.abs(np.asarray(orc.diesel) - np.asarray(base.diesel))
    rec = {
        "load_variant": v, "hours": int(n), "soc_init_kwh": soc0,
        "oracle_diesel_kwh": float(np.sum(orc.diesel)),
        "baseline_diesel_kwh": float(np.sum(base.diesel)),
        "diesel_difference_kwh": float(np.sum(orc.diesel) - np.sum(base.diesel)),
        "oracle_objective": jo, "baseline_objective": jb,
        "objective_rel_difference": (jb - jo) / jo,
        "objective_oracle_14sf": f"{jo:.13e}", "objective_baseline_14sf": f"{jb:.13e}",
        "hours_diesel_differs": {f"{t:g}": int(np.sum(dg > t)) for t in TOLS},
        "hours_any_flow_differs_1e-6": int(np.sum(
            (dg > 1e-6) | (np.abs(np.asarray(orc.charge) - np.asarray(base.charge)) > 1e-6)
            | (np.abs(np.asarray(orc.discharge) - np.asarray(base.discharge)) > 1e-6))),
        "lp_oracle_full_year_solve_s": t_lp,
        "note": "baseline called with uc_parity=False (flat model), oracle = lp_dispatch full-year LP",
    }
    (out / f"flat_tie_load_{v}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
