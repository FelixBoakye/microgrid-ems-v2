"""Re-solve the 2025 direct rolling MILP from the saved leak-check forecasts.

Writes under the rerun directory. Does not replace
results/v2/leak_check_2025_2026-10-16/. Exits non-zero unless the cost and
the saving against the saved heuristic match the leak-check direct row.
"""
from __future__ import annotations

import json
import os
import time

os.environ["MG_LOAD"] = "v2"

from pathlib import Path

import numpy as np

from . import controllers as ct
from . import cost_model as cm
from . import leak_check_2025 as L

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v2" / "full_rerun_2026-10-16" / "confirm_2025"
EXPECTED_COST = 228052.38207419496
EXPECTED_SAVING = 14.360043491251517
COST_TOL_GHS = 0.01
SAVING_TOL_PP = 1e-6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = L.observed_frame()
    idx = L.year_iloc(df)
    z = np.load(L.OUT / "direct_pred_2025.npz")
    if z["pv"].shape[1] != len(df):
        raise SystemExit("saved direct predictions do not match the observed frame")
    setup = json.loads((L.OUT / "setup.json").read_text())
    yr = df.iloc[idx]
    PVa = yr.PV_power_kW.to_numpy(dtype=float)
    LDa = yr.total_load_kW.to_numpy(dtype=float)
    cost = cm.corrected("sourced_mean", start_cost_ghs=L.START_GHS)
    heur, _, _ = L.load_dispatch("heuristic")
    heur_c = cost.evaluate(heur)["total_cost_GHS"]
    nan = np.full(len(PVa), np.nan)
    print("re-solving B direct from the saved 2025 predictions", flush=True)
    t0 = time.time()
    res, run = ct.run_rolling_milp(
        PVa, LDa, nan, nan, soc_init=float(setup["soc0"]), cost=cost,
        trajectory_at=L.direct_calls(z["pv"], z["load"], idx), **L.SOLVE_KW)
    e = cost.evaluate(res)
    total = float(e["total_cost_GHS"])
    saving = (1 - total / heur_c) * 100
    rec = dict(total_cost_GHS=total, saving_vs_heuristic_pct=saving,
               expected_cost_GHS=EXPECTED_COST, expected_saving_pct=EXPECTED_SAVING,
               cost_difference_GHS=total - EXPECTED_COST,
               saving_difference_pp=saving - EXPECTED_SAVING,
               fuel_litres=float(e["fuel_litres"]),
               genset_starts=int(e["genset_starts"]),
               seconds=time.time() - t0,
               soc0=float(setup["soc0"]), theta=setup["theta"], psi=setup["psi"])
    (OUT / "confirm.json").write_text(json.dumps(rec, indent=2))
    np.savez_compressed(OUT / "dispatch_B_direct.npz",
                        diesel=res.diesel, soc=res.soc, charge=res.charge,
                        discharge=res.discharge, unmet=res.unmet, curtail=res.curtail,
                        running=np.asarray(run, dtype=bool))
    print(f"cost {total:,.5f} GHS  (expected {EXPECTED_COST:,.5f}, "
          f"diff {total - EXPECTED_COST:+.5f})", flush=True)
    print(f"saving vs heuristic {saving:.8f}%  (expected {EXPECTED_SAVING:.8f})",
          flush=True)
    ok = (abs(total - EXPECTED_COST) <= COST_TOL_GHS
          and abs(saving - EXPECTED_SAVING) <= SAVING_TOL_PP)
    if not ok:
        raise SystemExit("direct 2025 result did not reproduce 228,052 GHS / 14.36%")
    print("CONFIRM_OK", flush=True)


if __name__ == "__main__":
    main()
