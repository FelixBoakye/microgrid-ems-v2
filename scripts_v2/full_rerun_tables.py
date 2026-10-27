"""Table IX structure and the forecast/design split, from a rerun's dispatches.

Banked share is charge above the PV surplus, divided by generator kWh.
Block length and evening starts use the stored running flag.
The 86/14 split is the gap between the direct rolling MILP and the
perfect-foresight ceiling: the part that remains with perfect forecasts is
the design share, and the rest is the forecast share.

    python -m scripts_v2.full_rerun_tables
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import direct_forecast as D
from . import paths as P


CONFIGS = ("baseline", "heuristic", "oracle", "roll24", "roll24p")


def _blocks(running):
    run = np.asarray(running, bool)
    lens, starts = [], []
    t, n = 0, len(run)
    while t < n:
        if run[t]:
            s = t
            while t < n and run[t]:
                t += 1
            lens.append(t - s)
            starts.append(s)
        else:
            t += 1
    return np.asarray(lens, int), np.asarray(starts, int)


def structure_row(charge, diesel, running, pv, load, hours):
    surplus = np.maximum(pv - load, 0.0)
    bank = np.maximum(np.asarray(charge, float) - surplus, 0.0)
    lens, starts = _blocks(running)
    diesel_sum = float(np.asarray(diesel, float).sum())
    return dict(
        banked_pct=float(100 * bank.sum() / diesel_sum) if diesel_sum else float("nan"),
        mean_block_h=float(lens.mean()) if len(lens) else float("nan"),
        blocks_1_2h_pct=(float(100 * np.mean(np.isin(lens, [1, 2])))
                         if len(lens) else float("nan")),
        starts_ge_18_pct=(float(100 * np.mean(hours[starts] >= 18))
                          if len(starts) else float("nan")),
        n_blocks=int(len(lens)),
    )


def _year_series(variant):
    os.environ["MG_LOAD"] = variant
    frame = D.observed_frame()
    idx = D.year_index(frame, 2025)
    yr = P.smoke_trim(frame.iloc[idx])
    return (yr.PV_power_kW.to_numpy(dtype=float),
            yr.total_load_kW.to_numpy(dtype=float),
            pd.DatetimeIndex(yr.datetime).hour.to_numpy())


def table_ix(root: P.Path):
    rows = []
    jobs = (
        ("v2", "sourced_mean", root / "phase5_sourced_mean"),
        ("mix12", "sourced_mean", root / "mix12" / "phase5_sourced_mean"),
    )
    for variant, curve, directory in jobs:
        if not directory.exists():
            print(f"skip table IX {variant}: {directory} is missing", flush=True)
            continue
        pv, load, hours = _year_series(variant)
        for name in CONFIGS:
            path = directory / f"{name}_dispatch.npz"
            if not path.exists():
                continue
            z = np.load(path)
            if len(z["diesel"]) != len(pv):
                raise RuntimeError(f"{path} has {len(z['diesel'])} hours, series has {len(pv)}")
            running = z["running"] if "running" in z.files else z["diesel"] > 1e-9
            rec = structure_row(z["charge"], z["diesel"], running, pv, load, hours)
            rec.update(variant=variant, curve=curve, config=name)
            rows.append(rec)
    out = pd.DataFrame(rows)
    dest = root / "table_ix_structure.csv"
    out.to_csv(dest, index=False)
    print(f"saved {dest}", flush=True)
    return out


def attribution(root: P.Path, ladder="phase5_ladder_lv2_sourced_mean.csv",
                out_name="attribution_86_14.csv"):
    path = root / ladder
    if not path.exists():
        raise SystemExit(f"missing {path}")
    d = pd.read_csv(path).set_index("config")
    o = float(d.loc["oracle", "total_cost_GHS"])
    p = float(d.loc["roll24p", "total_cost_GHS"])
    r = float(d.loc["roll24", "total_cost_GHS"])
    gap = r - o
    if gap == 0:
        raise RuntimeError("direct rolling MILP ties the ceiling, so the split is undefined")
    starts = float(d.loc["roll24", "genset_starts"])
    row = dict(
        start_cost_ghs=(float(d.loc["roll24", "start_cost_GHS"]) / starts
                        if starts else float("nan")),
        oracle_ghs=o,
        perfect_forecast_ghs=p,
        direct_ghs=r,
        perfect_fc_premium_vs_oracle_pct=(p / o - 1) * 100,
        design_share_pct=100 * (p - o) / gap,
        forecast_share_pct=100 * (r - p) / gap,
    )
    out = pd.DataFrame([row])
    dest = root / out_name
    out.to_csv(dest, index=False)
    print(f"design {row['design_share_pct']:.4f}%  "
          f"forecast {row['forecast_share_pct']:.4f}%", flush=True)
    print(f"saved {dest}", flush=True)
    return out


def main():
    import sys
    root = P.results_root()
    if "--attribution-only" in sys.argv:
        i = sys.argv.index("--attribution-only")
        ladder, out_name = sys.argv[i + 1], sys.argv[i + 2]
        os.environ["MG_LOAD"] = "v2"
        attribution(root, ladder, out_name)
        return
    table_ix(root)
    # Table IX switches MG_LOAD. The split file does not depend on it.
    os.environ["MG_LOAD"] = "v2"
    attribution(root)


if __name__ == "__main__":
    main()
