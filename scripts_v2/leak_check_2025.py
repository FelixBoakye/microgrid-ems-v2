"""Size the rolling-MILP look-ahead leak on the 2025 fold only.

Writes results/v2/leak_check_2025_2026-10-16/ and
results/v2/LEAK_CHECK_2025_2026-10-16.md. Does not run the full pipeline,
does not edit the manuscript, and does not overwrite canonical CSVs.

    python -m scripts_v2.leak_check_2025
    python -m scripts_v2.leak_check_2025 --stage train
    python -m scripts_v2.leak_check_2025 --stage dispatch
    python -m scripts_v2.leak_check_2025 --stage report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

os.environ["MG_LOAD"] = "v2"

from pathlib import Path

import numpy as np
import pandas as pd

from . import controllers as ct
from . import cost_model as cm
from . import direct_forecast as D
from . import dispatch_core as dc
from . import heuristic as H
from . import milp_dispatch as mp
from . import paths as P
from . import walkforward as wf
from .forecast import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v2" / "leak_check_2025_2026-10-16"
REPORT = ROOT / "results" / "v2" / "LEAK_CHECK_2025_2026-10-16.md"
YEAR = 2025
START_GHS = 33.75
REVISION = 1
SAVED_FOLD = (ROOT / "results" / "v2" / "start_cost_sourced_2026-10-15"
              / "usd3_liu2017" / "fold_2025.csv")
CANON = {
    "walkforward_summary": ROOT / "results" / "v2" / "walkforward" / "walkforward_summary.csv",
    "forecasts_2025": ROOT / "results" / "v2" / "walkforward" / "forecasts_2025.csv",
    "bootstrap_ci": ROOT / "results" / "v2" / "bootstrap_ci.csv",
}
CONTROLLERS = (
    "baseline", "heuristic", "oracle",
    "A_leaky", "B_direct", "C_seasonal",
)
SOLVE_KW = dict(horizon=24, commit=1, hierarchical=True,
                time_limit=10.0, mip_gap=1e-3, progress_every=1000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon_hashes() -> dict:
    return {k: sha256(p) for k, p in CANON.items()}


def observed_frame() -> pd.DataFrame:
    pv = pd.read_csv(ROOT / "data" / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    df = pv[["datetime", "PV_power_kW", "GHI_Wm2", "T2M", "CLRSKY_SFC_SW_DWN"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    step = df.datetime.diff().iloc[1:]
    if not bool((step == pd.Timedelta(hours=1)).all()):
        raise RuntimeError("the merged series is not hourly throughout")
    D.assert_lead1_matches_dataset_builder(df)
    return df


def year_iloc(df: pd.DataFrame) -> np.ndarray:
    idx = np.flatnonzero(df.datetime.dt.year.to_numpy() == YEAR)
    if len(idx) != 8760:
        raise RuntimeError(f"{YEAR} has {len(idx)} hours")
    if int(idx[-1]) != len(df) - 1:
        raise RuntimeError(f"{YEAR} is not the last year of the series")
    if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
        raise RuntimeError(f"{YEAR} is not a contiguous block")
    return idx


def train_and_predict(df: pd.DataFrame) -> tuple:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "direct_pred_2025.npz"
    if path.exists():
        z = np.load(path)
        if int(z["revision"]) == REVISION and z["pv"].shape[1] == len(df):
            print(f"reuse {path.name}", flush=True)
            return z["pv"], z["load"], json.loads(Path(OUT / "train_notes.json").read_text())
    print("training direct models", flush=True)
    t0 = time.time()
    models, notes = D.fit_walkforward(df, YEAR)
    pv, load = D.predict_leads(models, df, YEAR)
    idx = year_iloc(df)
    fc = pd.read_csv(CANON["forecasts_2025"], parse_dates=["datetime"])
    got = df.loc[idx, ["datetime"]].reset_index(drop=True)
    got["pv_k1"] = pv[1, idx]
    got["load_k1"] = load[1, idx]
    merged = got.merge(fc[["datetime", "xgb_PV_pred", "xgb_load_pred"]], on="datetime")
    if len(merged) != 8760:
        raise RuntimeError("k=1 predictions did not align to forecasts_2025.csv")
    mae_pv = float(np.mean(np.abs(merged.pv_k1 - merged.xgb_PV_pred)))
    mae_ld = float(np.mean(np.abs(merged.load_k1 - merged.xgb_load_pred)))
    print(f"  lead-1 vs stored 1-step forecast  PV MAE {mae_pv:.6f} kW  "
          f"load MAE {mae_ld:.6f} kW", flush=True)
    if mae_pv > 1.0 or mae_ld > 1.0:
        raise RuntimeError(
            "lead-1 direct model disagrees with the stored 1-step forecast; "
            "the feature build does not match the model that produced it")
    notes.append(dict(lead=1, target="agreement", mae_pv_kW=mae_pv, mae_load_kW=mae_ld))
    (OUT / "train_notes.json").write_text(json.dumps(notes, indent=2))
    np.savez_compressed(path, pv=pv, load=load, revision=np.int64(REVISION))
    print(f"training done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return pv, load, notes


def skill_table(df, pred_pv, pred_load) -> pd.DataFrame:
    idx = year_iloc(df)
    pv = df.PV_power_kW.to_numpy(dtype=float)
    ld = df.total_load_kW.to_numpy(dtype=float)
    rows = []
    for k in (1, 6, 12, 24):
        for target, actual, pred in (
                ("PV", pv, pred_pv), ("load", ld, pred_load)):
            y = actual[idx]
            hats = {
                "direct": pred[k, idx],
                "persistence": actual[idx - k],
                "seasonal_naive": actual[idx - 24],
            }
            if k == 24 and not np.allclose(hats["persistence"], hats["seasonal_naive"]):
                raise RuntimeError(f"{target}: 24-hour persistence is not seasonal-naive")
            for method, hat in hats.items():
                if not np.isfinite(hat).all():
                    raise RuntimeError(f"non-finite {method} {target} lead {k}")
                m = metrics(y, hat)
                rows.append(dict(target=target, lead_h=k, method=method,
                                 mae_kW=m["mae"], nrmse_pct=m["nrmse_pct"],
                                 n=int(len(y))))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "skill_by_horizon.csv", index=False)
    return out


def direct_calls(pred_pv, pred_load, idx):
    def trajectory_at(i):
        abs_i = int(idx[i])
        h = min(D.HORIZON, len(idx) - i)
        return (D.assemble_trajectory(pred_pv, abs_i, horizon=h),
                D.assemble_trajectory(pred_load, abs_i, horizon=h))
    return trajectory_at


def seasonal_calls(pv, load, idx):
    def trajectory_at(i):
        abs_i = int(idx[i])
        h = min(D.HORIZON, len(idx) - i)
        src = D.seasonal_source_indices(abs_i, len(pv))[:h]
        if int(src.max()) > abs_i - 1:
            raise RuntimeError("seasonal trajectory reads past the origin")
        return pv[src], load[src]
    return trajectory_at


def check_plan_times(idx, n):
    for i in (0, 1, 23, 100, 4000, 8750, 8759):
        decision = int(idx[i])
        origin = decision - 1
        used = D.plan_source_indices(decision, n)
        if not used or max(used) != origin or decision in used or min(used) < 0:
            raise RuntimeError(f"plan at year hour {i} reads outside the origin")
        seas = D.seasonal_source_indices(decision, n)
        # A full 24 h window ends on the origin. A shorter tail ends earlier.
        if int(seas.max()) > origin or int(seas.min()) < 0 or decision in set(seas.tolist()):
            raise RuntimeError(f"seasonal plan at year hour {i} reads outside the origin")


def save_dispatch(name, res, running, when):
    path = OUT / f"dispatch_{name}.npz"
    np.savez_compressed(
        path,
        soc=res.soc, charge=res.charge, discharge=res.discharge, diesel=res.diesel,
        unmet=res.unmet, curtail=res.curtail, pv=res.pv, load=res.load,
        running=np.asarray(running, dtype=bool),
        datetime=when.astype("datetime64[ns]").astype(np.int64),
    )
    return path


def load_dispatch(name):
    z = np.load(OUT / f"dispatch_{name}.npz")
    res = dc.SimResult(
        soc=z["soc"], charge=z["charge"], discharge=z["discharge"], diesel=z["diesel"],
        unmet=z["unmet"], curtail=z["curtail"], pv=z["pv"], load=z["load"])
    when = z["datetime"].astype("datetime64[ns]")
    return res, np.asarray(z["running"], dtype=bool), when


def run_controllers(df, pred_pv, pred_load):
    OUT.mkdir(parents=True, exist_ok=True)
    idx = year_iloc(df)
    check_plan_times(idx, len(df))
    ref = wf.full_series()
    ref_y = ref[ref.datetime.dt.year == YEAR].reset_index(drop=True)
    yr = df.iloc[idx].reset_index(drop=True)
    if not np.allclose(ref_y.PV_power_kW.to_numpy(), yr.PV_power_kW.to_numpy(), atol=0.0, rtol=0.0):
        raise RuntimeError("PV series disagrees with walkforward.full_series")
    if not np.allclose(ref_y.total_load_kW.to_numpy(), yr.total_load_kW.to_numpy(), atol=0.0, rtol=0.0):
        raise RuntimeError("load series disagrees with walkforward.full_series")
    PVa = yr.PV_power_kW.to_numpy(dtype=float)
    LDa = yr.total_load_kW.to_numpy(dtype=float)
    when = yr.datetime.to_numpy()
    fc = pd.read_csv(CANON["forecasts_2025"], parse_dates=["datetime"])
    merged = yr.merge(fc[["datetime", "xgb_PV_pred", "xgb_load_pred"]], on="datetime", how="left")
    if merged[["xgb_PV_pred", "xgb_load_pred"]].isna().any().any():
        raise RuntimeError("leaky forecast file does not cover every 2025 hour")
    PVf = merged.xgb_PV_pred.to_numpy(dtype=float)
    LDf = merged.xgb_load_pred.to_numpy(dtype=float)

    cost = cm.corrected("sourced_mean", start_cost_ghs=START_GHS)
    if abs(cost.genset.start_cost_ghs - START_GHS) > 0:
        raise RuntimeError("start cost did not stick")
    soc0 = wf.soc_at_year_start(YEAR)
    theta, psi = wf.select_heuristic(YEAR, cost)
    print(f"heuristic on {YEAR - 1}: theta={theta:.0%} psi={psi:.0%}  soc0={soc0:.6f}",
          flush=True)
    (OUT / "setup.json").write_text(json.dumps(dict(
        year=YEAR, theta=theta, psi=psi, soc0=soc0, start_cost_ghs=START_GHS,
        curve="sourced_mean",
        fuel_A=cost.genset.a_l_per_kw_rated_h,
        fuel_B=cost.genset.b_l_per_kwh,
        timing=D.TIMING_CONVENTION,
    ), indent=2))

    def have(name):
        return (OUT / f"dispatch_{name}.npz").exists()

    if not have("baseline"):
        print("baseline", flush=True)
        res = dc.baseline(PVa, LDa, soc_init=soc0, genset=cost.genset)
        save_dispatch("baseline", res, res.diesel > 1e-9, when)
    if not have("heuristic"):
        print("heuristic", flush=True)
        res, run = H.commitment_heuristic(
            PVa, LDa, soc_init=soc0, threshold=theta, bank_to_soc=psi, genset=cost.genset)
        save_dispatch("heuristic", res, run, when)
    if not have("oracle"):
        print("oracle", flush=True)
        t0 = time.time()
        res, run = mp.solve_year_blocked(
            PVa, LDa, soc0, cost, block_h=744, overlap_h=48,
            time_limit=300.0, mip_gap=1e-3, verbose=True)
        save_dispatch("oracle", res, run, when)
        print(f"oracle done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not have("A_leaky"):
        print("A leaky rolling MILP", flush=True)
        t0 = time.time()
        res, run = ct.run_rolling_milp(
            PVa, LDa, PVf, LDf, soc_init=soc0, cost=cost, **SOLVE_KW)
        save_dispatch("A_leaky", res, run, when)
        print(f"A done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not have("B_direct"):
        print("B direct rolling MILP", flush=True)
        t0 = time.time()
        nan = np.full(len(PVa), np.nan)
        res, run = ct.run_rolling_milp(
            PVa, LDa, nan, nan, soc_init=soc0, cost=cost,
            trajectory_at=direct_calls(pred_pv, pred_load, idx), **SOLVE_KW)
        save_dispatch("B_direct", res, run, when)
        print(f"B done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not have("C_seasonal"):
        print("C seasonal-naive rolling MILP", flush=True)
        t0 = time.time()
        nan = np.full(len(PVa), np.nan)
        pv = df.PV_power_kW.to_numpy(dtype=float)
        ld = df.total_load_kW.to_numpy(dtype=float)
        res, run = ct.run_rolling_milp(
            PVa, LDa, nan, nan, soc_init=soc0, cost=cost,
            trajectory_at=seasonal_calls(pv, ld, idx), **SOLVE_KW)
        save_dispatch("C_seasonal", res, run, when)
        print(f"C done in {(time.time() - t0) / 60:.1f} min", flush=True)


def blocks_of(running):
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


def structure_of(res, running, hours):
    surplus = np.maximum(res.pv - res.load, 0.0)
    bank = np.maximum(res.charge - surplus, 0.0)
    lens, starts = blocks_of(running)
    diesel = float(res.diesel.sum())
    return dict(
        banked_pct=float(100 * bank.sum() / diesel) if diesel else float("nan"),
        mean_block_h=float(lens.mean()) if len(lens) else float("nan"),
        blocks_1_2h_pct=float(100 * np.mean(np.isin(lens, [1, 2]))) if len(lens) else float("nan"),
        starts_ge_18_pct=float(100 * np.mean(hours[starts] >= 18)) if len(starts) else float("nan"),
        n_blocks=int(len(lens)),
        commitment_run_h=int(lens.sum()) if len(lens) else 0,
        banked_kwh=float(bank.sum()),
    ), lens, starts, bank


def score_all():
    cost = cm.corrected("sourced_mean", start_cost_ghs=START_GHS)
    setup = json.loads((OUT / "setup.json").read_text())
    packed = {}
    for name in CONTROLLERS:
        packed[name] = load_dispatch(name)
    base_c = cost.evaluate(packed["baseline"][0])["total_cost_GHS"]
    heur_c = cost.evaluate(packed["heuristic"][0])["total_cost_GHS"]
    orc_c = cost.evaluate(packed["oracle"][0])["total_cost_GHS"]
    rows = []
    for name, (res, _run, _when) in packed.items():
        m = res.metrics(hours_per_year=len(res.load))
        e = cost.evaluate(res)
        total = e["total_cost_GHS"]
        rows.append(dict(
            controller=name, theta=setup["theta"], psi=setup["psi"],
            start_cost_ghs=START_GHS, **m,
            **{k: v for k, v in e.items() if k != "cost_model"},
            saving_vs_baseline_pct=(1 - total / base_c) * 100,
            saving_vs_heuristic_pct=(1 - total / heur_c) * 100,
            premium_vs_oracle_pct=(total / orc_c - 1) * 100,
            gap_closed_pct=(heur_c - total) / (heur_c - orc_c) * 100,
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    saved = pd.read_csv(SAVED_FOLD).set_index("config")
    compare_rows = []
    for mine, theirs in (("baseline", "baseline"), ("heuristic", "heuristic"),
                         ("oracle", "oracle"), ("A_leaky", "roll24")):
        for col in ("total_cost_GHS", "fuel_litres", "genset_run_hours", "genset_starts",
                    "saving_vs_heuristic_pct"):
            compare_rows.append(dict(
                controller=mine, column=col,
                this_run=float(summary.set_index("controller").loc[mine, col]),
                saved_33p75=float(saved.loc[theirs, col]),
                difference=float(summary.set_index("controller").loc[mine, col]
                                 - saved.loc[theirs, col]),
            ))
    pd.DataFrame(compare_rows).to_csv(OUT / "compare_to_saved_33p75.csv", index=False)

    res, running, when = packed["B_direct"]
    hours = pd.DatetimeIndex(when).hour.to_numpy()
    rec, lens, starts, bank = structure_of(res, running, hours)
    pd.DataFrame([rec]).to_csv(OUT / "table_ix_B.csv", index=False)
    times = pd.DatetimeIndex(when)
    blocks = pd.DataFrame({
        "start_time": times[starts].astype(str),
        "start_hour": hours[starts],
        "length_h": lens,
        "diesel_kwh": [float(res.diesel[s:s + L].sum()) for s, L in zip(starts, lens)],
        "banked_kwh": [float(bank[s:s + L].sum()) for s, L in zip(starts, lens)],
    })
    blocks.to_csv(OUT / "blocks_B.csv", index=False)
    hist = (blocks.groupby("start_hour").size()
            .reindex(range(24), fill_value=0).rename("n_starts").reset_index())
    hist["share_pct"] = 100 * hist.n_starts / hist.n_starts.sum()
    hist.to_csv(OUT / "start_hours_B.csv", index=False)
    length_hist = (blocks.groupby("length_h").size().rename("n_blocks").reset_index())
    length_hist["share_pct"] = 100 * length_hist.n_blocks / length_hist.n_blocks.sum()
    length_hist.to_csv(OUT / "block_lengths_B.csv", index=False)

    others = []
    for name in CONTROLLERS:
        r, run, w = packed[name]
        hh = pd.DatetimeIndex(w).hour.to_numpy()
        row, _, _, _ = structure_of(r, run, hh)
        row["controller"] = name
        others.append(row)
    pd.DataFrame(others).to_csv(OUT / "table_ix_all.csv", index=False)
    return summary


def _fmt(v, spec):
    return format(float(v), spec)


def write_report(summary: pd.DataFrame, skill: pd.DataFrame):
    s = summary.set_index("controller")
    a = s.loc["A_leaky"]
    b = s.loc["B_direct"]
    c = s.loc["C_seasonal"]
    h = s.loc["heuristic"]
    o = s.loc["oracle"]
    base = s.loc["baseline"]
    delta = float(b.saving_vs_heuristic_pct - a.saving_vs_heuristic_pct)
    if delta < 0:
        move = f"smaller by {abs(delta):.6f} percentage points"
    elif delta > 0:
        move = f"larger by {delta:.6f} percentage points"
    else:
        move = "unchanged"
    ix = pd.read_csv(OUT / "table_ix_B.csv").iloc[0]
    hours = pd.read_csv(OUT / "start_hours_B.csv")
    lengths = pd.read_csv(OUT / "block_lengths_B.csv")
    notes = json.loads((OUT / "train_notes.json").read_text())
    agree = next(n for n in notes if n.get("target") == "agreement")
    cmp = pd.read_csv(OUT / "compare_to_saved_33p75.csv")
    setup = json.loads((OUT / "setup.json").read_text())

    def line(name, row):
        return (f"| {name} | {_fmt(row.total_cost_GHS, ',.2f')} | "
                f"{_fmt(row.fuel_litres, ',.2f')} | {int(row.genset_run_hours)} | "
                f"{int(row.genset_starts)} | {_fmt(row.saving_vs_heuristic_pct, '.4f')} | "
                f"{_fmt(row.saving_vs_baseline_pct, '.4f')} | "
                f"{_fmt(row.premium_vs_oracle_pct, '.4f')} | "
                f"{_fmt(row.gap_closed_pct, '.4f')} |")

    skill_lines = [
        "| target | lead h | method | MAE kW | nRMSE % |",
        "|---|---:|---|---:|---:|",
    ]
    for _, r in skill.sort_values(["target", "lead_h", "method"]).iterrows():
        skill_lines.append(
            f"| {r.target} | {int(r.lead_h)} | {r.method} | "
            f"{r.mae_kW:.4f} | {r.nrmse_pct:.4f} |")

    hour_bits = ", ".join(
        f"{int(r.start_hour):02d}:00 {int(r.n_starts)}"
        for _, r in hours.iterrows() if r.n_starts)
    length_bits = ", ".join(
        f"{int(r.length_h)} h x {int(r.n_blocks)}" for _, r in lengths.iterrows())

    a_cost_diff = float(cmp[(cmp.controller == "A_leaky") & (cmp.column == "total_cost_GHS")].difference.iloc[0])
    h_cost_diff = float(cmp[(cmp.controller == "heuristic") & (cmp.column == "total_cost_GHS")].difference.iloc[0])
    o_cost_diff = float(cmp[(cmp.controller == "oracle") & (cmp.column == "total_cost_GHS")].difference.iloc[0])
    base_diff = float(cmp[(cmp.controller == "baseline") & (cmp.column == "total_cost_GHS")].difference.iloc[0])

    text = f"""# Look-ahead check on the 2025 fold, 2026-10-16

On the 2025 fold the leaky rolling MILP saves {a.saving_vs_heuristic_pct:.6f}% against the heuristic. With direct forecasts that saving is {b.saving_vs_heuristic_pct:.6f}%. The MILP-versus-heuristic saving is {move}.

That is {a.total_cost_GHS:,.2f} GHS under the leaky forecasts and {b.total_cost_GHS:,.2f} GHS under the direct forecasts. Fuel moves from {a.fuel_litres:,.2f} L to {b.fuel_litres:,.2f} L. Realised starts move from {int(a.genset_starts)} to {int(b.genset_starts)}.

## When a measurement is known

The rolling MILP decides at the start of an hour and then executes that hour on the realised PV and load. The state passed into the planner is the state at the end of the previous hour. Realised PV, load, GHI and temperature of the hour about to be executed are not known.

In `run_rolling_milp` the loop index is that hour about to be executed. The planning origin t0 used by the direct forecaster is the previous hour, the last hour with measurements in hand. Data through t0 is known. Data after t0 is not. The plan covers t0+1 through t0+24, and hour t0+1 is the hour that is executed.

A stored 1-step forecast of hour h is fit with actuals through h-1. The leaky planner takes the slice of those forecasts over the next 24 hours. The entry for hour t0+1 is causal. The entry for hour t0+2 is fit with the actual at t0+1, which is after the origin, and each later entry reaches further past t0.

The corrected forecast of hour t0+k, for k from 1 to 24, uses lags L >= k and rolling windows that end at t0 or earlier. The same rule applies to PV, load, GHI, temperature and the clear-sky index. Calendar fields of the target hour are known ahead of time. One XGBoost model is fit per lead, on years through 2023, with 2024 used only for early stopping. The MILP, the costs, the start cost of 33.75 GHS, the heuristic, the sizing and the terminal SOC credit are the ones already in the code.

## 2025 dispatch

Load is load_v2. Fuel curve is sourced_mean. Start cost is 33.75 GHS. The heuristic was re-selected on 2024 at theta {setup['theta']:.0%} and psi {setup['psi']:.0%}. SOC at the start of 2025 is {setup['soc0']:.6f} kWh. Solver limits match `walkforward.run_fold`: rolling MILP horizon 24 h, commit 1 h, 10 s and MIP gap 1e-3; perfect-foresight blocks of 744 h with 48 h overlap, 300 s and MIP gap 1e-3.

The gap column is the share of the heuristic-to-bound cost gap that the controller closes. 0 is the heuristic. 100 is the perfect-foresight bound.

| controller | cost GHS | fuel L | run h | starts | vs heuristic % | vs baseline % | vs bound % | gap closed % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{line("baseline", base)}
{line("heuristic", h)}
{line("A leaky MILP", a)}
{line("B direct MILP", b)}
{line("C seasonal-naive MILP", c)}
{line("perfect foresight", o)}

Seasonal-naive means the same clock hour yesterday. At a 24-hour horizon that reference is still inside the origin, so this forecast is causal.

Re-solved costs against the stored 33.75 GHS walk-forward fold (`results/v2/start_cost_sourced_2026-10-15/usd3_liu2017/fold_2025.csv`): baseline {base_diff:+.6f} GHS, heuristic {h_cost_diff:+.6f} GHS, perfect foresight {o_cost_diff:+.6f} GHS, leaky MILP {a_cost_diff:+.6f} GHS. Row-level differences are in `compare_to_saved_33p75.csv`.

The retrained 1-hour-ahead model, which is lead k = 1 of the direct forecaster, differs from the stored 2025 forecast by {agree['mae_pv_kW']:.6f} kW MAE on PV and {agree['mae_load_kW']:.6f} kW MAE on load.

## Forecast skill on 2025

MAE is in kW. nRMSE is RMSE divided by the range of the actual series, in percent, the same definition as `forecast.metrics`. Persistence repeats the value at the origin. Seasonal-naive repeats the same hour yesterday. For a 24-hour lead those two references are the same hour, so their scores match.

{chr(10).join(skill_lines)}

## Generator structure of the direct MILP

Blocks are the commitment flags the rolling MILP hands to the executor, the same object the existing Table IX file uses. The cost table above counts an hour as running when diesel output is positive. Those two counts differ when the executor starts the unit the plan left off, or the plan commits an hour the executor does not fuel.

| banked % | mean block h | blocks of 1-2 h % | starts at 18:00 or later % | blocks | commitment hours | banked kWh |
|---|---:|---:|---:|---:|---:|---:|
| {ix.banked_pct:.4f} | {ix.mean_block_h:.4f} | {ix.blocks_1_2h_pct:.4f} | {ix.starts_ge_18_pct:.4f} | {int(ix.n_blocks)} | {int(ix.commitment_run_h)} | {ix.banked_kwh:.2f} |

Banked energy is battery charge above the PV surplus, divided by genset output.

Start hours: {hour_bits}.

Block lengths: {length_bits}.

## What this run did not do

The full `reproduce.py` pipeline was not run. The manuscript was not edited. `walkforward.fit_fold` still writes 1-step forecasts, so the stored walk-forward can be rebuilt. Canonical hashes were recorded at the start and checked at the end.
"""
    REPORT.write_text(text)
    print(REPORT, flush=True)


def guard(stage: str, hashes: dict):
    got = canon_hashes()
    if got != hashes:
        raise SystemExit(f"canonical file changed during {stage}: {got} vs {hashes}")
    print(f"canonical hashes unchanged after {stage}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["train", "dispatch", "report", "all"])
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = canon_hashes()
    (OUT / "canonical_hashes_start.json").write_text(json.dumps(hashes, indent=2))
    df = observed_frame()
    if args.stage in ("train", "dispatch", "all"):
        pred_pv, pred_load, _notes = train_and_predict(df)
        skill_table(df, pred_pv, pred_load)
    else:
        z = np.load(OUT / "direct_pred_2025.npz")
        pred_pv, pred_load = z["pv"], z["load"]
    if args.stage in ("dispatch", "all"):
        run_controllers(df, pred_pv, pred_load)
    if args.stage in ("report", "dispatch", "all"):
        if not (OUT / "skill_by_horizon.csv").exists():
            skill_table(df, pred_pv, pred_load)
        summary = score_all()
        skill = pd.read_csv(OUT / "skill_by_horizon.csv")
        write_report(summary, skill)
        print(summary[["controller", "total_cost_GHS", "fuel_litres",
                       "genset_run_hours", "genset_starts",
                       "saving_vs_heuristic_pct", "gap_closed_pct"]]
              .to_string(index=False), flush=True)
    guard(args.stage, hashes)
    (OUT / "canonical_hashes_end.json").write_text(json.dumps(canon_hashes(), indent=2))


if __name__ == "__main__":
    main()
