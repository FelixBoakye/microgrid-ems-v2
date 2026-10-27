"""Phase 6 Step 4 — expanding-window walk-forward validation.

Only one test year has been used so far. Eight are on hand, so the strongest
available robustness evidence costs no new data.

For each test year Y in 2022..2025:

    train    2018 .. Y-2
    validate Y-1            (early stopping only)
    test     Y

Forecasters are retrained at every fold. The heuristic's (theta, psi) are
re-selected on **year Y-1**, never on the test year — the Step 3 result showed
that tuning does not transfer across load shapes, so carrying one setting across
folds would either flatter or handicap it arbitrarily.

Usage:
    python -m scripts_v2.walkforward --stage forecast
    python -m scripts_v2.walkforward --stage forecast_direct
    python -m scripts_v2.walkforward --stage dispatch --year 2022 --start-cost 33.75
    python -m scripts_v2.walkforward --stage collect
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import controllers as ct
from . import cost_model as cm
from . import direct_forecast as D
from . import dispatch_core as dc
from . import heuristic as H
from . import milp_dispatch as mp
from . import paths as P
from .config import SOC_INIT
from .forecast import FEATURES_LOAD, FEATURES_PV, XGB_PARAMS, metrics
from .run_ladder import manifest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEST_YEARS = (2022, 2023, 2024, 2025)
THETAS_V40 = (0.50, 0.60, 0.70, 0.80)
# V4.2: grid widened past 0.80 (0.80 was the edge and was selected); 0.30 is
# the minimum-load end, 1.00 is cycle-charging-like loading.
THETAS = (0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
PSIS = (0.30, 0.40, 0.50, 0.60, 0.75, 1.00)


def out_dir(name="walkforward"):
    d = P.results_subdir(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ data
def full_series():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    return pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime"
    ).sort_values("datetime").reset_index(drop=True)


def soc_at_year_start(year):
    """SOC the baseline carries into 1 January of `year`, so each fold starts
    from the state the system would actually be in."""
    d = full_series()
    res = dc.baseline(d.PV_power_kW.values, d.total_load_kW.values,
                      soc_init=SOC_INIT, uc_parity=False)  # year-start SOC: legacy rule so MILP rows are unchanged
    i = int(d.index[d.datetime.dt.year == year][0])
    return float(res.soc[i - 1])


# -------------------------------------------------------------- forecasts
def fit_fold(year, seed=42):
    """Train on 2018..year-2, early-stop on year-1, predict year.

    Each saved row is a 1-step forecast of that hour, built with actuals
    through the previous hour. ``run_fold`` then plans on a 24-hour slice of
    those rows. Hours after the decision were fit with measurements that had
    not happened yet. ``direct_forecast`` is the corrected per-lead model.
    This function is unchanged so the stored walk-forward can still be rebuilt.
    """
    ds = pd.read_csv(P.data("ml_dataset.csv"), parse_dates=["datetime"])
    tr = ds[ds.datetime.dt.year <= year - 2]
    va = ds[ds.datetime.dt.year == year - 1]
    te = ds[ds.datetime.dt.year == year]
    assert len(te) in (8760, 8784), f"{year}: {len(te)} test hours"

    out = te[["datetime"]].copy()
    mets = {}
    for target, feats, short in (("target_PV_kW", FEATURES_PV, "PV"),
                                 ("target_load_kW", FEATURES_LOAD, "load")):
        m = xgb.XGBRegressor(**dict(XGB_PARAMS, random_state=seed))
        m.fit(tr[feats].values, tr[target].values,
              eval_set=[(va[feats].values, va[target].values)], verbose=False)
        pred = np.clip(m.predict(te[feats].values), 0, None)
        out[f"{short}_actual"] = te[target].values
        out[f"xgb_{short}_pred"] = pred
        mets[f"xgb_{short}"] = metrics(te[target].values, pred)
        print(f"  {year} {short:<5} R2={mets[f'xgb_{short}']['r2']:.4f} "
              f"RMSE={mets[f'xgb_{short}']['rmse']:.3f} "
              f"(train {tr.datetime.dt.year.min()}-{tr.datetime.dt.year.max()}, "
              f"val {year - 1}, n_train={len(tr)})", flush=True)
    out.to_csv(out_dir() / f"forecasts_{year}.csv", index=False)
    (out_dir() / f"forecast_metrics_{year}.json").write_text(json.dumps(mets, indent=2))
    return mets


def forecast_direct(years=TEST_YEARS, seed=42):
    """One direct model per lead, for each test year. Planning reads these."""
    df = D.observed_frame()
    saved = {}
    for year in years:
        path = D.cache_path(year, seed)
        pv, load = D.ensure_direct(df, year, path, seed=seed)
        saved[year] = str(path)
        _ = pv, load
    return saved


def skill_by_horizon(years=TEST_YEARS, seed=42):
    """Lead 1..24 nRMSE against persistence and seasonal-naive, per test year."""
    df = D.observed_frame()
    pv = df.PV_power_kW.to_numpy(dtype=float)
    ld = df.total_load_kW.to_numpy(dtype=float)
    rows = []
    for year in years:
        pred_pv, pred_ld = D.ensure_direct(df, year, D.cache_path(year, seed), seed=seed)
        idx = D.year_index(df, year)
        for k in range(1, D.HORIZON + 1):
            for target, actual, pred in (("PV", pv, pred_pv), ("load", ld, pred_ld)):
                y = actual[idx]
                hats = {
                    "direct": pred[k, idx],
                    "persistence": actual[idx - k],
                    "seasonal_naive": actual[idx - 24],
                }
                if k == 24 and not np.allclose(hats["persistence"], hats["seasonal_naive"]):
                    raise RuntimeError(f"{year} {target}: 24 h persistence is not seasonal-naive")
                for method, hat in hats.items():
                    if not np.isfinite(hat).all():
                        raise RuntimeError(f"non-finite {method} {target} lead {k} in {year}")
                    m = metrics(y, hat)
                    rows.append(dict(year=year, target=target, lead_h=k, method=method,
                                     mae_kW=m["mae"], nrmse_pct=m["nrmse_pct"],
                                     n=int(len(y))))
    out = pd.DataFrame(rows)
    dest = P.results_root() / "skill_by_horizon.csv"
    out.to_csv(dest, index=False)
    print(f"saved {dest} ({len(out)} rows)", flush=True)
    return out


def select_heuristic_on(pv_prev, load_prev, soc_prev, cost):
    """The V4.2 selection rule: the (theta, psi) pair of THETAS x PSIS with the
    lowest total cost on the year-1 series. Every study uses this rule."""
    sw = H.sweep_threshold(P.smoke_trim(np.asarray(pv_prev, float)),
                           P.smoke_trim(np.asarray(load_prev, float)),
                           soc_init=soc_prev, thresholds=THETAS,
                           bank_ceilings=PSIS, cost=cost)
    b = sw.loc[sw.total_cost_GHS.idxmin()]
    return float(b.threshold_pct) / 100.0, float(b.bank_to_soc_pct) / 100.0


def soc_at_year_start_of(frame, year):
    """soc_at_year_start for an arbitrary series (demand variants)."""
    res = dc.baseline(frame.PV_power_kW.values, frame.total_load_kW.values,
                      soc_init=SOC_INIT, uc_parity=False)  # year-start SOC: legacy rule so MILP rows are unchanged
    i = int(frame.index[frame.datetime.dt.year == year][0])
    return float(res.soc[i - 1])


def select_heuristic(year, cost):
    """Re-select (theta, psi) on year-1 — never on the test year."""
    d = full_series()
    prev = d[d.datetime.dt.year == year - 1]
    soc0 = soc_at_year_start(year - 1)
    return select_heuristic_on(prev.PV_power_kW.values, prev.total_load_kW.values,
                               soc0, cost)


# --------------------------------------------------------------- dispatch
def _cost(curve, start_cost_ghs):
    kw = {} if start_cost_ghs is None else {"start_cost_ghs": float(start_cost_ghs)}
    return cm.corrected(curve, **kw)


def run_fold(year, curve="sourced_mean", start_cost_ghs=None, subdir="walkforward",
             seed=42):
    cost = _cost(curve, start_cost_ghs)
    frame = D.observed_frame()
    idx = D.year_index(frame, year)
    yr = frame.iloc[idx].reset_index(drop=True)
    PVa = yr.PV_power_kW.to_numpy(dtype=float)
    LDa = yr.total_load_kW.to_numpy(dtype=float)
    pred_pv, pred_ld = D.ensure_direct(frame, year, D.cache_path(year, seed), seed=seed)
    direct = D.trajectory_fn(pred_pv, pred_ld, idx)
    seasonal_fc = D.seasonal_trajectory(
        frame.PV_power_kW.to_numpy(dtype=float),
        frame.total_load_kW.to_numpy(dtype=float), idx)
    nan = np.full(len(PVa), np.nan)
    soc0 = soc_at_year_start(year)

    theta, psi = select_heuristic(year, cost)
    print(f"  {year}: heuristic re-selected on {year - 1} -> "
          f"theta={theta:.0%}, psi={psi:.0%}", flush=True)

    rows = {}
    base = dc.baseline(PVa, LDa, soc_init=soc0, genset=cost.genset)
    rows["baseline"] = base
    heur, hrun = H.commitment_heuristic(PVa, LDa, soc_init=soc0,
                                        threshold=theta, bank_to_soc=psi,
                                        genset=cost.genset)
    rows["heuristic"] = heur
    orc, orun = mp.solve_year_blocked(PVa, LDa, soc0, cost, block_h=744,
                                      overlap_h=48, time_limit=300.0,
                                      mip_gap=1e-3, verbose=False)
    rows["oracle"] = orc
    milp, mrun = ct.run_rolling_milp(PVa, LDa, nan, nan, soc_init=soc0,
                                     horizon=24, commit=1, cost=cost,
                                     hierarchical=True, time_limit=10.0,
                                     mip_gap=1e-3, progress_every=2000,
                                     trajectory_at=direct)
    rows["roll24"] = milp
    seas, _srun = ct.run_rolling_milp(PVa, LDa, nan, nan, soc_init=soc0,
                                      horizon=24, commit=1, cost=cost,
                                      hierarchical=True, time_limit=10.0,
                                      mip_gap=1e-3, progress_every=2000,
                                      trajectory_at=seasonal_fc)
    rows["seasonal"] = seas

    recs = []
    for name, res in rows.items():
        m = res.metrics(hours_per_year=len(PVa))
        e = cost.evaluate(res)
        # `m` already carries "hours"; do not pass it again.
        recs.append(dict(year=year, config=name, theta=theta, psi=psi,
                         start_cost_ghs=cost.genset.start_cost_ghs, **m,
                         **{k: v for k, v in e.items() if k != "cost_model"}))
    df = pd.DataFrame(recs)
    b = df[df.config == "baseline"].total_cost_GHS.iloc[0]
    h = df[df.config == "heuristic"].total_cost_GHS.iloc[0]
    o = df[df.config == "oracle"].total_cost_GHS.iloc[0]
    df["saving_vs_baseline_pct"] = (1 - df.total_cost_GHS / b) * 100
    df["saving_vs_heuristic_pct"] = (1 - df.total_cost_GHS / h) * 100
    df["premium_vs_oracle_pct"] = (df.total_cost_GHS / o - 1) * 100
    df.to_csv(out_dir(subdir) / f"fold_{year}.csv", index=False)
    np.savez_compressed(out_dir(subdir) / f"fold_{year}_dispatch.npz",
                        **{f"{k}_{f}": getattr(v, f) for k, v in rows.items()
                           for f in ("diesel", "soc", "unmet", "charge",
                                     "discharge", "curtail")},
                        # Executed on/off (output > 0); one start count.
                        heuristic_running=hrun, roll24_running=mrun,
                        oracle_running=orun,
                        datetime=yr.datetime.values.astype("datetime64[h]").astype(int))
    print(df[["config", "total_cost_GHS", "saving_vs_baseline_pct",
              "saving_vs_heuristic_pct", "premium_vs_oracle_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return df


def seasonal(year=2025, curve="sourced_mean", start_cost_ghs=None,
             subdir="walkforward"):
    """Disaggregate one fold by season and month, at the fold's start cost."""
    cost = _cost(curve, start_cost_ghs)
    z = np.load(out_dir(subdir) / f"fold_{year}_dispatch.npz")
    d = full_series()
    yr = d[d.datetime.dt.year == year].reset_index(drop=True)
    mon = yr.datetime.dt.month.values
    seasons = {"harmattan (Nov-Feb)": np.isin(mon, [11, 12, 1, 2]),
               "hot dry (Mar-May)": np.isin(mon, [3, 4, 5]),
               "rainy (Jun-Oct)": np.isin(mon, [6, 7, 8, 9, 10])}
    rows = []
    for label, mask in list(seasons.items()) + [(f"month {m:02d}", mon == m)
                                                for m in range(1, 13)]:
        rec = dict(period=label, hours=int(mask.sum()))
        costs = {}
        for c in ("baseline", "heuristic", "oracle", "roll24", "seasonal"):
            r = dc.SimResult(soc=z[f"{c}_soc"][mask], charge=z[f"{c}_charge"][mask],
                             discharge=z[f"{c}_discharge"][mask],
                             diesel=z[f"{c}_diesel"][mask],
                             unmet=z[f"{c}_unmet"][mask],
                             curtail=z[f"{c}_curtail"][mask],
                             pv=yr.PV_power_kW.values[mask],
                             load=yr.total_load_kW.values[mask])
            e = cost.evaluate(r)
            costs[c] = e["total_cost_GHS"]
            rec[f"{c}_LF_pct"] = e["mean_load_factor_pct"]
            rec[f"{c}_run_h"] = e["genset_run_hours"]
        rec["milp_vs_baseline_pct"] = (1 - costs["roll24"] / costs["baseline"]) * 100
        rec["milp_vs_heuristic_pct"] = (1 - costs["roll24"] / costs["heuristic"]) * 100
        rec["heur_vs_baseline_pct"] = (1 - costs["heuristic"] / costs["baseline"]) * 100
        rec["milp_premium_vs_oracle_pct"] = (costs["roll24"] / costs["oracle"] - 1) * 100
        rec["seasonal_vs_heuristic_pct"] = (1 - costs["seasonal"] / costs["heuristic"]) * 100
        rec["seasonal_vs_baseline_pct"] = (1 - costs["seasonal"] / costs["baseline"]) * 100
        rec["seasonal_premium_vs_oracle_pct"] = (costs["seasonal"] / costs["oracle"] - 1) * 100
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir(subdir) / f"seasonal_{year}.csv", index=False)
    return df


def collect(subdir="walkforward"):
    folds = [pd.read_csv(p) for p in sorted(out_dir(subdir).glob("fold_*.csv"))]
    if not folds:
        raise SystemExit("no folds yet")
    df = pd.concat(folds, ignore_index=True)
    df.to_csv(out_dir(subdir) / "walkforward_folds.csv", index=False)

    keys = ["saving_vs_baseline_pct", "saving_vs_heuristic_pct",
            "premium_vs_oracle_pct", "genset_run_hours", "genset_starts",
            "mean_load_factor_pct", "LPSP_pct"]
    agg = (df[df.config == "roll24"][["year"] + keys]
           .set_index("year"))
    print("\nROLLING MILP, per fold")
    print(agg.to_string(float_format=lambda v: f"{v:,.2f}"))
    print("\nAGGREGATE (mean +- std over 4 folds)")
    for k in keys:
        v = agg[k]
        print(f"  {k:<28} {v.mean():>10,.2f} +- {v.std(ddof=1):>8,.2f}")
    summ = pd.DataFrame({"mean": agg.mean(), "std": agg.std(ddof=1)})
    summ.to_csv(out_dir(subdir) / "walkforward_summary.csv")
    seas = df[df.config == "seasonal"]
    if len(seas):
        sagg = seas[["year"] + [k for k in keys if k in seas.columns]].set_index("year")
        # The seasonal-naive comparator is scored on its own saving columns.
        skeys = [c for c in ("saving_vs_baseline_pct", "saving_vs_heuristic_pct",
                             "premium_vs_oracle_pct") if c in seas.columns]
        if skeys:
            sagg = seas.set_index("year")[skeys]
            pd.DataFrame({"mean": sagg.mean(), "std": sagg.std(ddof=1)}).to_csv(
                out_dir(subdir) / "walkforward_seasonal_summary.csv")
            print("\nSEASONAL-NAIVE MILP, per fold")
            print(sagg.to_string(float_format=lambda v: f"{v:,.2f}"))
    (out_dir(subdir) / "walkforward_manifest.json").write_text(
        json.dumps(manifest({"test_years": list(TEST_YEARS),
                             "variant": P.variant(),
                             "subdir": subdir,
                             "forecast": "direct per-horizon"}), indent=2))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["forecast", "forecast_direct", "skill",
                             "dispatch", "seasonal", "collect"])
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--curve", default="sourced_mean")
    ap.add_argument("--start-cost", type=float, default=None,
                    help="GHS per start. Default is the cost-model 150.")
    ap.add_argument("--out-subdir", default="walkforward")
    a = ap.parse_args(argv)
    years = [a.year] if a.year else list(TEST_YEARS)

    if a.stage == "forecast":
        for y in years:
            fit_fold(y)
    elif a.stage == "forecast_direct":
        forecast_direct(years)
    elif a.stage == "skill":
        skill_by_horizon(years)
    elif a.stage == "dispatch":
        for y in years:
            t0 = time.time()
            run_fold(y, a.curve, start_cost_ghs=a.start_cost, subdir=a.out_subdir)
            print(f"  fold {y} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    elif a.stage == "seasonal":
        for y in years:
            print(seasonal(y, a.curve, start_cost_ghs=a.start_cost,
                           subdir=a.out_subdir).to_string(
                index=False, float_format=lambda v: f"{v:,.2f}"))
    else:
        collect(a.out_subdir)


if __name__ == "__main__":
    main()
