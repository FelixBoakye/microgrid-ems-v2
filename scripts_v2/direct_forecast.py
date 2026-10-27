"""Direct multi-horizon forecasts for the rolling MILP.

Timing convention
-----------------
``run_rolling_milp`` decides at the start of an hour and then executes that
hour on the realised PV and load. The state passed into the planner is the
state at the end of the previous hour. Realised PV, load, GHI and temperature
of the hour about to be executed are not known yet.

In that function the loop index is the hour about to be executed. The planning
origin used here is the previous hour, the last hour whose measurements are
in hand. Call that origin ``t0``. The plan covers ``t0+1`` through ``t0+24``,
and hour ``t0+1`` is the hour that gets executed.

Data through ``t0`` is known. Data after ``t0`` is not. The forecast of hour
``t0+k`` (``k = 1..24``) may use lags ``L >= k`` of PV, load, GHI, temperature
and the clear-sky index, and rolling windows that end at ``t0`` or earlier.
Calendar fields of the target hour are known ahead of time. One model is fit
per lead time, on years before the test year, with the previous year used only
for early stopping. That is the same walk-forward split as ``walkforward.fit_fold``.

A 1-step forecast of hour ``h`` uses actuals through ``h-1``. Slicing a year
of those 1-step forecasts across a 24-hour plan therefore shows the planner
actuals from after the origin. ``trajectory_at`` is the path that does not.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths as P
from .build_dataset import build_features
from .forecast import FEATURES_LOAD, FEATURES_PV, XGB_PARAMS

CACHE_REVISION = 1

HORIZON = 24
LAGS = {
    "pv": (1, 2, 3, 24, 48, 168),
    "load": (1, 2, 3, 24, 48, 168),
    "ghi": (1, 24, 48),
    "t2m": (1, 24),
    "clearsky": (1, 24),
}
ROLL_MEAN = {
    "pv": (6, 24, 48),
    "ghi": (6, 24),
    "load": (6, 24, 48),
}
ROLL_STD = {
    "load": (6, 24),
}
PREFIX = {
    "pv": "PV",
    "load": "load",
    "ghi": "GHI",
    "t2m": "T2M",
    "clearsky": "clearsky_index",
}
CALENDAR = {
    "hour", "month", "dayofweek", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "is_weekend",
}

TIMING_CONVENTION = """
The rolling MILP decides at the start of an hour and then executes that hour
on the realised PV and load. The state passed into the planner is the state
at the end of the previous hour. Realised PV, load, GHI and temperature of
the hour about to be executed are not known.

In run_rolling_milp the loop index is that hour about to be executed. The
planning origin t0 used by the direct forecaster is the previous hour, the
last hour whose measurements are known. The plan covers t0+1 through t0+24,
and hour t0+1 is the one executed.

Data through t0 is known. Data after t0 is not. The forecast of hour t0+k
(k = 1 to 24) uses lags L >= k of PV, load, GHI, temperature and the
clear-sky index, and rolling windows that end at t0 or earlier. Calendar
fields of the target hour are known ahead of time. One model is fit per
lead time.
""".strip()


def clearsky_index(df: pd.DataFrame) -> pd.Series:
    """Same night rule as build_dataset.build_features: 0 when clear-sky is 0."""
    clr = df["CLRSKY_SFC_SW_DWN"]
    idx = np.where(clr > 0, df["GHI_Wm2"] / clr.where(clr > 0), 0.0)
    return pd.Series(idx, index=df.index)


def _raw_series(df: pd.DataFrame) -> dict:
    return {
        "pv": df["PV_power_kW"],
        "load": df["total_load_kW"],
        "ghi": df["GHI_Wm2"],
        "t2m": df["T2M"],
        "clearsky": clearsky_index(df),
    }


def _lag_ok(name: str, k: int) -> bool:
    if name in CALENDAR:
        return True
    if "_lag" in name:
        return int(name.rsplit("_lag", 1)[-1]) >= k
    if "_roll" in name:
        # Recomputed so the window ends at the origin (target - k), not at
        # target - 1. The column stays in the model at every lead.
        return True
    raise ValueError(f"unrecognised feature {name}")


def features_for(short: str, k: int) -> list:
    """Columns the model at lead ``k`` may see. Lead 1 matches the 1-step model."""
    if k < 1:
        raise ValueError(k)
    base = FEATURES_PV if short == "PV" else FEATURES_LOAD
    return [name for name in base if _lag_ok(name, k)]


def build_horizon_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Feature row of target hour ``t`` uses measurements at or before ``t - k``.

    Lead ``k = 1`` reproduces ``build_dataset.build_features`` on the columns
    the 1-step model actually reads.
    """
    if k < 1:
        raise ValueError(k)
    out = pd.DataFrame(index=df.index)
    out["datetime"] = df["datetime"]
    out["target_PV_kW"] = df["PV_power_kW"]
    out["target_load_kW"] = df["total_load_kW"]
    dt = df["datetime"]
    out["hour"] = dt.dt.hour
    out["month"] = dt.dt.month
    out["dayofweek"] = dt.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    raw = _raw_series(df)
    for series, lags in LAGS.items():
        for lag in lags:
            if lag >= k:
                out[f"{PREFIX[series]}_lag{lag}"] = raw[series].shift(lag)
    shifted = {series: raw[series].shift(k) for series in ("pv", "load", "ghi")}
    for series, windows in ROLL_MEAN.items():
        for w in windows:
            out[f"{PREFIX[series]}_roll{w}_mean"] = shifted[series].rolling(w).mean()
    for series, windows in ROLL_STD.items():
        for w in windows:
            out[f"{PREFIX[series]}_roll{w}_std"] = shifted[series].rolling(w).std()
    return out


def measured_sources(target: int, k: int) -> list:
    """``(series, index)`` pairs read for the forecast of ``target`` at lead ``k``.

    Every index is at most ``target - k``, the planning origin. Calendar fields
    of the target hour are not measurements and are not listed.
    """
    if k < 1:
        raise ValueError(k)
    origin = target - k
    found = []
    for series, lags in LAGS.items():
        for lag in lags:
            if lag >= k:
                found.append((series, target - lag))
    # A mean and a std of the same window read the same hours. List them once.
    windows_by_series = {}
    for series, windows in list(ROLL_MEAN.items()) + list(ROLL_STD.items()):
        windows_by_series.setdefault(series, set()).update(windows)
    for series, windows in windows_by_series.items():
        for w in windows:
            for j in range(origin - w + 1, origin + 1):
                found.append((series, j))
    return found


def assert_lead1_matches_dataset_builder(df: pd.DataFrame, atol=1e-8) -> None:
    """Lead-1 columns agree with ``build_features`` wherever both exist."""
    mine = build_horizon_features(df, 1)
    theirs = build_features(df)
    cols = FEATURES_PV + FEATURES_LOAD + ["target_PV_kW", "target_load_kW"]
    for col in cols:
        a = mine[col].to_numpy(dtype=float)
        b = theirs[col].to_numpy(dtype=float)
        nan_a = np.isnan(a)
        nan_b = np.isnan(b)
        if not np.array_equal(nan_a, nan_b):
            raise AssertionError(f"{col}: NaN mask differs")
        if not np.allclose(a[~nan_a], b[~nan_b], atol=atol, rtol=0.0):
            err = np.max(np.abs(a[~nan_a] - b[~nan_b]))
            raise AssertionError(f"{col}: max abs diff {err}")


def fit_models_on_targets(df, target_ilocs, params, horizons=range(1, HORIZON + 1)):
    """Fit one small or full model per lead. ``target_ilocs`` are row positions."""
    import xgboost as xgb

    models = {}
    targets = np.asarray(list(target_ilocs), dtype=int)
    for k in horizons:
        feat = build_horizon_features(df, k)
        block = feat.iloc[targets]
        for ycol, short in (("target_PV_kW", "PV"), ("target_load_kW", "load")):
            cols = features_for(short, k)
            sub = block[cols + [ycol]].dropna()
            if len(sub) < 30:
                raise RuntimeError(f"only {len(sub)} training rows for {short} lead {k}")
            model = xgb.XGBRegressor(**params)
            model.fit(sub[cols].to_numpy(dtype=float), sub[ycol].to_numpy(dtype=float),
                      verbose=False)
            models[(short, k)] = model
    return models


def fit_walkforward(df: pd.DataFrame, year: int, horizons=range(1, HORIZON + 1),
                    seed=42):
    """Train on years ``<= year-2``, early-stop on ``year-1``. One model per lead."""
    import xgboost as xgb

    models = {}
    notes = []
    for k in horizons:
        feat = build_horizon_features(df, k)
        cols = sorted(set(features_for("PV", k) + features_for("load", k)))
        kept = feat.dropna(subset=cols).copy()
        train = kept[kept.datetime.dt.year <= year - 2]
        val = kept[kept.datetime.dt.year == year - 1]
        if train.empty or val.empty:
            raise RuntimeError(f"lead {k}: empty train or val")
        if int(train.datetime.dt.year.max()) > year - 2:
            raise RuntimeError("training year leaked into the test window")
        for ycol, short in (("target_PV_kW", "PV"), ("target_load_kW", "load")):
            use = features_for(short, k)
            params = dict(XGB_PARAMS, random_state=seed)
            model = xgb.XGBRegressor(**params)
            model.fit(train[use].to_numpy(dtype=float), train[ycol].to_numpy(dtype=float),
                      eval_set=[(val[use].to_numpy(dtype=float), val[ycol].to_numpy(dtype=float))],
                      verbose=False)
            models[(short, k)] = model
            best = getattr(model, "best_iteration", None)
            notes.append(dict(lead=k, target=short, n_train=int(len(train)),
                              n_val=int(len(val)), n_features=len(use),
                              best_iteration=None if best is None else int(best)))
            print(f"  lead {k:2d} {short:<4} trees {best}  "
                  f"train {len(train)} val {len(val)} features {len(use)}",
                  flush=True)
    return models, notes


def predict_leads(models, df: pd.DataFrame, year: int,
                  horizons=range(1, HORIZON + 1)):
    """``pv[k, i]`` is the k-step forecast of row ``i``. Other rows stay NaN.

    Row ``i`` at lead ``k`` is the forecast issued at origin ``i - k``.
    """
    n = len(df)
    pv = np.full((HORIZON + 1, n), np.nan)
    load = np.full((HORIZON + 1, n), np.nan)
    mask = (df.datetime.dt.year == year).to_numpy()
    if int(mask.sum()) not in (8760, 8784):
        raise RuntimeError(f"{year} has {int(mask.sum())} hours")
    for k in horizons:
        feat = build_horizon_features(df, k)
        for arr, short in ((pv, "PV"), (load, "load")):
            cols = features_for(short, k)
            block = feat.loc[mask, cols]
            if block.isna().any().any():
                raise RuntimeError(f"NaN features for {short} lead {k} in {year}")
            hat = models[(short, k)].predict(block.to_numpy(dtype=float))
            arr[k, mask] = np.clip(hat, 0, None)
    return pv, load


def forecast_at_origin(models, df: pd.DataFrame, origin: int, horizon=HORIZON):
    """Forecast ``origin+1 .. origin+horizon`` from measurements through ``origin``.

    Returns PV, load, and the measurement indices those forecasts read.
    """
    if origin < 0 or origin + horizon >= len(df):
        raise ValueError(f"origin {origin} cannot see a {horizon}-hour plan")
    pv_hat = np.empty(horizon)
    ld_hat = np.empty(horizon)
    sources = []
    for step, k in enumerate(range(1, horizon + 1)):
        target = origin + k
        feat = build_horizon_features(df, k).iloc[target]
        for slot, short in ((pv_hat, "PV"), (ld_hat, "load")):
            cols = features_for(short, k)
            row = feat[cols]
            if row.isna().any():
                raise RuntimeError(f"NaN feature at target {target} lead {k}")
            hat = models[(short, k)].predict(row.to_numpy(dtype=float).reshape(1, -1))[0]
            slot[step] = float(max(hat, 0.0))
        sources.extend(measured_sources(target, k))
    return pv_hat, ld_hat, sources


def assemble_trajectory(pred, decision_index: int, horizon=HORIZON) -> np.ndarray:
    """Plan issued at ``decision_index``, the hour about to be executed.

    Offset ``step`` is the ``step+1`` lead forecast of that hour. Every entry
    was issued at origin ``decision_index - 1``.
    """
    n = pred.shape[1]
    h = min(horizon, n - decision_index)
    if h < 1:
        raise ValueError("decision index is past the series")
    out = np.empty(h)
    for step in range(h):
        out[step] = pred[step + 1, decision_index + step]
    if not np.isfinite(out).all():
        raise ValueError(f"non-finite trajectory at decision {decision_index}")
    return out


def seasonal_source_indices(decision_index: int, n_hours: int, horizon=HORIZON):
    """Same clock hour yesterday, for the plan that executes ``decision_index``.

    The latest source is ``decision_index - 1``.
    """
    h = min(horizon, n_hours - decision_index)
    return np.array([decision_index + step - 24 for step in range(h)], dtype=int)


def observed_frame() -> pd.DataFrame:
    """Hourly PV, weather and the active load variant, 2018 through 2025."""
    pv = pd.read_csv(P.DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    df = pv[["datetime", "PV_power_kW", "GHI_Wm2", "T2M", "CLRSKY_SFC_SW_DWN"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    step = df.datetime.diff().iloc[1:]
    if not bool((step == pd.Timedelta(hours=1)).all()):
        raise RuntimeError("observed frame is not hourly throughout")
    return df


def year_index(df: pd.DataFrame, year: int) -> np.ndarray:
    idx = np.flatnonzero(df.datetime.dt.year.to_numpy() == year)
    if len(idx) not in (8760, 8784):
        raise RuntimeError(f"{year} has {len(idx)} hours")
    if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
        raise RuntimeError(f"{year} is not a contiguous block")
    return idx


def cache_path(year: int, seed: int = 42, directory=None) -> Path:
    d = Path(directory) if directory is not None else P.results_subdir("direct")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"direct_pred_{int(year)}_seed{int(seed)}.npz"


def ensure_direct(df: pd.DataFrame, year: int, path, seed: int = 42):
    """Train once per (series, year, seed) and reuse the npz on the next call."""
    path = Path(path)
    if path.exists():
        z = np.load(path)
        same = ("revision" in z.files and int(z["revision"]) == CACHE_REVISION
                and int(z["year"]) == int(year) and int(z["seed"]) == int(seed)
                and z["pv"].shape[1] == len(df))
        if same:
            print(f"reuse {path}", flush=True)
            return z["pv"], z["load"]
    print(f"training direct models  year {year}  seed {seed}", flush=True)
    t0 = time.time()
    models, notes = fit_walkforward(df, year, seed=seed)
    pv, load = predict_leads(models, df, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pv=pv, load=load, year=np.int64(year),
                        seed=np.int64(seed), revision=np.int64(CACHE_REVISION))
    path.with_suffix(".notes.json").write_text(json.dumps(notes, indent=2))
    print(f"direct models {year} seed {seed} done in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return pv, load


def contiguous_block_ends(abs_index) -> np.ndarray:
    """Last absolute row of the contiguous run that contains each local hour.

    February and June are separate runs when the evaluation window drops the
    months between them, so a plan that starts in February stops on 28 February.
    """
    abs_index = np.asarray(abs_index, dtype=int)
    n = len(abs_index)
    end_abs = np.empty(n, dtype=int)
    start = 0
    for i in range(1, n + 1):
        if i == n or int(abs_index[i]) != int(abs_index[i - 1]) + 1:
            end_abs[start:i] = int(abs_index[i - 1])
            start = i
    return end_abs


def trajectory_fn(pred_pv, pred_load, abs_index, block_end_abs=None, scale_pv=1.0):
    """``trajectory_at(i)`` for local decision ``i``.

    ``abs_index[i]`` is the row in the full series. The horizon stops at the
    year, or earlier at ``block_end_abs[i]`` when the evaluation window is a
    set of month blocks. ``scale_pv`` scales the PV forecast only.
    """
    abs_index = np.asarray(abs_index, dtype=int)
    if block_end_abs is not None:
        block_end_abs = np.asarray(block_end_abs, dtype=int)
    scale = float(scale_pv)

    def trajectory_at(i):
        i = int(i)
        abs_i = int(abs_index[i])
        if block_end_abs is None:
            h = min(HORIZON, len(abs_index) - i)
        else:
            h = min(HORIZON, int(block_end_abs[i]) - abs_i + 1)
        if h < 1:
            raise ValueError(f"empty plan at local hour {i} (absolute {abs_i})")
        pv = assemble_trajectory(pred_pv, abs_i, horizon=h) * scale
        ld = assemble_trajectory(pred_load, abs_i, horizon=h)
        return pv, ld

    return trajectory_at


def seasonal_trajectory(pv, load, abs_index):
    """Same clock hour yesterday, cut at the end of ``abs_index``."""
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    abs_index = np.asarray(abs_index, dtype=int)

    def trajectory_at(i):
        i = int(i)
        abs_i = int(abs_index[i])
        h = min(HORIZON, len(abs_index) - i)
        src = seasonal_source_indices(abs_i, len(pv))[:h]
        if len(src) != h or int(src.max()) > abs_i - 1 or int(src.min()) < 0:
            raise RuntimeError(f"seasonal plan at {abs_i} reads outside the origin")
        return pv[src], load[src]

    return trajectory_at


def plan_source_indices(decision_index: int, n_hours: int, horizon=HORIZON) -> list:
    """Every measurement index the direct plan at ``decision_index`` can read."""
    h = min(horizon, n_hours - decision_index)
    found = []
    for step in range(h):
        k = step + 1
        target = decision_index + step
        found.extend(index for _, index in measured_sources(target, k))
    return found
