"""v2 feature matrix and forecasts, over the whole year.

Fixes a defect the original review did not catch, recorded here as **D5**.

In `04_build_ml_dataset.py` the clear-sky index is

    GHI / CLRSKY_SFC_SW_DWN.replace(0, nan)

and CLRSKY is exactly 0 for every night hour — 34,543 of 70,128 hours, 49.3% of
the series. The subsequent `dropna()` therefore deletes **every night hour**,
not "the first 168 hours" as the script's own message claims. It drops 37,542
rows and leaves 32,586, of which the 2025 test set is 4,081 hours spanning only
07:00-18:00.

Two consequences for the published result:

  * the framework is evaluated on daylight hours only, then annualised by
    dividing by 4081/8760, extrapolating a daytime-only dispatch to a full year
  * consecutive rows are treated as consecutive hours, so a "24-hour" planning
    horizon actually spans two days and the SOC recursion is run across gaps of
    13 hours as if they were one

The fix is to define the clear-sky index as 0 when there is no clear-sky
irradiance, which is what it physically means: no sun, no clearness to measure.
Only the first 168 hours are then dropped, as intended.

The five leakage tests from the original are retained and extended.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths as P

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

LAG_HOURS = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [6, 12, 24, 48]

TRAIN_END = "2023-12-31 23:59:59"
VAL_START, VAL_END = "2024-01-01 00:00:00", "2024-12-31 23:59:59"
TEST_START = "2025-01-01 00:00:00"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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

    for lag in LAG_HOURS:
        out[f"PV_lag{lag}"] = df["PV_power_kW"].shift(lag)
        out[f"load_lag{lag}"] = df["total_load_kW"].shift(lag)
        out[f"GHI_lag{lag}"] = df["GHI_Wm2"].shift(lag)
        out[f"T2M_lag{lag}"] = df["T2M"].shift(lag)

    pv_s = df["PV_power_kW"].shift(1)
    ld_s = df["total_load_kW"].shift(1)
    gh_s = df["GHI_Wm2"].shift(1)
    for w in ROLLING_WINDOWS:
        out[f"PV_roll{w}_mean"] = pv_s.rolling(w).mean()
        out[f"PV_roll{w}_std"] = pv_s.rolling(w).std()
        out[f"load_roll{w}_mean"] = ld_s.rolling(w).mean()
        out[f"load_roll{w}_std"] = ld_s.rolling(w).std()
        out[f"GHI_roll{w}_mean"] = gh_s.rolling(w).mean()

    # --- D5 FIX -------------------------------------------------------------
    # Clearness is undefined, not missing, when there is no clear-sky
    # irradiance. Define it as 0 at night instead of NaN, so dropna() removes
    # only the lag/rolling warm-up rather than every night hour.
    clr = df["CLRSKY_SFC_SW_DWN"]
    idx = np.where(clr > 0, df["GHI_Wm2"] / clr.where(clr > 0), 0.0)
    idx = pd.Series(idx, index=df.index)
    out["clearsky_index_lag1"] = idx.shift(1)
    out["clearsky_index_lag24"] = idx.shift(24)
    return out


def verify_no_leakage(feat: pd.DataFrame, pv: pd.Series, load: pd.Series):
    """The original five tests, plus two for the D5 fix."""
    checks = []

    ems = ["SOC", "soc", "battery_charge", "battery_discharge",
           "diesel_output", "unmet_load", "curtailed"]
    checks.append(("no EMS output columns",
                   not [c for c in feat.columns
                        if any(e in c.lower() for e in ems)]))

    ind = pv.shift(24)
    v = ~ind.isna()
    checks.append(("lag features bit-exact",
                   bool(np.allclose(ind[v].values, feat["PV_lag24"][v].values))))

    i, w = 200, 24
    checks.append(("rolling excludes current step",
                   abs(pv.iloc[i - w:i].mean() - feat[f"PV_roll{w}_mean"].iloc[i]) < 1e-6))

    s = feat["datetime"].iloc[100]
    checks.append(("calendar deterministic",
                   abs(np.sin(2 * np.pi * s.hour / 24) - feat["hour_sin"].iloc[100]) < 1e-10))

    checks.append(("targets aligned to t",
                   abs(feat["target_PV_kW"].iloc[50] - pv.iloc[50]) < 1e-10))

    # --- extended for D5 ---
    checks.append(("clearsky index finite everywhere",
                   bool(np.isfinite(feat["clearsky_index_lag1"].iloc[1:]).all())))
    kept = feat.dropna()
    checks.append(("all 24 hours of the day survive dropna",
                   sorted(kept["datetime"].dt.hour.unique()) == list(range(24))))

    print("\n=== LEAKAGE + INTEGRITY VERIFICATION ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok in checks if not ok]
    if failed:
        raise ValueError(f"verification failed: {failed}")
    return True


def main():
    os.makedirs(DATA, exist_ok=True)
    pv_df = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld_df = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    df = pv_df.merge(ld_df[["datetime", "total_load_kW"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Merged: {len(df)} hours")

    feat = build_features(df)
    verify_no_leakage(feat, df["PV_power_kW"], df["total_load_kW"])

    before = len(feat)
    feat = feat.dropna().reset_index(drop=True)
    print(f"\nDropped {before - len(feat)} rows (original dropped 37,542)")
    print(f"Final dataset: {len(feat)} rows, hours of day: "
          f"{feat.datetime.dt.hour.min()}-{feat.datetime.dt.hour.max()}")

    train = feat[feat.datetime <= TRAIN_END]
    val = feat[(feat.datetime >= VAL_START) & (feat.datetime <= VAL_END)]
    test = feat[feat.datetime >= TEST_START]
    assert train.datetime.max() < val.datetime.min()
    assert val.datetime.max() < test.datetime.min()
    for n, s in (("train", train), ("val", val), ("test", test)):
        assert s.datetime.is_monotonic_increasing
        print(f"  {n:<6} {len(s):>6} hours  "
              f"{s.datetime.min().date()} to {s.datetime.max().date()}")

    feat.to_csv(P.data("ml_dataset.csv"), index=False)
    train.to_csv(P.data("train.csv"), index=False)
    val.to_csv(P.data("validation.csv"), index=False)
    test.to_csv(P.data("test.csv"), index=False)
    print("\nSaved v2_ml_dataset.csv, v2_train.csv, v2_validation.csv, v2_test.csv")


if __name__ == "__main__":
    main()
