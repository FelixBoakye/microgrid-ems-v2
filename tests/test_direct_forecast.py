"""The rolling MILP's direct forecasts must not read measurements after the origin.

The origin t0 is the last observed hour. The hour about to be executed is
t0+1. Randomising every actual after t0, or finding a measurement timestamp
after t0, is a look-ahead leak.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts_v2 import controllers as ct
from scripts_v2 import cost_model as cm
from scripts_v2 import direct_forecast as dfcast
from scripts_v2 import milp_dispatch as mp
from scripts_v2.forecast import FEATURES_LOAD, FEATURES_PV

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MEASURED = ["PV_power_kW", "total_load_kW", "GHI_Wm2", "T2M", "CLRSKY_SFC_SW_DWN"]


def synthetic_frame(n=720, seed=1):
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-01", periods=n, freq="h")
    hour = np.asarray(dt.hour)
    ghi = np.clip(np.where((hour >= 6) & (hour <= 17),
                           700 * np.sin(np.pi * (hour - 6) / 12), 0.0)
                  + rng.normal(0, 15, n), 0, None)
    pv = np.clip(ghi / 1000.0 * 70.0 + rng.normal(0, 0.4, n), 0, None)
    load = np.clip(7 + 10 * ((hour >= 18) & (hour <= 22))
                   + rng.normal(0, 0.3, n), 0.5, None)
    t2m = 26 + 4 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.2, n)
    clr = np.where(hour >= 6, np.maximum(ghi, 40.0), 0.0)
    return pd.DataFrame({
        "datetime": dt,
        "PV_power_kW": pv,
        "total_load_kW": load,
        "GHI_Wm2": ghi,
        "T2M": t2m,
        "CLRSKY_SFC_SW_DWN": clr,
    })


def test_lead1_feature_list_matches_the_one_step_model():
    assert dfcast.features_for("PV", 1) == list(FEATURES_PV)
    assert dfcast.features_for("load", 1) == list(FEATURES_LOAD)


def test_later_leads_drop_lags_shorter_than_the_lead():
    pv6 = dfcast.features_for("PV", 6)
    assert "PV_lag1" not in pv6 and "PV_lag2" not in pv6 and "PV_lag3" not in pv6
    assert "GHI_lag1" not in pv6 and "T2M_lag1" not in pv6
    assert "clearsky_index_lag1" not in pv6
    assert "PV_lag24" in pv6 and "PV_roll6_mean" in pv6
    assert "PV_lag24" in dfcast.features_for("PV", 24)
    assert "PV_lag48" in dfcast.features_for("PV", 24)
    assert "load_lag24" in dfcast.features_for("load", 24)
    assert "load_lag3" not in dfcast.features_for("load", 4)


def test_lead1_features_match_the_dataset_builder():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    load = pd.read_csv(DATA / "synthetic_load_lv2.csv", parse_dates=["datetime"])
    frame = pv.merge(load[["datetime", "total_load_kW"]], on="datetime")
    frame = frame.sort_values("datetime").reset_index(drop=True)
    dfcast.assert_lead1_matches_dataset_builder(frame)


def test_plan_inputs_have_no_timestamp_after_the_origin():
    """Fail if any measurement behind the t0 plan is stamped after t0."""
    frame = synthetic_frame()
    decision = 500
    origin = decision - 1
    sources = dfcast.plan_source_indices(decision, n_hours=len(frame))
    assert sources
    assert max(sources) == origin
    assert decision not in sources
    times = frame.datetime.to_numpy()
    assert times[max(sources)] <= times[origin]
    assert not any(times[i] > times[origin] for i in sources)
    seasonal = dfcast.seasonal_source_indices(decision, len(frame))
    assert seasonal.max() == origin
    assert not any(times[i] > times[origin] for i in seasonal)


def test_features_ignore_measurements_after_the_origin():
    frame = synthetic_frame()
    origin = 500
    spoiled = frame.copy()
    spoiled.loc[spoiled.index > origin, MEASURED] = np.nan
    for k in range(1, dfcast.HORIZON + 1):
        target = origin + k
        fresh = dfcast.build_horizon_features(frame, k).iloc[target]
        blind = dfcast.build_horizon_features(spoiled, k).iloc[target]
        cols = dfcast.features_for("PV", k) + dfcast.features_for("load", k)
        got = fresh[cols].to_numpy(dtype=float)
        assert np.isfinite(got).all()
        assert np.array_equal(got, blind[cols].to_numpy(dtype=float))
        used = [i for _, i in dfcast.measured_sources(target, k)]
        assert max(used) <= origin
        assert target not in used


def test_short_tail_seasonal_stays_at_or_before_the_origin():
    decision = 100
    seas = dfcast.seasonal_source_indices(decision, n_hours=110)
    assert len(seas) == 10
    assert int(seas.max()) <= decision - 1
    assert decision not in seas


def test_one_step_forecast_of_a_later_hour_reads_past_the_origin():
    """The leak: a 1-step row for hour t0+2 is built from the actual at t0+1."""
    frame = synthetic_frame()
    origin = 400
    target = origin + 2
    feat = dfcast.build_horizon_features(frame, 1)
    assert feat.iloc[target]["PV_lag1"] == pytest.approx(
        frame.iloc[origin + 1]["PV_power_kW"])
    direct = dfcast.build_horizon_features(frame, 2)
    assert "PV_lag1" not in dfcast.features_for("PV", 2)
    assert direct.iloc[target]["PV_lag2"] == pytest.approx(
        frame.iloc[origin]["PV_power_kW"])


def test_perturbing_future_actuals_leaves_the_forecast_and_the_plan_unchanged():
    frame = synthetic_frame()
    origin = 600
    params = dict(n_estimators=15, max_depth=2, learning_rate=0.1,
                  subsample=1.0, colsample_bytree=1.0, random_state=0,
                  n_jobs=1, verbosity=0)
    models = dfcast.fit_models_on_targets(
        frame, range(200, 560), params)
    pv1, ld1, sources = dfcast.forecast_at_origin(models, frame, origin)
    assert max(i for _, i in sources) <= origin
    times = frame.datetime.to_numpy()
    assert not any(times[i] > times[origin] for _, i in sources)

    spoiled = frame.copy()
    rng = np.random.default_rng(0)
    n_fut = int((spoiled.index > origin).sum())
    spoiled.loc[spoiled.index > origin, "PV_power_kW"] = rng.random(n_fut) * 90
    spoiled.loc[spoiled.index > origin, "total_load_kW"] = rng.random(n_fut) * 40
    spoiled.loc[spoiled.index > origin, "GHI_Wm2"] = rng.random(n_fut) * 1000
    spoiled.loc[spoiled.index > origin, "T2M"] = 20 + rng.random(n_fut) * 15
    spoiled.loc[spoiled.index > origin, "CLRSKY_SFC_SW_DWN"] = rng.random(n_fut) * 1000
    pv2, ld2, _ = dfcast.forecast_at_origin(models, spoiled, origin)
    assert np.array_equal(pv1, pv2)
    assert np.array_equal(ld1, ld2)

    soc = 65.0
    kw = dict(time_limit=5.0, mip_gap=1e-6, z_init=0)
    plan1 = mp.solve(pv1, ld1, soc, cm.CORRECTED, **kw)
    plan2 = mp.solve(pv2, ld2, soc, cm.CORRECTED, **kw)
    assert np.array_equal(plan1.diesel, plan2.diesel)
    assert np.array_equal(plan1.running, plan2.running)
    assert np.array_equal(plan1.charge, plan2.charge)
    assert np.array_equal(plan1.discharge, plan2.discharge)


def test_rolling_milp_plans_from_trajectory_at_not_the_forecast_array():
    n = 30
    hour = np.arange(n) % 24
    pv = np.where((hour >= 8) & (hour <= 16), 40.0, 0.0)
    load = np.full(n, 8.0)
    load[(hour >= 18) & (hour <= 21)] = 25.0
    seen = []

    def trajectory_at(t0):
        seen.append(t0)
        h = min(24, n - t0)
        # Depends on the decision index only, never on a stored forecast array.
        pv_fc = np.full(h, 0.0 if (t0 % 24) >= 18 else 30.0)
        ld_fc = np.full(h, 20.0)
        return pv_fc, ld_fc

    res, run = ct.run_rolling_milp(
        pv, load, np.full(n, np.nan), np.full(n, np.nan),
        soc_init=65.0, horizon=24, commit=1, cost=cm.CORRECTED,
        hierarchical=True, time_limit=2.0, mip_gap=1e-3,
        trajectory_at=trajectory_at)
    assert seen[0] == 0 and seen[-1] == n - 1
    assert len(res.diesel) == n
    assert np.isfinite(res.diesel).all()
    assert run.shape == (n,)
