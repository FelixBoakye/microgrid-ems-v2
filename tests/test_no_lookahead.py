"""No-look-ahead tests for the rolling MILP's forecasts, on the real 2025 fold.

Drop into tests/. Needs data/pv_output.csv and data/synthetic_load_lv2.csv.
Origins are sampled from 2025 (seed fixed). Set LEAK_N_ORIGINS / LEAK_N_MILP to
change the sample sizes (defaults 200 / 20).

Convention (direct_forecast.TIMING_CONVENTION): origin t0 is the last hour with
measurements; the plan covers t0+1..t0+24 and t0+1 is executed. In
run_rolling_milp terms the decision index is t0+1.

Each check is run twice:
  * the direct path (must be clean), and
  * the old 1-step path (a 24 h slice of 1-step forecasts), which the detector
    must flag. That proves the tests have teeth.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MG_LOAD", "v2")

from scripts_v2 import cost_model as cm
from scripts_v2 import direct_forecast as D
from scripts_v2 import milp_dispatch as mp
from scripts_v2.build_dataset import build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
YEAR = 2025
H = D.HORIZON
MEASURED = ["PV_power_kW", "total_load_kW", "GHI_Wm2", "T2M", "CLRSKY_SFC_SW_DWN"]
N_ORIGINS = int(os.environ.get("LEAK_N_ORIGINS", "200"))
N_MILP = int(os.environ.get("LEAK_N_MILP", "20"))
# Small models: the property under test is causality, not accuracy.
FAST = dict(n_estimators=40, max_depth=4, learning_rate=0.2, subsample=1.0,
            colsample_bytree=1.0, random_state=0, n_jobs=4, verbosity=0)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def frame():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(DATA / "synthetic_load_lv2.csv", parse_dates=["datetime"])
    df = pv[["datetime"] + MEASURED[:1] + MEASURED[2:]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    assert (df.datetime.diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    return df


@pytest.fixture(scope="module")
def year_idx(frame):
    idx = np.flatnonzero(frame.datetime.dt.year.to_numpy() == YEAR)
    assert len(idx) in (8760, 8784)
    return idx


@pytest.fixture(scope="module")
def origins(year_idx):
    # Origins whose full 24 h plan stays inside the series; include the
    # year boundary (origin = last hour of 2024) explicitly.
    rng = np.random.default_rng(20261016)
    pool = year_idx[(year_idx + H) <= year_idx[-1]]
    o = rng.choice(pool[1:], size=N_ORIGINS - 1, replace=False)
    return np.sort(np.concatenate([[year_idx[0] - 1], o])).astype(int)


@pytest.fixture(scope="module")
def models(frame):
    """One model per lead, trained on years <= YEAR-2 only (walk-forward rule)."""
    train = np.flatnonzero(frame.datetime.dt.year.to_numpy() <= YEAR - 2)
    return D.fit_models_on_targets(frame, train, FAST)


@pytest.fixture(scope="module")
def feats(frame):
    return {k: D.build_horizon_features(frame, k) for k in range(1, H + 1)}


# ------------------------------------------------------------------ paths
def direct_plan(models, feats_by_k, origin):
    pv = np.empty(H)
    ld = np.empty(H)
    for k in range(1, H + 1):
        row = feats_by_k[k].iloc[origin + k]
        for out, short in ((pv, "PV"), (ld, "load")):
            x = row[D.features_for(short, k)].to_numpy(dtype=float).reshape(1, -1)
            out[k - 1] = max(float(models[(short, k)].predict(x)[0]), 0.0)
    return pv, ld


def leaky_plan(models, feats1, origin):
    """The pre-fix path: lead-1 (= old 1-step) model on rows t0+1..t0+24."""
    rows = feats1.iloc[origin + 1: origin + 1 + H]
    pv = np.clip(models[("PV", 1)].predict(
        rows[D.features_for("PV", 1)].to_numpy(dtype=float)), 0, None)
    ld = np.clip(models[("load", 1)].predict(
        rows[D.features_for("load", 1)].to_numpy(dtype=float)), 0, None)
    return pv, ld


def spoil(frame, origin, seed):
    rng = np.random.default_rng(seed)
    s = frame.copy()
    fut = s.index > origin
    n = int(fut.sum())
    scale = {"PV_power_kW": 90, "total_load_kW": 40, "GHI_Wm2": 1100,
             "T2M": 40, "CLRSKY_SFC_SW_DWN": 1100}
    for c in MEASURED:
        s.loc[fut, c] = rng.random(n) * scale[c]
    return s


def spoiled_feats(frame, origin, seed, leads):
    # Only a window around the origin is needed; slicing keeps this fast and
    # cannot hide a leak (all rows > origin are randomised before slicing).
    s = spoil(frame, origin, seed)
    lo = max(0, origin - 400)
    sub = s.iloc[lo: origin + H + 2].reset_index(drop=True)
    return {k: D.build_horizon_features(sub, k) for k in leads}, lo


def plans_differ_after_spoil(models, frame, feats, origin, path):
    if path == "direct":
        a = direct_plan(models, feats, origin)
        f2, lo = spoiled_feats(frame, origin, origin, range(1, H + 1))
        b = direct_plan(models, f2, origin - lo)
    else:
        a = leaky_plan(models, feats[1], origin)
        f2, lo = spoiled_feats(frame, origin, origin, [1])
        b = leaky_plan(models, f2[1], origin - lo)
    same = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    return not same, a, b


# ------------------------------------------------------- a) perturbation
def test_direct_forecasts_ignore_future_actuals(models, frame, feats, origins):
    bad = [int(o) for o in origins
           if plans_differ_after_spoil(models, frame, feats, int(o), "direct")[0]]
    assert not bad, f"direct forecast changed when post-origin data was randomised at {bad[:10]}"


def test_direct_first_step_milp_decision_ignores_future_actuals(models, frame, feats, origins):
    rng = np.random.default_rng(7)
    pick = rng.choice(origins, size=min(N_MILP, len(origins)), replace=False)
    for o in pick:
        differ, (pv1, ld1), (pv2, ld2) = plans_differ_after_spoil(
            models, frame, feats, int(o), "direct")
        assert not differ
        kw = dict(z_init=0, time_limit=10.0, mip_gap=1e-3)
        s1 = mp.solve(pv1, ld1, 65.0, cm.CORRECTED, **kw)
        s2 = mp.solve(pv2, ld2, 65.0, cm.CORRECTED, **kw)
        assert s1.diesel[0] == s2.diesel[0] and bool(s1.running[0]) == bool(s2.running[0])


def test_production_trajectory_equals_per_origin_forecast(models, frame, feats, year_idx, origins):
    """predict_leads + assemble_trajectory (what leak_check/run uses) == the
    per-origin forecast that the perturbation test proved causal."""
    pv, ld = D.predict_leads(models, frame, YEAR)
    for o in origins[:50]:
        o = int(o)
        dec = o + 1
        if dec < year_idx[0] or dec + H - 1 > year_idx[-1]:
            continue
        a = direct_plan(models, feats, o)
        assert np.allclose(D.assemble_trajectory(pv, dec), a[0], rtol=0, atol=1e-9)
        assert np.allclose(D.assemble_trajectory(ld, dec), a[1], rtol=0, atol=1e-9)


def test_perturbation_detector_flags_the_old_one_step_path(models, frame, feats, origins):
    """Teeth: the same detector must fire on the pre-fix path."""
    flagged = sum(plans_differ_after_spoil(models, frame, feats, int(o), "leaky")[0]
                  for o in origins[:50])
    assert flagged == len(origins[:50])


# ---------------------------------------------------------- b) timestamp
def _index_encoded(frame):
    """Every measured column replaced by its own row index (its timestamp)."""
    e = frame.copy()
    pos = np.arange(len(e), dtype=float)
    for c in ["PV_power_kW", "total_load_kW", "GHI_Wm2", "T2M"]:
        e[c] = pos
    return e


def _max_source(col, value):
    """Latest row a feature value was read from, given index-encoded inputs."""
    if "_roll" in col:
        w = int(col.split("_roll")[1].split("_")[0])
        if col.endswith("_mean"):
            return value + (w - 1) / 2.0
        return None                      # std of consecutive ints: covered by NaN test
    return value


def latest_input_row(feat_row, cols):
    m = -np.inf
    for c in cols:
        if c in D.CALENDAR or c.startswith("clearsky"):
            continue
        src = _max_source(c, float(feat_row[c]))
        if src is not None:
            m = max(m, src)
    return m


def nan_poison_ok(frame, origin, k, cols):
    s = frame.copy()
    s.loc[s.index > origin, MEASURED] = np.nan
    lo = max(0, origin - 400)
    sub = s.iloc[lo: origin + H + 2].reset_index(drop=True)
    row = D.build_horizon_features(sub, k).iloc[origin + k - lo] if k else None
    return bool(np.isfinite(row[cols].to_numpy(dtype=float)).all())


def test_direct_inputs_are_stamped_at_or_before_origin(frame, origins):
    enc = _index_encoded(frame)
    fk = {k: D.build_horizon_features(enc, k) for k in range(1, H + 1)}
    times = frame.datetime.to_numpy()
    for o in origins:
        o = int(o)
        for k in range(1, H + 1):
            cols = sorted(set(D.features_for("PV", k) + D.features_for("load", k)))
            latest = latest_input_row(fk[k].iloc[o + k], cols)
            assert latest <= o + 1e-9, f"origin {o} lead {k}: input row {latest}"
            assert times[int(round(latest))] <= times[o]
    # Columns the index encoding cannot resolve (std, clear-sky index): NaN poison.
    for o in origins[:25]:
        for k in (1, 2, 6, 12, 24):
            cols = sorted(set(D.features_for("PV", k) + D.features_for("load", k)))
            assert nan_poison_ok(frame, int(o), k, cols), f"origin {o} lead {k}"


def test_timestamp_detector_flags_the_old_one_step_path(frame, origins):
    enc = _index_encoded(frame)
    f1 = build_features(enc.assign(CLRSKY_SFC_SW_DWN=frame.CLRSKY_SFC_SW_DWN))
    cols = sorted(set(D.features_for("PV", 1) + D.features_for("load", 1)))
    for o in origins[:50]:
        o = int(o)
        # Plan step s (s=0..23) of the old path reads the lead-1 row t0+1+s.
        latest = max(latest_input_row(f1.iloc[o + 1 + s], cols) for s in range(H))
        assert latest > o, "detector failed to see the 1-step leak"
        assert latest == o + H - 1
