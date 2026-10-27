"""Phase 6 Step 1 acceptance — the demand model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts_v2 import load_model as LM

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def frames():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    orig = pd.read_csv(DATA / "synthetic_load.csv", parse_dates=["datetime"])
    return pv, orig


def _residual_acf(load, dt, lags=(1, 2, 24)):
    """Nonparametric detrending: mean of log load within month x dow x hour."""
    d = pd.DataFrame({"y": np.log(np.asarray(load, float)), "m": dt.dt.month,
                      "dw": dt.dt.dayofweek, "h": dt.dt.hour})
    r = (d.y - d.groupby(["m", "dw", "h"]).y.transform("mean")).values
    return [float(np.corrcoef(r[:-k], r[k:])[0, 1]) for k in lags], len(r)


def test_load_v1_reproduces_the_delivered_generator(frames):
    """Every published result was produced with load_v1, so it must stay exact."""
    pv, orig = frames
    v1 = LM.generate_v1(pv.datetime)
    for c in ("basic_load_kW", "productive_load_kW", "total_load_kW"):
        assert np.max(np.abs(v1[c].values - orig[c].values)) < 1e-9, c


def test_load_v2_preserves_annual_energy(frames):
    pv, orig = frames
    v1_annual = orig.total_load_kW.sum() / (len(orig) / 8760)
    v2 = LM.generate_v2(pv.datetime, pv.T2M.values, v1_annual_kwh=v1_annual)
    v2_annual = v2.total_load_kW.sum() / (len(v2) / 8760)
    assert abs(v2_annual / v1_annual - 1.0) < 0.01


def test_load_v2_residuals_are_autocorrelated(frames):
    """The point of v2: the residual carries information a forecaster can use."""
    pv, orig = frames
    v1_annual = orig.total_load_kW.sum() / (len(orig) / 8760)
    v2 = LM.generate_v2(pv.datetime, pv.T2M.values, v1_annual_kwh=v1_annual)

    (a1, _, _), n = _residual_acf(orig.total_load_kW, orig.datetime)
    (b1, b2, _), _ = _residual_acf(v2.total_load_kW, v2.datetime)
    thresh = 1.96 / np.sqrt(n)

    assert b1 > 20 * thresh, f"v2 lag-1 {b1:.4f} should be strongly non-zero"
    assert b2 > 5 * thresh
    # v1's is detectable at N=70k but must remain an order of magnitude smaller.
    assert abs(a1) < 0.05
    assert b1 > 20 * abs(a1), f"v2 {b1:.4f} vs v1 {a1:.4f}"


def test_ar1_multiplier_is_mean_preserving():
    rng = np.random.default_rng(0)
    m = LM._ar1(200_000, 0.7, 0.08, rng)
    assert abs(m.mean() - 1.0) < 5e-3
    assert abs(np.corrcoef(m[:-1], m[1:])[0, 1] - 0.7) < 0.05


def test_load_v2_is_deterministic_given_a_seed(frames):
    pv, orig = frames
    a = LM.generate_v2(pv.datetime, pv.T2M.values, seed=7)
    b = LM.generate_v2(pv.datetime, pv.T2M.values, seed=7)
    assert np.array_equal(a.total_load_kW.values, b.total_load_kW.values)


def test_weekend_structure_is_present_in_v2_only(frames):
    pv, orig = frames
    v2 = LM.generate_v2(pv.datetime, pv.T2M.values)
    for df, expect_gap in ((orig, False), (v2, True)):
        d = df.copy()
        d["dw"] = d.datetime.dt.dayofweek
        wk = d[d.dw < 5].productive_load_kW.mean()
        sun = d[d.dw == 6].productive_load_kW.mean()
        gap = 1.0 - sun / wk
        if expect_gap:
            assert gap > 0.4, f"expected a Sunday drop, got {gap:.3f}"
        else:
            assert abs(gap) < 0.05, f"v1 should have no weekly structure, got {gap:.3f}"


def test_temperature_response_is_present_in_v2_only(frames):
    pv, orig = frames
    v2 = LM.generate_v2(pv.datetime, pv.T2M.values)
    # T2M is strongly diurnal, so a global hot/cold decile split rarely puts
    # both groups inside the same hour. Split within each hour instead, which is
    # the correct control: it holds the diurnal shape fixed and varies only
    # temperature.
    t2m = pv.T2M.values
    hour = pv.datetime.dt.hour.values
    for df, expect in ((orig, False), (v2, True)):
        vals = df.basic_load_kW.values
        rel = []
        for h in range(24):
            m = hour == h
            if m.sum() < 200:
                continue
            th = t2m[m]
            med = np.median(th)
            hot, cold = th > med, th < med
            if hot.sum() > 50 and cold.sum() > 50:
                rel.append(vals[m][hot].mean() / vals[m][cold].mean())
        assert len(rel) >= 20, "not enough hours with a usable split"
        r = float(np.mean(rel))
        if expect:
            assert r > 1.02, f"expected hot hours to draw more, ratio {r:.3f}"
        else:
            assert abs(r - 1.0) < 0.02, f"v1 should be temperature-blind, {r:.3f}"
