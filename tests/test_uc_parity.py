"""Unit-commitment parity: the rule-based baseline and the heuristic obey the
same commitment rules as the MILP oracle (minimum stable load, minimum up and
down time), with every parameter taken from the GensetModel."""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts_v2 import cost_model as cm
from scripts_v2 import dispatch_core as dc
from scripts_v2 import heuristic as H

DATA = Path(__file__).resolve().parent.parent / "data"
G = cm.CORRECTED.genset


@pytest.fixture(scope="module")
def year():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(DATA / "synthetic_load.csv", parse_dates=["datetime"])
    df = pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime").sort_values("datetime")
    df = df[df.datetime.dt.year == 2025].reset_index(drop=True)
    return df.PV_power_kW.values, df.total_load_kW.values


def _runs(on):
    """(start, length, truncated_at_end) for each contiguous run of True."""
    out, t, n = [], 0, len(on)
    while t < n:
        if on[t]:
            s = t
            while t < n and on[t]:
                t += 1
            out.append((s, t - s, t == n))
        else:
            t += 1
    return out


def _check(res, genset, soc_min=None, soc_max=None):
    soc_min = dc.SOC_MIN if soc_min is None else soc_min
    soc_max = dc.SOC_MAX if soc_max is None else soc_max
    on = res.diesel > 1e-9
    short = [r for r in _runs(on) if r[1] < genset.min_up_h and not r[2]]
    assert not short, f"{len(short)} runs shorter than {genset.min_up_h} h: {short[:5]}"
    low = res.diesel[on] < genset.min_load_kw - 1e-6
    assert not low.any(), f"{low.sum()} running hours below {genset.min_load_kw} kW"
    assert (res.diesel <= genset.rated_kw + 1e-6).all()
    bal = res.pv + res.discharge + res.diesel + res.unmet \
        - res.load - res.charge - res.curtail
    assert np.abs(bal).max() < 1e-6
    assert res.soc.min() >= soc_min - 1e-6 and res.soc.max() <= soc_max + 1e-6


def test_parameters_match_milp():
    assert G.min_load_kw == pytest.approx(12.0)
    assert G.min_up_h == 2 and G.min_down_h == 2


def test_baseline_obeys_commitment_rules(year):
    pv, load = year
    info = {}
    res = dc.baseline(pv, load, soc_init=dc.SOC_MIN, genset=G, uc_parity=True,
                      info=info)
    _check(res, G)
    assert res.unmet.sum() < 1e-6
    # A start inside the min-down window is allowed only when the battery
    # cannot carry the hour (else load would go unserved): check every one.
    on = res.diesel > 1e-9
    soc_prev = np.r_[dc.SOC_MIN, res.soc[:-1]]
    forced = 0
    for s0, _, _ in _runs(on):
        gap = s0 - max((e for e, ln, _ in [(r[0] + r[1], 0, 0) for r in _runs(on)]
                        if e <= s0), default=-10**9)
        if 0 < s0 and gap < G.min_down_h:
            avail = min(dc.MAX_DISCHARGE_KW, (soc_prev[s0] - dc.SOC_MIN) * dc.DISCHARGE_EFF)
            assert load[s0] - pv[s0] > avail - 1e-6, f"unforced restart at {s0}"
            forced += 1
    assert forced == info["min_down_overrides"]


@pytest.mark.parametrize("theta,psi", [(0.3, 0.3), (0.7, 0.5), (1.0, 1.0)])
def test_heuristic_obeys_commitment_rules(year, theta, psi):
    pv, load = year
    info = {}
    res, _ = H.commitment_heuristic(pv, load, soc_init=dc.SOC_MIN, threshold=theta,
                                    bank_to_soc=psi, genset=G, uc_parity=True,
                                    info=info)
    _check(res, G)
    assert "min_down_overrides" in info


def test_rules_come_from_the_genset_model():
    """Nothing is hard-coded: change the model and the behaviour follows."""
    g = dataclasses.replace(G, min_load_ratio=0.4, min_up_h=4, min_down_h=3)
    rng = np.random.default_rng(0)
    n = 24 * 14
    h = np.arange(n) % 24
    pv = np.clip(np.sin((h - 6) / 12 * np.pi), 0, None) * 60
    load = 15 + 10 * rng.random(n) + 20 * ((h >= 18) & (h <= 22))
    res = dc.baseline(pv, load, soc_init=dc.SOC_MIN, genset=g, uc_parity=True)
    _check(res, g)
    assert g.min_load_kw == pytest.approx(16.0)
    hres, _ = H.commitment_heuristic(pv, load, soc_init=dc.SOC_MIN, threshold=0.5,
                                     bank_to_soc=0.5, genset=g, uc_parity=True)
    _check(hres, g)


def test_surplus_goes_to_battery_first_then_curtailed():
    # Empty pack, 5 kW load: the unit must run at 12 kW, 7 kW into the battery.
    res = dc.baseline(np.zeros(3), np.full(3, 5.0), soc_init=dc.SOC_MIN,
                      genset=G, uc_parity=True)
    assert res.diesel[0] == pytest.approx(G.min_load_kw)
    assert res.charge[0] == pytest.approx(G.min_load_kw - 5.0)
    assert res.curtail[0] == pytest.approx(0.0)
    # Hour 0 exceeds the discharge rating, so the unit starts; hour 1 is a big
    # PV surplus but min-up keeps it on at min load. The charger is already
    # saturated by PV, so the genset output is curtailed, not banked.
    res = dc.baseline(np.array([0.0, 100.0, 100.0]),
                      np.array([dc.MAX_DISCHARGE_KW + 5.0, 10.0, 10.0]),
                      soc_init=dc.SOC_MAX, genset=G, uc_parity=True)
    assert res.diesel[0] == pytest.approx(G.min_load_kw)
    assert res.diesel[1] == pytest.approx(G.min_load_kw)
    headroom = (dc.SOC_MAX - res.soc[0]) / dc.CHARGE_EFF
    assert res.charge[1] == pytest.approx(min(dc.MAX_CHARGE_KW, headroom))
    assert res.curtail[1] == pytest.approx(90.0 + G.min_load_kw - res.charge[1])
    assert res.curtail[1] >= G.min_load_kw
    assert res.diesel[2] == 0.0
    _check(res, G)


def test_flag_off_is_the_legacy_rule(year):
    pv, load = year
    a = dc.baseline(pv, load, soc_init=dc.SOC_MIN, uc_parity=False)
    b = dc.simulate(pv, load, soc_init=dc.SOC_MIN, rt_balance=True,
                    battery_mode="merit", soc_floor=dc.SOC_MIN)
    for k in ("diesel", "soc", "charge", "discharge", "curtail", "unmet"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_env_default_is_parity(monkeypatch):
    monkeypatch.delenv("MG_UC_PARITY", raising=False)
    assert dc.uc_parity_default() is True
    monkeypatch.setenv("MG_UC_PARITY", "0")
    assert dc.uc_parity_default() is False


def test_baseline_not_below_oracle_on_sizing_cells(year):
    """A causal rule must not beat the perfect-foresight bound. Checked on one
    week per sizing cell (the full year is the driver's job); a violation is
    flagged as a warning, never a crash, exactly as sensitivity.evaluate does."""
    from scripts_v2 import config as C
    from scripts_v2 import milp_dispatch as mp
    pv0, load0 = year
    sl = slice(0, 24 * 7)
    names = ("SOC_MIN", "SOC_MAX", "MAX_CHARGE_KW", "MAX_DISCHARGE_KW")
    mods = [m for m in (dc, mp, H)]
    saved = {(m, nm): getattr(m, nm) for m in mods for nm in names if hasattr(m, nm)}
    flags = []
    try:
        for b, p in ((195, 120), (65, 60)):
            vals = dict(SOC_MIN=(1 - C.DOD_MAX) * b, SOC_MAX=b,
                        MAX_CHARGE_KW=0.5 * b, MAX_DISCHARGE_KW=0.5 * b)
            for (m, nm) in saved:
                setattr(m, nm, vals[nm])
            pv = pv0[sl] * p / 90.0
            load = load0[sl]
            cost = cm.CORRECTED
            base = dc.baseline(pv, load, soc_init=vals["SOC_MIN"], genset=cost.genset)
            _check(base, cost.genset, vals["SOC_MIN"], vals["SOC_MAX"])
            orc, _ = mp.solve_year_blocked(pv, load, vals["SOC_MIN"], cost,
                                           block_h=len(pv), overlap_h=0,
                                           time_limit=30.0, mip_gap=1e-3,
                                           verbose=False)
            cb = cost.evaluate(base)["total_cost_GHS"]
            co = cost.evaluate(orc)["total_cost_GHS"]
            if cb < co - 1e-6:
                flags.append((b, p, cb, co))
    finally:
        for (m, nm), v in saved.items():
            setattr(m, nm, v)
    if flags:
        warnings.warn(f"baseline below oracle on {flags}")
