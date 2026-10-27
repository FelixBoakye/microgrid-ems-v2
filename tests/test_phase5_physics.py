"""Phase 5 physics: genset charging, discharge blocking, and the spill bound.

The corrected cost model only makes sense if the genset can bank surplus into
the battery, and only stays honest if it cannot dump surplus anywhere else.
Both directions are tested here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts_v2 import cost_model as cm
from scripts_v2 import dispatch_core as dc
from scripts_v2 import heuristic as H
from scripts_v2 import lp_dispatch, milp_dispatch as mp
from scripts_v2.config import (CHARGE_EFF, MAX_CHARGE_KW, SOC_MAX, SOC_MIN)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def year():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(DATA / "synthetic_load.csv", parse_dates=["datetime"])
    df = pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime").sort_values("datetime")
    df = df[df.datetime.dt.year == 2025].reset_index(drop=True)
    return df.PV_power_kW.values, df.total_load_kW.values


# ------------------------------------------------------ genset charging
def test_genset_cannot_charge_by_default():
    """The published baseline policy charges from PV surplus only, and that is
    what every flat-cost-model result depends on."""
    _, c, d, g, u, k = dc.step(pv=0.0, load=10.0, soc=65.0,
                               c_cmd=MAX_CHARGE_KW, d_cmd=0.0, g_cmd=40.0,
                               rt_balance=True)
    assert c == 0.0
    assert g == pytest.approx(10.0)      # serves the deficit, banks nothing


def test_genset_charging_banks_surplus_when_enabled():
    """With banking on, a genset commanded above the deficit puts the rest in
    the pack -- and the energy balance still closes."""
    soc0 = 65.0
    soc, c, d, g, u, k = dc.step(pv=0.0, load=10.0, soc=soc0,
                                 c_cmd=MAX_CHARGE_KW, d_cmd=0.0, g_cmd=32.0,
                                 rt_balance=True, allow_genset_charging=True)
    assert g == pytest.approx(32.0)
    assert c == pytest.approx(22.0)      # 32 commanded - 10 serving the load
    assert u == 0.0 and k == 0.0
    assert soc == pytest.approx(soc0 + 22.0 * CHARGE_EFF)


def test_genset_banking_is_bounded_by_headroom():
    soc0 = SOC_MAX - 1.0
    soc, c, d, g, u, k = dc.step(pv=0.0, load=5.0, soc=soc0,
                                 c_cmd=MAX_CHARGE_KW, d_cmd=0.0, g_cmd=40.0,
                                 rt_balance=True, allow_genset_charging=True)
    headroom = (SOC_MAX - soc0) / CHARGE_EFF
    assert c == pytest.approx(headroom)
    assert g == pytest.approx(5.0 + headroom)   # never generates into nowhere
    assert soc == pytest.approx(SOC_MAX)


def test_banked_genset_output_is_never_booked_as_curtailed_pv():
    """The D4 accounting error must not reappear through the banking path."""
    _, c, d, g, u, k = dc.step(pv=0.0, load=5.0, soc=65.0,
                               c_cmd=MAX_CHARGE_KW, d_cmd=0.0, g_cmd=40.0,
                               rt_balance=True, allow_genset_charging=True)
    assert k == 0.0, "no PV was generated, so nothing can be curtailed"


def test_no_discharge_blocks_the_battery():
    soc0 = 100.0
    soc, c, d, g, u, k = dc.step(pv=0.0, load=10.0, soc=soc0,
                                 c_cmd=0.0, d_cmd=65.0, g_cmd=10.0,
                                 rt_balance=True, battery_mode="merit",
                                 no_discharge=True)
    assert d == 0.0
    assert g == pytest.approx(10.0)


def test_energy_balance_holds_with_banking_over_a_year(year):
    pv, load = year
    res = dc.simulate(pv, load, soc_init=SOC_MIN,
                      g_cmd=np.full(len(pv), 28.0), rt_balance=True,
                      battery_mode="merit", allow_genset_charging=True)
    lhs = res.pv + res.discharge + res.diesel + res.unmet
    rhs = res.load + res.charge + res.curtail
    assert np.max(np.abs(lhs - rhs)) < 1e-9
    res.check_soc_conservation(SOC_MIN, tol=1e-6)


# ------------------------------------------------------ the spill bound
def test_lp_cannot_spill_more_than_the_pv_it_generated(year):
    pv, load = year
    sol = lp_dispatch.solve(pv[:168], load[:168], 65.0)
    assert np.all(sol.spill <= pv[:168] + 1e-9)


def test_milp_cannot_dump_genset_output_into_spill(year):
    """Without this bound the solver satisfies the minimum-load constraint by
    spilling at the curtailment price, which no diesel generator can do."""
    pv, load = year
    sol = mp.solve(pv[:48], load[:48], SOC_MIN, cm.CORRECTED)
    assert np.all(sol.spill <= pv[:48] + 1e-6)
    # Everything the genset produced was either served or banked.
    served_or_banked = sol.diesel.sum()
    assert served_or_banked <= (load[:48].sum() + sol.charge.sum() + 1e-6)


def test_milp_terminal_credit_does_not_hoard_through_evening_deficit():
    """Daytime PV surplus, then a 12 h evening deficit. Charge is surplus-only
    so the genset cannot fill the pack. The planner must discharge through the
    evening rather than sit at SOC_MAX and cover the night with diesel.

    Matches the lp_dispatch terminal-credit tie-break: full throughput
    coefficient is netted from the terminal price, not half.
    """
    pv = np.array([40.0] * 12 + [0.0] * 12)
    load = np.array([10.0] * 12 + [20.0] * 12)
    sol = mp.solve(pv, load, 65.0, cm.CORRECTED,
                   charge_from_surplus_only=True, time_limit=30.0, mip_gap=1e-4)
    assert sol.soc[-1] < SOC_MAX - 1.0
    assert sol.discharge[12:].sum() > 1.0
    assert sol.diesel[12:].sum() < 240.0 - 1.0


def test_milp_respects_minimum_load_ratio(year):
    pv, load = year
    sol = mp.solve(pv[:48], load[:48], SOC_MIN, cm.CORRECTED)
    g = cm.CORRECTED.genset
    on = sol.running
    if on.any():
        assert np.all(sol.diesel[on] >= g.min_load_kw - 1e-6)
    assert np.all(sol.diesel[~on] <= 1e-6)


# ------------------------------------------------------ the heuristic
def test_heuristic_serves_all_load(year):
    pv, load = year
    res, run = H.commitment_heuristic(pv, load, soc_init=SOC_MIN,
                                      threshold=0.70, bank_to_soc=0.50)
    assert res.metrics()["LPSP_pct"] == pytest.approx(0.0, abs=1e-9)
    lhs = res.pv + res.discharge + res.diesel + res.unmet
    rhs = res.load + res.charge + res.curtail
    assert np.max(np.abs(lhs - rhs)) < 1e-9


def test_heuristic_raises_load_factor_over_the_baseline(year):
    pv, load = year
    base = dc.baseline(pv, load, soc_init=SOC_MIN, uc_parity=False)
    res, _ = H.commitment_heuristic(pv, load, soc_init=SOC_MIN,
                                    threshold=0.70, bank_to_soc=0.50)
    lf_base = cm.CORRECTED.evaluate(base)["mean_load_factor_pct"]
    lf_heur = cm.CORRECTED.evaluate(res)["mean_load_factor_pct"]
    assert lf_heur > 3 * lf_base, f"{lf_heur:.1f}% vs {lf_base:.1f}%"
    # Under unit-commitment parity the baseline already runs at >= minimum
    # stable load, so the gap narrows but the heuristic must still lead.
    par = dc.baseline(pv, load, soc_init=SOC_MIN, uc_parity=True)
    lf_par = cm.CORRECTED.evaluate(par)["mean_load_factor_pct"]
    assert lf_heur > 1.5 * lf_par, f"{lf_heur:.1f}% vs parity {lf_par:.1f}%"


def test_heuristic_banking_ceiling_is_respected(year):
    """The genset must stop charging at the ceiling, leaving headroom for PV.
    SOC may still exceed it later -- from the sun, which is free."""
    pv, load = year
    ceiling_frac = 0.50
    res, run = H.commitment_heuristic(pv, load, soc_init=SOC_MIN,
                                      threshold=0.70, bank_to_soc=ceiling_frac)
    ceiling = SOC_MIN + ceiling_frac * (SOC_MAX - SOC_MIN)
    genset_charging_hours = run & (res.charge > 1e-9) & (res.pv <= res.load)
    assert np.all(res.soc[genset_charging_hours] <= ceiling + 1e-6)


def test_banking_to_full_is_worse_than_leaving_headroom(year):
    """The literal 'bank until full' rule loses to the baseline; the finding is
    that the ceiling, not the loading threshold, is what makes it work."""
    pv, load = year
    full, _ = H.commitment_heuristic(pv, load, soc_init=SOC_MIN,
                                     threshold=0.70, bank_to_soc=1.00)
    half, _ = H.commitment_heuristic(pv, load, soc_init=SOC_MIN,
                                     threshold=0.70, bank_to_soc=0.50)
    base = dc.baseline(pv, load, soc_init=SOC_MIN)
    c_full = cm.CORRECTED.evaluate(full)["total_cost_GHS"]
    c_half = cm.CORRECTED.evaluate(half)["total_cost_GHS"]
    c_base = cm.CORRECTED.evaluate(base)["total_cost_GHS"]
    assert c_full > c_base, "banking to full should lose to merit order"
    assert c_half < c_base, "banking to half should beat merit order"
    assert full.curtail.sum() > half.curtail.sum()


def test_rolling_milp_never_drops_load_the_baseline_serves():
    """The hierarchical executor blocks battery discharge in hours the plan
    commits the genset. Blocking unconditionally causes unmet load whenever the
    real deficit exceeds the genset rating: on a household-heavy demand series
    it dropped 29 kWh/yr that the merit-order baseline served. The block is now
    conditional on the genset being able to carry the hour.

    The window is constructed rather than taken from load_v1, whose peak deficit
    is 36.8 kW and so can never exhibit the defect.
    """
    from scripts_v2 import controllers as ct
    rated = cm.CORRECTED.genset.rated_kw
    n = 72
    hours = np.arange(n) % 24
    pv = np.where((hours >= 8) & (hours <= 16), 45.0, 0.0)
    load = np.full(n, 8.0)
    load[(hours >= 18) & (hours <= 21)] = rated + 12.0   # 52 kW: genset alone cannot cover
    assert (load - pv).max() > rated

    base = dc.baseline(pv, load, soc_init=SOC_MAX)
    res, _ = ct.run_rolling_milp(pv, load, pv, load, soc_init=SOC_MAX, horizon=24,
                                 commit=1, cost=cm.CORRECTED, hierarchical=True,
                                 time_limit=10.0, mip_gap=1e-3,
                                 perfect_foresight=True)
    assert res.unmet.sum() <= base.unmet.sum() + 1e-6, (
        f"rolling MILP dropped {res.unmet.sum():.3f} kWh where the baseline "
        f"dropped {base.unmet.sum():.3f}")
