"""Phase 2 acceptance — the corrected physics.

Mirror image of tests/test_defects.py: every defect confirmed there must be
absent here, and the energy-balance and SOC-conservation invariants must hold on
the full 8,760-hour series.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts_v2 import dispatch_core as dc
from scripts_v2.config import (CHARGE_EFF, DISCHARGE_EFF, DIESEL_MAX_KW,
                               MAX_CHARGE_KW, SOC_INIT, SOC_MAX, SOC_MIN)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def full_series():
    """The whole 2018-2025 actual series, as the original Script 03 simulates it."""
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(DATA / "synthetic_load.csv", parse_dates=["datetime"])
    df = pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    assert len(df) == 70128
    return df


@pytest.fixture(scope="module")
def year_series(full_series):
    """Actual PV and load for the 2025 test year."""
    df = full_series[full_series.datetime.dt.year == 2025]
    assert len(df) == 8760
    return df.PV_power_kW.values, df.total_load_kW.values


# ------------------------------------------------------------ invariants
def test_energy_balance_holds_on_every_hour_of_the_year(year_series):
    """The assertion inside `step` runs on all 8,760 hours and never fires."""
    pv, load = year_series
    res = dc.baseline(pv, load, soc_init=SOC_INIT)
    assert len(res.load) == 8760
    lhs = res.pv + res.discharge + res.diesel + res.unmet
    rhs = res.load + res.charge + res.curtail
    assert np.max(np.abs(lhs - rhs)) < 1e-9


def test_soc_conservation_reconciles_to_1e6(year_series):
    pv, load = year_series
    res = dc.baseline(pv, load, soc_init=SOC_INIT)
    booked, realised = res.check_soc_conservation(SOC_INIT, tol=1e-6)
    assert abs(booked - realised) < 1e-6


def test_soc_stays_inside_envelope(year_series):
    pv, load = year_series
    res = dc.baseline(pv, load, soc_init=SOC_INIT)
    assert res.soc.min() >= SOC_MIN - 1e-9
    assert res.soc.max() <= SOC_MAX + 1e-9


# ------------------------------------------- baseline reproduces the original
def test_baseline_reproduces_original_hour_by_hour(full_series):
    """The baseline was already self-consistent, so routing it through
    `dispatch_core.step` must reproduce Script 03. If it does not, `step` is
    wrong.

    Run over the same 8-year series from the same starting SOC that Script 03
    uses; agreement is then bit-level, not merely within 0.1%. (Simulating the
    2025 slice in isolation instead gives -0.23% on diesel purely because the
    original carries an empty battery into 2025-01-01, not a half-full one.)
    """
    res = dc.baseline(full_series.PV_power_kW.values,
                      full_series.total_load_kW.values, soc_init=SOC_INIT,
                      uc_parity=False)  # reproduces legacy Script 03
    orig = pd.read_csv(DATA / "baseline_simulation.csv", parse_dates=["datetime"])
    orig = orig.sort_values("datetime").reset_index(drop=True)

    for name, got, want in [
        ("diesel", res.diesel, orig.diesel_output_kW.values),
        ("charge", res.charge, orig.battery_charge_kW.values),
        ("discharge", res.discharge, orig.battery_discharge_kW.values),
        ("curtail", res.curtail, orig.curtailed_pv_kW.values),
        ("unmet", res.unmet, orig.unmet_load_kW.values),
        ("soc", res.soc, orig.battery_SOC_kWh.values),
    ]:
        assert np.max(np.abs(got - want)) < 1e-9, f"{name} diverges from Script 03"


def test_baseline_2025_slice_matches_published_within_0p1pct(full_series):
    """And the 2025 slice of that run reproduces the published baseline row."""
    res = dc.baseline(full_series.PV_power_kW.values,
                      full_series.total_load_kW.values, soc_init=SOC_INIT,
                      uc_parity=False)  # reproduces legacy Script 03
    m = (full_series.datetime.dt.year == 2025).values
    orig = pd.read_csv(DATA / "baseline_simulation.csv", parse_dates=["datetime"])
    o = orig[orig.datetime.dt.year == 2025]

    assert res.diesel[m].sum() == pytest.approx(o.diesel_output_kW.sum(), rel=1e-3)
    assert res.curtail[m].sum() == pytest.approx(o.curtailed_pv_kW.sum(), rel=1e-3)
    assert res.discharge[m].sum() == pytest.approx(
        o.battery_discharge_kW.sum(), rel=1e-3)
    assert res.unmet[m].sum() == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------- defects must be absent now
def test_d1_fixed_charging_cannot_exceed_real_surplus():
    """The Phase 1 D1 scenario: no PV, no load, a plan that charges 65 kW."""
    soc, c, d, g, unmet, curtail = dc.step(
        pv=0.0, load=0.0, soc=13.0, c_cmd=65.0, d_cmd=0.0, g_cmd=0.0)
    assert c == 0.0, "charging drew on energy that does not exist"
    assert soc == pytest.approx(13.0), "SOC rose with no generation"
    assert unmet == 0.0 and curtail == 0.0

    res = dc.simulate(np.zeros(24), np.zeros(24), soc_init=13.0,
                      c_cmd=np.full(24, 65.0))
    assert res.charge.sum() == 0.0
    assert res.soc[-1] == pytest.approx(13.0)


def test_d2_fixed_diesel_capped_at_residual():
    """Phase 1 D2 scenario: PV 0, load 10, discharge 2, diesel commanded 40."""
    soc, c, d, g, unmet, curtail = dc.step(
        pv=0.0, load=10.0, soc=65.0, c_cmd=0.0, d_cmd=2.0, g_cmd=40.0,
        rt_balance=False)
    assert d == pytest.approx(2.0)
    assert g == pytest.approx(8.0), "diesel must be capped at the residual gap"
    assert unmet == 0.0


def test_d4_fixed_surplus_diesel_is_not_burned_or_misbooked():
    """Phase 1 D4 scenario: planner commits 20 kW, actual load is 14 kW."""
    soc, c, d, g, unmet, curtail = dc.step(
        pv=0.0, load=14.0, soc=13.0, c_cmd=0.0, d_cmd=0.0, g_cmd=20.0,
        rt_balance=False)
    assert g == pytest.approx(14.0), "surplus fuel should not be burned"
    assert curtail == 0.0, "diesel must never be booked as curtailed PV"

    # And with the real-time balancing layer the result is identical.
    soc2, c2, d2, g2, u2, k2 = dc.step(
        pv=0.0, load=14.0, soc=13.0, c_cmd=0.0, d_cmd=0.0, g_cmd=20.0,
        rt_balance=True)
    assert g2 == pytest.approx(14.0)
    assert k2 == 0.0


def test_curtailment_counts_pv_only():
    """Curtailment is PV that could not be absorbed, never surplus diesel."""
    _, c, d, g, unmet, curtail = dc.step(
        pv=50.0, load=10.0, soc=SOC_MAX, c_cmd=65.0, d_cmd=0.0, g_cmd=0.0)
    assert c == 0.0                       # battery full
    assert curtail == pytest.approx(40.0)  # exactly the unabsorbed PV
    assert g == 0.0


# ------------------------------------------------- hierarchical balancing
def test_merit_mode_ignores_a_starved_battery_setpoint(year_series):
    """A plan that under-uses the battery must not force the genset to cover the
    difference: in merit mode the executor dispatches the battery itself, down
    to the reserve floor the planner set."""
    pv, load = year_series
    starved = np.zeros(len(pv))          # plan says: never discharge

    setpoint = dc.simulate(pv, load, soc_init=SOC_INIT, d_cmd=starved,
                           c_cmd=np.full(len(pv), MAX_CHARGE_KW),
                           rt_balance=True, battery_mode="setpoint")
    merit = dc.simulate(pv, load, soc_init=SOC_INIT, d_cmd=starved,
                        c_cmd=np.full(len(pv), MAX_CHARGE_KW),
                        rt_balance=True, battery_mode="merit", soc_floor=SOC_MIN)

    assert setpoint.discharge.sum() == 0.0
    assert merit.discharge.sum() > 0.0
    assert merit.diesel.sum() < setpoint.diesel.sum(), \
        "merit-order execution should displace diesel with the idle battery"


def test_reserve_floor_is_respected(year_series):
    pv, load = year_series
    floor = 40.0
    res = dc.simulate(pv, load, soc_init=SOC_INIT, rt_balance=True,
                      battery_mode="merit", soc_floor=floor)
    assert res.soc.min() >= floor - 1e-9


def test_merit_plus_rt_balance_equals_baseline(year_series):
    """A controller whose plan carries no useful information must degenerate
    gracefully to the greedy rule and tie the baseline exactly."""
    pv, load = year_series
    base = dc.baseline(pv, load, soc_init=SOC_INIT, uc_parity=False)  # legacy rule
    noise = np.full(len(pv), 7.0)     # an arbitrary, uninformative plan
    degenerate = dc.simulate(pv, load, soc_init=SOC_INIT,
                             c_cmd=noise, d_cmd=noise, g_cmd=noise,
                             rt_balance=True, battery_mode="merit",
                             soc_floor=SOC_MIN)
    assert degenerate.diesel.sum() == pytest.approx(base.diesel.sum())
    assert degenerate.unmet.sum() == pytest.approx(base.unmet.sum())
    assert degenerate.curtail.sum() == pytest.approx(base.curtail.sum())


# ------------------------------------------------------- terminal SOC value
def test_terminal_soc_value_is_available_and_credits_stored_energy(year_series):
    pv, load = year_series
    res = dc.baseline(pv, load, soc_init=SOC_INIT)
    from scripts_v2.config import DIESEL_MARGINAL_COST
    plain = dc.objective(res)
    valued = dc.objective(res, terminal_soc_value=DIESEL_MARGINAL_COST,
                          soc_start=SOC_INIT)
    assert valued != plain
    # Ending above the starting SOC must be worth something, not nothing.
    delta = (res.soc[-1] - SOC_INIT)
    assert valued == pytest.approx(
        plain - DIESEL_MARGINAL_COST * delta * DISCHARGE_EFF)
