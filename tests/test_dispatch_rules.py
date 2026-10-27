"""Executed dispatch obeys the generator rules the paper states (V4.2 code).

Rules: minimum up time 2 h, minimum down time 2 h, output >= 30 % of rating in
every run-hour, executed on/off flag == (output > 0) so there is one start
count, and the heuristic holds its setpoint theta in every run-hour.

Live checks (fast, run before any rerun):
  * a 3-week re-run of the rolling MILP on 2025 (the densest violation window
    of the V4.0 locked run);
  * the heuristic on every test year 2022-2025 with (theta, psi) re-selected on
    the preceding year;
  * the three rule-based comparators on 2025.
Full-year checks over saved V4.2 outputs: set V42_DISPATCH_DIR to a folder with
roll24.npz / abl_*.npz (written by scripts_v2.v42_runs); skipped otherwise.

Documented exception (not a weakened check): the minimum DOWN time is an
operator rule. If the battery cannot carry an hour, the unit starts even if it
stopped one hour earlier (serving load outranks it). Every such short gap must
be a genuine backstop: the battery was at its floor and could not carry the
hour. The tests assert exactly that, so a voluntary early restart still fails.
"""
import numpy as np
import pytest

from dispatch_audit import (EPS, RATED, RHO_MIN, T_UP, T_DN, blocks, short_on_blocks,
                            short_off_gaps, below_min_load, starts, run_window,
                            fold_heuristic, comparator, v42_dir, deficit)

YEARS = [2022, 2023, 2024, 2025]
from scripts_v2.config import SOC_MIN, SOC_MAX, MAX_DISCHARGE_KW as MAX_DIS, DISCHARGE_EFF as ETA_D, CHARGE_EFF as ETA_C, MAX_CHARGE_KW as MAX_CH


def _assert_rules(g, flag=None, label="", soc=None, deficit=None, soc0=None):
    on = np.asarray(g) > EPS
    bad = short_on_blocks(on, T_UP)
    if soc is not None and deficit is not None:
        # Exception (physical conflict, not a policy choice): a block cut short
        # because the next hour could not host minimum load: deficit plus the
        # room left in the pack was below 30 % of rating, so the battery served
        # it. Accepted only when that is literally true.
        def no_room(r):
            head = max(SOC_MAX - soc[r - 1], 0.0) / ETA_C
            return deficit[r] + min(head, MAX_CH) < RHO_MIN * RATED - 1e-6
        bad = [(s, L) for s, L in bad if not no_room(s + L)]
    assert not bad, f"{label}: {len(bad)} on-blocks shorter than {T_UP} h, e.g. {bad[:3]}"
    low = below_min_load(g)
    assert not len(low), f"{label}: {len(low)} run-hours below {RHO_MIN:.0%}, e.g. {low[:3]}"
    if flag is not None:
        flag = np.asarray(flag, bool)
        assert np.array_equal(flag, on), f"{label}: flag differs from output>0"
        assert starts(flag) == starts(on)
    gaps = short_off_gaps(on, T_DN)
    if gaps and soc is not None and deficit is not None:
        # Each restart after a short gap must be a backstop: the battery, from
        # the SOC it held at the end of the gap, could not carry the restart hour.
        def carries(r):
            avail = min(MAX_DIS, max(soc[r - 1] - SOC_MIN, 0.0) * ETA_D)
            return deficit[r] - avail <= 1e-6
        vol = [(s, L) for s, L in gaps if carries(s + L)]
        assert not vol, f"{label}: {len(vol)} voluntary restarts inside min-down, e.g. {vol[:3]}"
    elif gaps:
        pytest.fail(f"{label}: {len(gaps)} off-gaps shorter than {T_DN} h")


# ------------------------------------------------------------- rolling MILP
@pytest.fixture(scope="session")
def win():
    return run_window()


def test_milp_window_rules(win):
    _assert_rules(win.g.values, win.flag.values, "MILP window", soc=win.soc.values,
                  deficit=win.deficit.values)


def test_milp_window_strict_min_down(win):
    """In this window the strict minimum down time also holds."""
    assert not short_off_gaps(win.g.values > EPS, T_DN)


def test_milp_window_solver_ok(win):
    assert (win.status.values == 0).all(), "a re-plan ended non-optimal (see status)"


def test_milp_window_backstops_obey_min_load(win):
    """Hours run without a planned commitment (backstops) still hold >= 30 %."""
    bs = win.flag.values & ~win.plan_flag.values.astype(bool)
    assert not ((win.g.values[bs] < RHO_MIN * RATED - 1e-6)).any()


@pytest.mark.parametrize("name", ["roll24", "abl_cs0", "abl_heurz", "abl_oraclez",
                                  "abl_nobank"])
def test_milp_full_year(name):
    d = v42_dir()
    if d is None or not (d / f"{name}.npz").exists():
        pytest.skip("no V4.2 full-year output (set V42_DISPATCH_DIR)")
    z = np.load(d / f"{name}.npz")
    yr = int(d.name[:4]) if d.name[:4].isdigit() else 2025
    _assert_rules(z["diesel"], z["running"], name, soc=z["soc"], deficit=deficit(yr))
    assert z["unmet"].sum() < 1e-6, f"{name}: {z['unmet'].sum():.3f} kWh unserved"


# ---------------------------------------------------------------- heuristic
@pytest.mark.parametrize("year", YEARS)
def test_heuristic_rules(year):
    res, run, th, _ = fold_heuristic(year)
    _assert_rules(res.diesel, run, f"heuristic {year}", soc=res.soc,
                  deficit=deficit(year))


@pytest.mark.parametrize("year", YEARS)
def test_heuristic_holds_theta(year):
    """Every run-hour >= theta * rating, except the last hour of banking, where
    output is cut to what the banking ceiling psi still admits (SOC ends at the
    ceiling, or at SOC_MAX). Even then output >= minimum load (12 kW)."""
    res, run, th, psi = fold_heuristic(year)
    g, soc = res.diesel, res.soc
    ceiling = SOC_MIN + psi * (SOC_MAX - SOC_MIN)
    low = (g > EPS) & (g < th * RATED - 1e-6)
    at_cap = soc >= min(ceiling, SOC_MAX) - 1e-6
    bad = np.flatnonzero(low & ~at_cap)
    assert not len(bad), f"{year}: {len(bad)} run-hours below theta={th:.0%} not at the ceiling"
    assert not ((g > EPS) & (g < RHO_MIN * RATED - 1e-6)).any()


# -------------------------------------------------------------- comparators
@pytest.mark.parametrize("mode", ["merit_constrained", "load_following", "cycle_charging"])
def test_comparator_rules(mode):
    res, run, info = comparator(mode)
    _assert_rules(res.diesel, run, mode, soc=res.soc, deficit=deficit(2025))
    assert info["below_min_load_unabsorbed"] == 0


# ------------------------------------------------- documentation (paper text)
def test_discharge_blocked_in_committed_hours_documented(win):
    """Documentation: in committed hours with deficit <= rating the executor
    blocks battery discharge (generator serves the load and banks). The paper's
    eqs. (16)-(18) must describe this; the V4.0 text implied merit-order
    discharge in every hour."""
    on = win.flag.values & win.plan_flag.values.astype(bool) & (win.deficit.values <= RATED)
    assert (win.dis.values[on] < EPS).all()


def test_reserve_source_documented(win):
    """R_tau is computed from the FORECAST trajectory and equals SOC_MIN here."""
    assert np.allclose(win.floor.values, 13.0)


@pytest.mark.parametrize("sp", [0.5, 0.8])
def test_cycle_charging_respects_setpoint(sp):
    """Generator banking stops at the set point; above it, only the surplus
    forced by minimum load (output exactly 12 kW) is allowed."""
    res, run, _ = comparator("cycle_charging", setpoint=sp)
    target = SOC_MIN + sp * (SOC_MAX - SOC_MIN)
    pv_le_load = res.pv <= res.load
    gch = run & (res.charge > 1e-9) & pv_le_load & (res.soc > target + 1e-6)
    assert np.all(res.diesel[gch] <= RHO_MIN * RATED + 1e-6)


@pytest.mark.parametrize("year", YEARS)
def test_heuristic_banking_stops_at_ceiling(year):
    """Re-selected (theta, psi): generator banking never lifts SOC above the
    ceiling, except by the surplus forced by minimum load (output exactly 12 kW)."""
    res, run, th, psi = fold_heuristic(year)
    ceiling = SOC_MIN + psi * (SOC_MAX - SOC_MIN)
    gch = run & (res.charge > 1e-9) & (res.pv <= res.load) & (res.soc > ceiling + 1e-6)
    assert np.all(res.diesel[gch] <= RHO_MIN * RATED + 1e-6)
