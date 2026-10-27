"""AUDIT of the V4.0 locked outputs (not collected by pytest by default).

Kept as evidence: these checks fail on the V4.0 locked dispatches (511 starts)
and the V4.0 box folds. Box only (needs /workspace/dispatch_fix_2026-10-18/locked
and folds). Run explicitly:  cd tests && python -m pytest audit_v40_locked.py
Its window test re-runs the CURRENT code, so window tests now pass.
The fixed code is checked by test_dispatch_rules.py.
"""
"""Do the EXECUTED dispatches obey the rules the paper states (eqs. 10-13, 16-18)?

Verify-first tests for mock-review item 1 (dispatch rules), 2026-10-18.
These tests are meant to FAIL where the executed dispatch breaks a stated rule.
Nothing in the controllers is changed here.

Data:
  * locked 2025 dispatches from Build (511 starts):  /workspace/dispatch_fix_2026-10-18/locked
  * box walk-forward folds 2022-2025, 33.75 GHS (output only, no flag): .../folds
  * a 3-week instrumented re-run of the locked configuration (2025-01-04..24),
    logging the MILP plan and the executed hour side by side (run_window()).
"""
import numpy as np
import pytest

from dispatch_audit import (EPS, RATED, RHO_MIN, T_UP, T_DN, blocks, short_on_blocks,
                            short_off_gaps, below_min_load, out_without_flag, starts,
                            load_locked, load_fold, run_window)

YEARS = [2022, 2023, 2024, 2025]
THETA_2025 = 0.80


@pytest.fixture(scope="session")
def win():
    return run_window()


@pytest.fixture(scope="session")
def milp():
    return load_locked("roll24")


@pytest.fixture(scope="session")
def heur():
    return load_locked("heuristic")


def _fmt(idx, n=3):
    return ", ".join(str(i) for i in list(idx)[:n])


# ------------------------------------------------------------- 1. minimum up time
def test_min_up_milp_window(win):
    bad = short_on_blocks(win.g.values > EPS, T_UP)
    assert not bad, (f"{len(bad)} one-hour generator runs in 3 weeks, e.g. "
                     f"{[str(win.time[s]) for s, _ in bad[:3]]}")


def test_min_up_milp_locked_2025(milp):
    bad = short_on_blocks(milp["diesel"] > EPS)
    assert not bad, f"{len(bad)} executed on-blocks shorter than {T_UP} h in 2025"


@pytest.mark.parametrize("year", YEARS)
def test_min_up_milp_folds(year):
    d, _ = load_fold(year, "roll24")
    bad = short_on_blocks(d["diesel"] > EPS)
    assert not bad, f"{year}: {len(bad)} on-blocks shorter than {T_UP} h"


def test_min_up_heuristic(heur):
    assert not short_on_blocks(heur["diesel"] > EPS)


# ----------------------------------------------------------- 2. minimum down time
def test_min_down_milp_window(win):
    bad = short_off_gaps(win.g.values > EPS, T_DN)
    assert not bad, f"{len(bad)} one-hour off-gaps in 3 weeks"


def test_min_down_milp_locked_2025(milp):
    bad = short_off_gaps(milp["diesel"] > EPS)
    assert not bad, f"{len(bad)} executed off-gaps shorter than {T_DN} h in 2025"


@pytest.mark.parametrize("year", YEARS)
def test_min_down_milp_folds(year):
    d, _ = load_fold(year, "roll24")
    bad = short_off_gaps(d["diesel"] > EPS)
    assert not bad, f"{year}: {len(bad)} off-gaps shorter than {T_DN} h"


def test_min_down_heuristic(heur):
    assert not short_off_gaps(heur["diesel"] > EPS)


# ------------------------------------------------- 3. minimum load 30% when on
def test_min_load_milp_window(win):
    bad = below_min_load(win.g.values)
    assert not len(bad), f"{len(bad)} run-hours below 30% of rating in 3 weeks"


def test_min_load_milp_locked_2025(milp):
    bad = below_min_load(milp["diesel"])
    assert not len(bad), f"{len(bad)} run-hours below {RHO_MIN:.0%} in 2025"


@pytest.mark.parametrize("year", YEARS)
def test_min_load_milp_folds(year):
    d, _ = load_fold(year, "roll24")
    bad = below_min_load(d["diesel"])
    assert not len(bad), f"{year}: {len(bad)} run-hours below {RHO_MIN:.0%}"


def test_min_load_heuristic_locked_2025(heur):
    bad = below_min_load(heur["diesel"])
    assert not len(bad), f"{len(bad)} heuristic run-hours below {RHO_MIN:.0%}"


@pytest.mark.parametrize("year", YEARS)
def test_min_load_heuristic_folds(year):
    d, _ = load_fold(year, "heuristic")
    bad = below_min_load(d["diesel"])
    assert not len(bad), f"{year}: {len(bad)} heuristic run-hours below {RHO_MIN:.0%}"


def test_heuristic_holds_theta(heur):
    """Paper §IV-E: the heuristic 'holds at no less than a fraction theta of rated output'."""
    g = heur["diesel"]
    bad = np.flatnonzero((g > EPS) & (g < THETA_2025 * RATED - 1e-6))
    assert not len(bad), f"{len(bad)} heuristic run-hours below theta={THETA_2025:.0%}"


# -------------------------------- 4. generator never runs when commitment is off
def test_no_output_without_commitment_window(win):
    bad = out_without_flag(win.g.values, win.flag.values)
    assert not len(bad), f"{len(bad)} hours generating while the plan says off"


def test_no_output_without_commitment_locked_2025(milp):
    bad = out_without_flag(milp["diesel"], milp["running"])
    assert not len(bad), f"{len(bad)} hours with output > 0 and running flag false"


def test_start_count_reconciliation(milp):
    """Not a rule test: pins down exactly where 511 (output) and 503 (flag) come from."""
    g = milp["diesel"] > EPS
    f = milp["running"].astype(bool)
    fs = f & ~np.r_[False, f[:-1]]
    ob = blocks(g)
    fallback_only = sum(1 for s, L in ob if fs[s:s + L].sum() == 0)
    merged = sum(int(fs[s:s + L].sum()) - 1 for s, L in ob if fs[s:s + L].sum() > 1)
    assert starts(f) == 503 and starts(g) == 511
    assert starts(g) == starts(f) + fallback_only - merged == 503 + 11 - 3


# ------------------------------- 5. battery discharge in committed hours (eq. 16)
def test_discharge_follows_eq16_in_committed_hours(win):
    """Eq. (16) discharges by merit order in every hour; eq. (18) lets the generator
    serve what is left. The code blocks discharge in committed hours (deficit <= rating)."""
    soc_prev = np.r_[np.nan, win.soc.values[:-1]]
    soc_prev[0] = load_locked("roll24")["soc"][71]
    eq16 = np.minimum(win.deficit.values,
                      np.maximum(soc_prev - win.floor.values, 0.0) * 0.95)
    blocked = win.flag.values & (eq16 > 1e-6) & (win.dis.values < EPS)
    assert not blocked.sum(), (f"{int(blocked.sum())} committed hours where eq. (16) "
                               f"would discharge but the executor discharges 0")


# ------------------------------------------------------------- 6. reserve floor
def test_reserve_source_documented(win):
    """Documentation test (passes): R_tau is computed from the FORECAST trajectory
    (controllers.run_rolling_milp -> dispatch_core.feasibility_reserve(pv_fc, ld_fc)),
    and on this system it equals SOC_MIN (13 kWh) in every hour of the window."""
    assert np.allclose(win.floor.values, 13.0)
