"""Quiet-hours scenario (unit off 23:00-05:00): flag off changes nothing, flag on
keeps planned generation out of quiet hours and counts every emergency run."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from dispatch_audit import WINDOW_START, WINDOW_HOURS, fold_heuristic, deficit

REF = Path(__file__).parent / "ref" / "v42_flag_off_ref.npz"
QUIET = (23, 0, 1, 2, 3, 4)
EPS = 1e-9


def _window(quiet):
    os.environ["MG_LOAD"] = "v2"
    from scripts_v2 import controllers as ct, cost_model as cm, dispatch_core as dc
    from scripts_v2 import walkforward as W
    from scripts_v2.fc_compat import year_trajectory
    d = W.full_series()
    df = d[d.datetime.dt.year == 2025].reset_index(drop=True)
    s = WINDOW_START
    sl = slice(s, s + WINDOW_HOURS)
    traj = year_trajectory(2025, sl)
    nan = np.full(WINDOW_HOURS, np.nan)
    b = dc.baseline(df.PV_power_kW.values[:s], df.total_load_kW.values[:s],
                    soc_init=W.soc_at_year_start(2025),
                    uc_parity=False)  # window-start SOC: legacy rule
    q = df.datetime.dt.hour.isin(QUIET).to_numpy()[sl] if quiet else None
    log = {}
    out, flag = ct.run_rolling_milp(
        df.PV_power_kW.values[sl], df.total_load_kW.values[sl], nan, nan,
        trajectory_at=traj,
        soc_init=float(b.soc[-1]), horizon=24, commit=1,
        cost=cm.corrected("sourced_mean", start_cost_ghs=33.75), hierarchical=True,
        time_limit=10.0, mip_gap=1e-3, log=log, quiet_hours=q)
    dfc = np.maximum(df.total_load_kW.values[sl] - df.PV_power_kW.values[sl], 0.0)
    return out, flag, log, q, float(b.soc[-1]), dfc


def test_heuristic_flag_off_identical(monkeypatch):
    # The reference predates unit-commitment parity: compare on the legacy rule.
    monkeypatch.setenv("MG_UC_PARITY", "0")
    ref = np.load(REF)
    res, run, _, _ = fold_heuristic(2025)
    for k in ("diesel", "soc", "charge", "discharge"):
        assert np.array_equal(getattr(res, k), ref[f"heur_{k}"]), k
    assert np.array_equal(run, ref["heur_running"])


def _box_api():
    try:
        import scripts_v2.direct_cache  # noqa: F401
        return sys.platform.startswith("linux")
    except ImportError:
        return False


@pytest.mark.skipif(not _box_api(),
                    reason="reference MILP window was solved on the Linux box with the "
                           "box replication's direct forecasts")
def test_milp_flag_off_identical():
    ref = np.load(REF)
    out, flag, _, _, _, _ = _window(False)
    assert np.array_equal(out.diesel, ref["win_g"])
    assert np.array_equal(out.soc, ref["win_soc"])
    assert np.array_equal(flag, ref["win_flag"])


@pytest.fixture(scope="module")
def qwin():
    return _window(True)


def test_milp_quiet_no_planned_generation(qwin):
    out, flag, log, q, _, _ = qwin
    assert not np.asarray(log["plan_running"])[q].any()


def test_milp_quiet_emergencies_counted(qwin):
    out, flag, log, q, soc0, dfc = qwin
    qe = flag & q
    assert log["quiet_emergency_hours"] == int(qe.sum())
    assert abs(log["quiet_emergency_kwh"] - out.diesel[qe].sum()) < 1e-6
    assert abs(log["quiet_unmet_kwh"] - out.unmet[q].sum()) < 1e-9
    # Every quiet run-hour is one the battery could not carry from its SOC.
    soc_prev = np.r_[soc0, out.soc[:-1]]
    from scripts_v2.config import SOC_MIN, MAX_DISCHARGE_KW, DISCHARGE_EFF
    avail = np.minimum(MAX_DISCHARGE_KW, np.maximum(soc_prev - SOC_MIN, 0) * DISCHARGE_EFF)
    assert not (qe & (dfc - avail <= 1e-6)).any()


def test_heuristic_quiet():
    os.environ["MG_LOAD"] = "v2"
    from scripts_v2 import walkforward as W, heuristic as H, cost_model as cm
    from scripts_v2.config import SOC_MIN, MAX_DISCHARGE_KW, DISCHARGE_EFF
    cost = cm.corrected("sourced_mean", start_cost_ghs=33.75)
    d = W.full_series()
    yr = d[d.datetime.dt.year == 2025]
    q = yr.datetime.dt.hour.isin(QUIET).to_numpy()
    info = {}
    soc0 = W.soc_at_year_start(2025)
    res, run = H.commitment_heuristic(yr.PV_power_kW.values, yr.total_load_kW.values,
                                      soc_init=soc0, threshold=0.7, bank_to_soc=0.4,
                                      genset=cost.genset, quiet_hours=q, info=info)
    assert info["quiet_run_hours"] == int((run & q).sum())
    # Starts inside quiet hours are emergencies only (battery could not carry).
    dfc = deficit(2025)
    soc_prev = np.r_[soc0, res.soc[:-1]]
    avail = np.minimum(MAX_DISCHARGE_KW, np.maximum(soc_prev - SOC_MIN, 0) * DISCHARGE_EFF)
    st = run & ~np.r_[False, run[:-1]]
    assert not (st & q & (dfc - avail <= 1e-6)).any()
    # A run may carry into 23:00 only to finish its minimum up time.
    carry = run & q & ~(dfc - avail > 1e-6)
    prev_on = np.r_[False, run[:-1]]
    assert (prev_on[carry]).all()
    assert info["quiet_unmet_kwh"] == float(res.unmet[q].sum())
