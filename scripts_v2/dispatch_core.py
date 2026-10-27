"""The one canonical dispatch timestep.

Every simulator in this codebase — rule-based baseline, GA planner, LP planner,
executor, oracle — calls `step()`. There is exactly one implementation of the
physics, and it carries a permanent energy-balance assertion.

The assertion is not a debugging aid. It runs on every timestep of every
experiment. If it fires, the model is wrong; it must never be disabled to make
a run pass.

Control hierarchy
-----------------
`step()` supports the two layers a real deployment has:

  planning layer   decides an SOC trajectory target (a reserve floor) and a
                   genset commitment, on forecasts
  execution layer  sees actual PV and load and dispatches battery and diesel by
                   merit order to meet the ACTUAL deficit, constrained only by
                   the envelope the planner set

`battery_mode` and `rt_balance` select how much of that loop is closed:

  battery_mode="setpoint"  battery follows the planned power setpoint
  battery_mode="merit"     battery is dispatched greedily down to `soc_floor`
  rt_balance=False         diesel follows the planned setpoint (bounded by need)
  rt_balance=True          diesel is recomputed against the actual residual

Open-loop commitment is ("setpoint", False). Full hierarchical balancing is
("merit", True), which is what Phase 4 configuration 10 uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (CHARGE_EFF, DIESEL_MARGINAL_COST, DIESEL_MAX_KW,
                     DISCHARGE_EFF, MAX_CHARGE_KW, MAX_DISCHARGE_KW, SOC_MAX,
                     SOC_MIN, TOL, W_CURTAIL, W_CYCLE, W_DIESEL, W_UNMET,
                     BATTERY_CAPACITY_KWH, DOD_MAX, DIESEL_FUEL_RATE,
                     DIESEL_PRICE_GHS)


def step(pv, load, soc, c_cmd, d_cmd, g_cmd, *,
         rt_balance=True, battery_mode="setpoint", soc_floor=None,
         diesel_max=DIESEL_MAX_KW, allow_genset_charging=False,
         no_discharge=False):
    """One hour of dispatch. Returns (soc, c, d, g, unmet, curtail_pv).

    All quantities are kW over a one-hour step, so they are numerically kWh.

    `allow_genset_charging` lets the genset run above the residual deficit and
    bank the surplus in the battery. It defaults off, which reproduces the
    published baseline policy (charge from PV surplus only) exactly.

    `no_discharge` blocks battery discharge for this hour. The hierarchical
    controller uses it in hours where the plan has committed the genset to run:
    the genset serves the load and charges the pack, so the battery must not be
    simultaneously discharging to meet the same deficit.
    """
    floor = SOC_MIN if soc_floor is None else max(SOC_MIN, float(soc_floor))
    # A floor above the current SOC must not force a negative discharge budget.
    floor = min(floor, SOC_MAX)

    if battery_mode == "merit":
        d_cap, c_cap = MAX_DISCHARGE_KW, MAX_CHARGE_KW
    else:
        d_cap, c_cap = max(d_cmd, 0.0), max(c_cmd, 0.0)
    if no_discharge:
        d_cap = 0.0

    # Battery discharge: rating, SOC envelope, and never more than the actual deficit.
    d = min(d_cap, MAX_DISCHARGE_KW,
            max((soc - floor) * DISCHARGE_EFF, 0.0),
            max(load - pv, 0.0))

    resid = max(load - pv - d, 0.0)
    headroom = max((SOC_MAX - soc) / CHARGE_EFF, 0.0)
    pv_surplus = max(pv - load, 0.0)

    # Charging from PV can only draw on real surplus after the load is served.
    c_pv = min(c_cap, MAX_CHARGE_KW, pv_surplus, headroom)

    # The genset serves the residual deficit.
    serve = min(resid, diesel_max) if rt_balance \
        else min(max(g_cmd, 0.0), diesel_max, resid)

    # It may additionally run above that deficit and bank the surplus in the
    # battery. Under the flat cost model this is never worth doing and the
    # default keeps it off, preserving the published baseline policy exactly.
    # Under the corrected cost model it is the whole point: a genset that must
    # run at all is far cheaper per kWh near rated output than idling, so the
    # surplus is banked rather than wasted. `g_cmd` is the commanded *total*
    # output, so the banked part is whatever it asks for above the deficit.
    bank = 0.0
    if allow_genset_charging:
        bank = min(max(max(g_cmd, 0.0) - serve, 0.0),
                   max(diesel_max - serve, 0.0),
                   max(MAX_CHARGE_KW - c_pv, 0.0),
                   max(headroom - c_pv, 0.0))

    c = c_pv + bank
    g = serve + bank

    soc = float(np.clip(soc + c * CHARGE_EFF - d / DISCHARGE_EFF, SOC_MIN, SOC_MAX))
    unmet = max(load - pv - d - serve, 0.0)
    # Curtailment counts PV that could not be absorbed, never genset output.
    curtail_pv = pv_surplus - c_pv

    assert abs((pv + d + g + unmet) - (load + c + curtail_pv)) < TOL, \
        "energy balance violated"
    return soc, c, d, g, unmet, curtail_pv


# --------------------------------------------------------------------------
@dataclass
class SimResult:
    soc: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    diesel: np.ndarray
    unmet: np.ndarray
    curtail: np.ndarray
    pv: np.ndarray
    load: np.ndarray

    def check_soc_conservation(self, soc_start, tol=1e-6):
        """Reconcile sum(c)*CE - sum(d)/DE against soc_end - soc_start.

        Only valid when the SOC clip never binds; `simulate` guarantees that by
        construction because `step` bounds c and d by the remaining headroom.
        """
        booked = self.charge.sum() * CHARGE_EFF - self.discharge.sum() / DISCHARGE_EFF
        realised = self.soc[-1] - soc_start
        assert abs(booked - realised) < tol, \
            f"SOC conservation violated: booked {booked:.9f} vs realised {realised:.9f}"
        return booked, realised

    def metrics(self, hours_per_year=8760.0):
        n_years = len(self.load) / hours_per_year
        total_load = self.load.sum()
        total_unmet = self.unmet.sum()
        total_diesel = self.diesel.sum()
        fuel_l = total_diesel * DIESEL_FUEL_RATE
        cost = fuel_l * DIESEL_PRICE_GHS
        return {
            "hours": int(len(self.load)),
            "LPSP_pct": 100.0 * total_unmet / total_load,
            "diesel_kWh_per_year": total_diesel / n_years,
            "diesel_cost_GHS_per_year": cost / n_years,
            "diesel_litres_per_year": fuel_l / n_years,
            "renewable_fraction_pct":
                100.0 * (total_load - total_unmet - total_diesel) / total_load,
            "curtailed_kWh_per_year": self.curtail.sum() / n_years,
            "battery_cycles_per_year":
                self.discharge.sum() / (BATTERY_CAPACITY_KWH * DOD_MAX) / n_years,
            "objective": objective(self),
        }


def objective(res: SimResult, terminal_soc_value=0.0, soc_start=None):
    """The published weighted objective, optionally with a terminal SOC credit.

    `terminal_soc_value` is a price in GHS/kWh applied to the stored energy the
    run ends with. Setting it to DIESEL_MARGINAL_COST values the terminal state
    at the fuel it displaces, which is what Phase 2 requires.
    """
    thru = (res.charge.sum() + res.discharge.sum()) / 2.0
    val = (W_DIESEL * res.diesel.sum() * DIESEL_MARGINAL_COST
           + W_UNMET * res.unmet.sum()
           + W_CYCLE * thru
           + W_CURTAIL * res.curtail.sum())
    if terminal_soc_value:
        ref = SOC_MIN if soc_start is None else soc_start
        val -= terminal_soc_value * (res.soc[-1] - ref) * DISCHARGE_EFF
    return val


# --------------------------------------------------------------------------
def simulate(pv, load, *, soc_init, c_cmd=None, d_cmd=None, g_cmd=None,
             rt_balance=True, battery_mode="setpoint", soc_floor=None,
             diesel_max=DIESEL_MAX_KW, allow_genset_charging=False,
             no_discharge=None):
    """Run `step` over a whole series. Commands default to greedy merit order.

    `soc_floor` may be a scalar or a per-hour array (the planner's reserve
    trajectory).
    """
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    n = len(pv)
    assert len(load) == n

    def _as_array(x, default):
        if x is None:
            return np.full(n, default, dtype=float)
        x = np.asarray(x, dtype=float)
        return np.full(n, float(x), dtype=float) if x.ndim == 0 else x

    # Greedy defaults: ask for everything, let `step` bound it by real need.
    c_cmd = _as_array(c_cmd, MAX_CHARGE_KW)
    d_cmd = _as_array(d_cmd, MAX_DISCHARGE_KW)
    g_cmd = _as_array(g_cmd, DIESEL_MAX_KW)
    floors = _as_array(soc_floor, SOC_MIN)
    blocks = (np.zeros(n, dtype=bool) if no_discharge is None
              else np.asarray(no_discharge, dtype=bool))
    dmax = _as_array(diesel_max, DIESEL_MAX_KW)

    soc = np.empty(n)
    c_out = np.empty(n)
    d_out = np.empty(n)
    g_out = np.empty(n)
    u_out = np.empty(n)
    k_out = np.empty(n)

    s = float(soc_init)
    for t in range(n):
        s, c, d, g, u, k = step(
            pv[t], load[t], s, c_cmd[t], d_cmd[t], g_cmd[t],
            rt_balance=rt_balance, battery_mode=battery_mode,
            soc_floor=floors[t], diesel_max=dmax[t],
            allow_genset_charging=allow_genset_charging,
            no_discharge=bool(blocks[t]))
        soc[t], c_out[t], d_out[t], g_out[t], u_out[t], k_out[t] = s, c, d, g, u, k

    res = SimResult(soc=soc, charge=c_out, discharge=d_out, diesel=g_out,
                    unmet=u_out, curtail=k_out, pv=pv, load=load)
    res.check_soc_conservation(float(soc_init))
    return res


def feasibility_reserve(pv, load, *, diesel_max=DIESEL_MAX_KW, from_hour=0):
    """The SOC the executor must hold back, derived from feasibility.

    Only energy that *nothing else can supply* has to be reserved: hours where
    PV plus the genset at full rating still fall short of demand. Everything else
    can be produced on demand, so holding it back does not buy reliability, it
    only stops the executor from running merit order.

    This replaces an earlier rule that used the plan's low-water mark over the
    horizon. That rule had no physical justification, and it bound above SOC_MIN
    on roughly 5% of decision points on both demand series -- harmless on
    load_v1, but costing 0.12-0.47% on load_v2, which broke the invariant that
    hierarchical execution ties the oracle at every re-planning interval.

    Under this system the genset covers demand in all but one hour of the test
    year, so the reserve is SOC_MIN and the executor is free. Under a genset
    outage or derating it becomes positive, which is the behaviour a reserve
    ought to have.
    """
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    shortfall = np.maximum(load[from_hour:] - pv[from_hour:] - diesel_max, 0.0)
    need = float(shortfall.sum()) / DISCHARGE_EFF
    return float(min(SOC_MIN + need, SOC_MAX))


def uc_parity_default():
    """Unit-commitment parity for the rule-based controllers (default on).

    MG_UC_PARITY=0 restores the legacy rules (genset follows the residual
    deficit, no minimum stable load, no minimum up/down time) for
    back-compatibility and for reproducing pre-v42b results.
    """
    import os
    return os.environ.get("MG_UC_PARITY", "1").strip() != "0"


def baseline(pv, load, *, soc_init, diesel_max=DIESEL_MAX_KW, genset=None,
             uc_parity=None, info=None):
    """The rule-based reactive controller: PV -> battery -> diesel, merit order.

    Legacy (``uc_parity=False``): greedy commands and real-time balancing
    reproduce the original Script 03 policy exactly.

    Parity (default, see `uc_parity_default`): the commitment rules the MILP
    oracle obeys, all read from ``genset`` (a cost_model.GensetModel; default
    the corrected model). A running unit delivers at least
    ``genset.min_load_kw`` and stays on at least ``min_up_h`` hours; it stops
    only once the pack can carry the current deficit for ``min_down_h`` hours.
    A start inside the minimum-down window happens only when load would
    otherwise go unserved and is counted in info["min_down_overrides"].
    Output above the residual deficit first displaces battery discharge, then
    charges the battery; the rest is curtailed.
    """
    if uc_parity is None:
        uc_parity = uc_parity_default()
    if not uc_parity:
        if info is not None:
            info["min_down_overrides"] = 0
        return simulate(pv, load, soc_init=soc_init, rt_balance=True,
                        battery_mode="merit", soc_floor=SOC_MIN,
                        diesel_max=diesel_max)
    if genset is None:
        from .cost_model import CORRECTED
        genset = CORRECTED.genset
    return _baseline_uc(pv, load, soc_init=soc_init, diesel_max=diesel_max,
                        genset=genset, info=info)


def _baseline_uc(pv, load, *, soc_init, diesel_max, genset, info=None):
    # Battery constants are read from the module at call time so sweeps that
    # patch them (sensitivity2.sizing_sweep) are honoured.
    g_ = globals()
    soc_min, soc_max = g_["SOC_MIN"], g_["SOC_MAX"]
    cmax, dmax_b = g_["MAX_CHARGE_KW"], g_["MAX_DISCHARGE_KW"]
    ce, de = g_["CHARGE_EFF"], g_["DISCHARGE_EFF"]
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    n = len(pv)
    assert len(load) == n
    gm = np.asarray(diesel_max, dtype=float)
    gm = np.full(n, float(gm)) if gm.ndim == 0 else gm
    keys = ("soc", "c", "d", "g", "u", "k")
    out = {k: np.empty(n) for k in keys}
    min_up, min_down = int(genset.min_up_h), int(genset.min_down_h)
    s = float(soc_init)
    on, up, down, overrides = False, 0, 10**9, 0
    for t in range(n):
        gmax = float(gm[t])
        mload = min(float(genset.min_load_kw), gmax)
        deficit = max(load[t] - pv[t], 0.0)
        pv_surplus = max(pv[t] - load[t], 0.0)
        d = min(dmax_b, max((s - soc_min) * de, 0.0), deficit)
        resid = deficit - d
        if on and up >= min_up and resid <= TOL:
            # Stop only if the pack can carry the current deficit through the
            # whole minimum-down window, so a stop never forces a restart.
            if (s - soc_min) * de >= deficit * min_down - TOL:
                on, up, down = False, 0, 0
        if not on and resid > TOL and gmax > 0:
            if down < min_down:
                overrides += 1
            on, up = True, 0
        if on:
            gen = min(max(resid, mload), gmax)
            up += 1
        else:
            gen = 0.0
            down += 1
        unmet = max(resid - gen, 0.0)
        surplus = max(gen - resid, 0.0)
        back = min(surplus, d)          # displace battery discharge first
        d -= back
        surplus -= back
        headroom = max((soc_max - s) / ce, 0.0)
        c = min(cmax, pv_surplus + surplus, headroom)
        curtail = pv_surplus + surplus - c
        s = s + c * ce - d / de
        assert soc_min - 1e-9 <= s <= soc_max + 1e-9, "SOC bound violated"
        s = min(max(s, soc_min), soc_max)
        assert abs((pv[t] + d + gen + unmet) - (load[t] + c + curtail)) < 1e-6, \
            "energy balance violated"
        for k, v in zip(keys, (s, c, d, gen, unmet, curtail)):
            out[k][t] = v
    if info is not None:
        info["min_down_overrides"] = overrides
    res = SimResult(soc=out["soc"], charge=out["c"], discharge=out["d"],
                    diesel=out["g"], unmet=out["u"], curtail=out["k"],
                    pv=pv, load=load)
    res.check_soc_conservation(float(soc_init))
    return res
