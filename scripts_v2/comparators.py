"""Additional rule-based comparators (V4.2).

All three obey the same generator rules as the MILP: output >= the minimum
stable load whenever running (surplus banked in the battery within its SOC and
charge limits), and the minimum up time. A stop is never forced before the
unit has run ``min_up_h`` hours. The minimum down time is honoured in the sense
that the unit is never *chosen* to restart early; if the battery cannot carry
an hour, the unit starts regardless (serving load outranks an operator rule)
and the event is counted in ``info["min_down_overrides"]``.

Strategies (HOMER-style definitions; Barley & Winn 1996, "Optimal dispatch
strategy in remote hybrid power systems", Solar Energy 58(4-6):165-179):

* ``merit_constrained`` -- the merit-order baseline made feasible: PV first,
  battery next, generator for the remainder, but at no less than minimum load
  and for at least ``min_up_h`` hours. Battery may discharge while the unit
  runs. Stops once the battery alone could carry the hour.
* ``load_following`` -- the generator, once on, serves the whole net load
  (battery does not discharge into it unless the load exceeds the rating), at
  no less than minimum load; it never charges the battery deliberately, only
  the surplus forced by minimum load is banked. Renewables charge the battery.
  Stops once the battery alone could carry the hour.
* ``cycle_charging`` -- the generator, once on, runs at full rated output,
  serving the load and charging the battery with the rest, until the state of
  charge reaches the set point ``setpoint`` (fraction of usable capacity); then
  stops. The last hour banks only up to the set point (output never below
  minimum load). The set point is a tunable parameter (selected on the preceding year,
  never on the test year).
"""
from __future__ import annotations

import numpy as np

from .config import CHARGE_EFF, DISCHARGE_EFF, MAX_CHARGE_KW, MAX_DISCHARGE_KW, SOC_MAX, SOC_MIN
from .cost_model import CORRECTED, GensetModel
from .dispatch_core import SimResult, step

MODES = ("merit_constrained", "load_following", "cycle_charging")
CC_SETPOINTS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def rule_dispatch(pv, load, *, soc_init, mode, genset: GensetModel = None,
                  setpoint=0.80):
    """Returns (SimResult, running_flags, info)."""
    assert mode in MODES, mode
    genset = genset or CORRECTED.genset
    pv = np.asarray(pv, float)
    load = np.asarray(load, float)
    n = len(pv)
    rated, lmin = genset.rated_kw, genset.min_load_kw
    target = SOC_MIN + setpoint * (SOC_MAX - SOC_MIN)
    out = {k: np.zeros(n) for k in
           ("soc", "charge", "discharge", "diesel", "unmet", "curtail")}
    running = np.zeros(n, dtype=bool)
    soc, on, up, down = float(soc_init), False, 0, 10 ** 6
    overrides = short = 0

    for t in range(n):
        deficit = max(load[t] - pv[t], 0.0)
        d_avail = min(MAX_DISCHARGE_KW, max((soc - SOC_MIN) * DISCHARGE_EFF, 0.0), deficit)
        battery_carries = deficit - d_avail <= 1e-9
        if on and up >= genset.min_up_h:
            stop = (soc >= target - 1e-9) if mode == "cycle_charging" else battery_carries
            if stop:
                on, down = False, 0
        if not on and not battery_carries:
            if down < genset.min_down_h:
                overrides += 1
            on, up = True, 0

        if on:
            if mode == "merit_constrained":
                block, g_cmd = False, lmin
            elif mode == "load_following":
                block, g_cmd = deficit <= rated, lmin
            else:
                # Full output, but bank only up to the set point (never below
                # minimum load), as the heuristic's ceiling.
                room = max((target - soc) / CHARGE_EFF, 0.0)
                block, g_cmd = deficit <= rated, min(rated, max(deficit + room, lmin))
            soc, c, d, g, u, k = step(
                pv[t], load[t], soc, MAX_CHARGE_KW,
                0.0 if block else MAX_DISCHARGE_KW, g_cmd, rt_balance=True,
                battery_mode="merit", soc_floor=SOC_MIN,
                allow_genset_charging=True, no_discharge=block)
            up += 1
            running[t] = g > 1e-9
            short += int(1e-9 < g < lmin - 1e-6)
        else:
            soc, c, d, g, u, k = step(
                pv[t], load[t], soc, MAX_CHARGE_KW, MAX_DISCHARGE_KW, 0.0,
                rt_balance=False, battery_mode="merit", soc_floor=SOC_MIN)
            down += 1
        out["soc"][t], out["charge"][t], out["discharge"][t] = soc, c, d
        out["diesel"][t], out["unmet"][t], out["curtail"][t] = g, u, k

    info = dict(min_down_overrides=overrides, below_min_load_unabsorbed=short)
    return SimResult(pv=pv, load=load, **out), running, info


def select_cc_setpoint(pv_prev, load_prev, soc0, cost, setpoints=CC_SETPOINTS):
    """Cycle-charging set point chosen on the preceding year (lowest cost)."""
    best = None
    for sp in setpoints:
        res, _, _ = rule_dispatch(pv_prev, load_prev, soc_init=soc0,
                                  mode="cycle_charging", genset=cost.genset,
                                  setpoint=sp)
        c = cost.evaluate(res)["total_cost_GHS"]
        if best is None or c < best[1]:
            best = (sp, c)
    return best[0]
