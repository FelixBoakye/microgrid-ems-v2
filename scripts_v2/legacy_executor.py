"""The published executor, reproduced exactly — defects included.

This is NOT physics and nothing else in the codebase may call it. It exists so
the attribution ladder can measure what each defect costs, which requires being
able to run the pipeline *with* the defect present. `dispatch_core.step` remains
the single implementation of the actual physics.

Reproduces `apply_locked_decisions_to_actual` from 10_mpc_evaluation.py:

  D1  `ch` is added to SOC but never drawn from supply, so the battery can
      absorb energy that was never generated
  D4  diesel is replayed as a rigid setpoint against actual conditions rather
      than recomputed against actual need, and
      `curtailed = max(supply - demand - ch, 0)` books surplus *diesel* as
      curtailed *PV*

Each defect is a separate flag so rows 3-5 of the ladder can switch them off one
at a time.
"""
from __future__ import annotations

import numpy as np

from .config import (CHARGE_EFF, DIESEL_MAX_KW, DISCHARGE_EFF, MAX_CHARGE_KW,
                     MAX_DISCHARGE_KW, SOC_MAX, SOC_MIN)


def legacy_step(pv, load, soc, c_cmd, d_cmd, g_cmd, *,
                conserve_energy=False, rt_balance=False,
                diesel_max=DIESEL_MAX_KW):
    """One hour under the published executor's rules.

    conserve_energy=False  reproduces D1: charging is bounded only by rating and
                           SOC headroom, never by surplus that actually exists
    rt_balance=False       reproduces D4: diesel follows the planned setpoint and
                           any surplus is booked as curtailed PV
    """
    c = float(np.clip(c_cmd, 0.0, MAX_CHARGE_KW))
    d = float(np.clip(d_cmd, 0.0, MAX_DISCHARGE_KW))
    g = float(np.clip(g_cmd, 0.0, diesel_max))

    if conserve_energy:
        c = min(c, max(pv - load, 0.0))

    c = min(c, max((SOC_MAX - soc) / CHARGE_EFF, 0.0))
    d = min(d, max((soc - SOC_MIN) * DISCHARGE_EFF, 0.0))
    if conserve_energy:
        d = min(d, max(load - pv, 0.0))

    soc = float(np.clip(soc + c * CHARGE_EFF - d / DISCHARGE_EFF, SOC_MIN, SOC_MAX))

    if rt_balance:
        g = min(max(load - pv - d, 0.0), diesel_max)

    supply = pv + d + g
    unmet = max(load - supply, 0.0)
    # The misbooking is part of D4, not D1: while diesel is still a rigid
    # setpoint, surplus fuel keeps being counted as curtailed PV.
    if rt_balance:
        curtail = max(pv - load - c, 0.0)          # PV only, as it should be
    else:
        curtail = max(supply - load - c, 0.0)      # D4: surplus diesel counted too

    return soc, c, d, g, unmet, curtail


def run(pv, load, c_plan, d_plan, g_plan, soc_init, **kw):
    """Apply a plan hour by hour under the legacy rules."""
    n = len(pv)
    out = {k: np.empty(n) for k in
           ("soc", "charge", "discharge", "diesel", "unmet", "curtail")}
    s = float(soc_init)
    for t in range(n):
        s, c, d, g, u, k = legacy_step(
            pv[t], load[t], s, c_plan[t], d_plan[t], g_plan[t], **kw)
        out["soc"][t], out["charge"][t], out["discharge"][t] = s, c, d
        out["diesel"][t], out["unmet"][t], out["curtail"][t] = g, u, k
    return out
