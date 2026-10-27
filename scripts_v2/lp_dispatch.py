"""Linear-programming dispatch.

Once diesel, unmet load and curtailment are *dependent* variables rather than
free genes, the dispatch problem is a linear program in battery power alone.

Variables per hour t:  c_t  charge kW
                       d_t  discharge kW
                       g_t  genset kW
                       u_t  unmet kW
                       s_t  spilled (curtailed) PV kW
                       soc_t  state of charge kWh

Constraints:
  energy balance   pv_t + d_t + g_t + u_t - c_t - s_t = load_t
  SOC recursion    soc_t - soc_{t-1} - eta_c c_t + d_t / eta_d = 0

SOC is carried as an explicit variable and the recursion is written as a local
two-term constraint, so the constraint matrix stays sparse and the formulation
scales to a full year. (A cumulative-sum formulation of the same problem is
dense in the lower triangle and does not scale.)

Objective is the published weighted cost, optionally plus a terminal SOC value
so that energy stored at the horizon edge is not free.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .config import (CHARGE_EFF, DIESEL_MARGINAL_COST, DIESEL_MAX_KW,
                     DISCHARGE_EFF, MAX_CHARGE_KW, MAX_DISCHARGE_KW, SOC_MAX,
                     SOC_MIN, W_CURTAIL, W_CYCLE, W_DIESEL, W_UNMET)


class LPSolution:
    __slots__ = ("charge", "discharge", "diesel", "unmet", "spill", "soc",
                 "objective")

    def __init__(self, c, d, g, u, s, soc, obj):
        self.charge, self.discharge, self.diesel = c, d, g
        self.unmet, self.spill, self.soc = u, s, soc
        self.objective = obj


def build(pv, load, soc_init, *, terminal_soc_value=0.0,
          diesel_max=DIESEL_MAX_KW, soc_min=SOC_MIN, soc_max=SOC_MAX,
          w_unmet=W_UNMET):
    """Assemble (obj, A_eq, b_eq, bounds) for the horizon."""
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    H = len(pv)
    N = 6 * H
    ic, id_, ig, iu, is_, isoc = (np.arange(H) + k * H for k in range(6))

    obj = np.zeros(N)
    obj[ig] = W_DIESEL * DIESEL_MARGINAL_COST
    obj[iu] = w_unmet
    obj[ic] = W_CYCLE / 2.0
    obj[id_] = W_CYCLE / 2.0
    obj[is_] = W_CURTAIL
    if terminal_soc_value:
        # Terminal credit, GHS per kWh of SOC. Cycle cost stays on c_t and d_t
        # (obj[ic] = obj[id_] = W_CYCLE/2 above) and is not subtracted here as
        # W_CYCLE/2: that cancellation is exactly degenerate.
        #
        # Compare using 1 kWh electrical now with keeping it to the horizon
        # edge. Discharge now pays W_CYCLE/2, saves diesel of
        # `terminal_soc_value`, and forgoes a credit of lambda/eta_d.
        # Net (use minus hoard) = W_CYCLE/2 + lambda/eta_d - terminal_soc_value.
        #
        #   lambda = terminal_soc_value * eta_d
        #       (credit diesel marginal only) prefers hoarding: net = +W_CYCLE/2.
        #   lambda = (terminal_soc_value - W_CYCLE/2) * eta_d
        #       is a tied face (net = 0). HiGHS then parks the pack at SOC_MAX
        #       and runs the genset through the evening; under 24 h setpoint
        #       execution that vertex is a modelling defect, not a result.
        #   lambda = (terminal_soc_value - W_CYCLE) * eta_d
        #       prefers use: net = -W_CYCLE/2. The extra W_CYCLE/2 is the same
        #       throughput coefficient already in the objective, not a tuned
        #       epsilon. Leftover SOC is still credited at recoverable fuel
        #       value minus a full one-way throughput price, so a rolling
        #       window does not dump the pack as if the terminal state were
        #       free.
        obj[isoc[-1]] = -(terminal_soc_value - W_CYCLE) * DISCHARGE_EFF

    rows, cols, vals = [], [], []
    beq = np.empty(2 * H)

    # --- energy balance: -c + d + g + u - s = load - pv ---
    t = np.arange(H)
    for idx, coef in ((ic, -1.0), (id_, 1.0), (ig, 1.0), (iu, 1.0), (is_, -1.0)):
        rows.append(t)
        cols.append(idx)
        vals.append(np.full(H, coef))
    beq[:H] = load - pv

    # --- SOC recursion: soc_t - soc_{t-1} - eta_c c_t + d_t/eta_d = 0 ---
    r = H + t
    rows.append(r); cols.append(isoc); vals.append(np.ones(H))
    rows.append(r); cols.append(ic); vals.append(np.full(H, -CHARGE_EFF))
    rows.append(r); cols.append(id_); vals.append(np.full(H, 1.0 / DISCHARGE_EFF))
    rows.append(r[1:]); cols.append(isoc[:-1]); vals.append(np.full(H - 1, -1.0))
    beq[H:] = 0.0
    beq[H] = float(soc_init)          # soc_{-1} moved to the right-hand side

    A_eq = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * H, N)).tocsr()

    # Charging draws on PV surplus only, matching dispatch_core.step's default
    # and the published baseline policy; and curtailment can only ever be PV
    # that was generated, so spill is bounded by pv. Without that second bound
    # the genset could satisfy a minimum-load constraint by dumping output into
    # the spill variable at the curtailment price, which is not a thing a diesel
    # generator can do.
    surplus = np.minimum(MAX_CHARGE_KW, np.maximum(pv - load, 0.0))
    bounds = ([(0.0, float(v)) for v in surplus]
              + [(0.0, MAX_DISCHARGE_KW)] * H
              + [(0.0, diesel_max)] * H
              + [(0.0, None)] * H
              + [(0.0, float(v)) for v in pv]
              + [(soc_min, soc_max)] * H)
    return obj, A_eq, beq, bounds, (ic, id_, ig, iu, is_, isoc)


def solve(pv, load, soc_init, **kw):
    """Solve one horizon with HiGHS. Returns an LPSolution."""
    obj, A_eq, b_eq, bounds, idx = build(pv, load, soc_init, **kw)
    r = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(f"LP failed: {r.message}")
    ic, id_, ig, iu, is_, isoc = idx
    return LPSolution(r.x[ic], r.x[id_], r.x[ig], r.x[iu], r.x[is_],
                      r.x[isoc], float(r.fun))


def reserve_floor(pv, load, *, diesel_max=DIESEL_MAX_KW, from_hour=0):
    """Deprecated shim. The single reserve rule lives in
    dispatch_core.feasibility_reserve; this delegates so there is only one."""
    from .dispatch_core import feasibility_reserve
    return feasibility_reserve(pv, load, diesel_max=diesel_max,
                               from_hour=from_hour)
