"""Unit-commitment dispatch — Phase 5.

Under the corrected cost model the genset has a no-load fuel term, a start-up
cost, a minimum load ratio and minimum up/down times. Dispatch is therefore no
longer a linear program in battery power: it is a mixed-integer program with a
binary commitment variable per hour.

Variables per hour t
    c    charge kW              d    discharge kW
    g    genset output kW       u    unmet kW
    s    spilled PV kW          soc  state of charge kWh
    z    genset running (binary)
    y    start indicator        w    shutdown indicator

Constraints
    pv + d + g + u - c - s = load                       energy balance
    soc_t - soc_{t-1} - eta_c c_t + d_t/eta_d = 0       SOC recursion
    g_t <= P_rated z_t                                  can only produce if on
    g_t >= L_min z_t                                    minimum stable load
    z_t - z_{t-1} = y_t - w_t                           commitment logic
    sum_{k>t-MU} y_k <= z_t                             minimum up time
    sum_{k>t-MD} w_k <= 1 - z_t                         minimum down time
    c_t <= max(pv_t - load_t, 0)                        charge only from surplus

Objective
    fuel:   price * (A P_rated z_t + B g_t)
    starts: C_start y_t
    battery throughput, unmet penalty, curtailment, terminal SOC credit

This is where foresight finally has something to optimise: the optimal policy is
to start the unit once, run it near rated output, serve the load and bank the
surplus, then shut down — rather than idling it for nine hours at 14% load. A
reactive controller cannot do that.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .config import (CHARGE_EFF, DISCHARGE_EFF, MAX_CHARGE_KW, MAX_DISCHARGE_KW,
                     SOC_MAX, SOC_MIN, W_CYCLE)
from .cost_model import CORRECTED, CostModel


class MILPSolution:
    __slots__ = ("charge", "discharge", "diesel", "unmet", "spill", "soc",
                 "running", "starts", "objective", "status", "gap")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def solve(pv, load, soc_init, cost: CostModel = CORRECTED, *,
          terminal_soc_value=None, z_init=0, soc_final_min=None,
          charge_from_surplus_only=False, time_limit=60.0, mip_gap=1e-4,
          hours_on=None, hours_off=None, fixed_z=None, quiet=None):
    """Solve one unit-commitment horizon.

    ``hours_on`` / ``hours_off`` carry the generator's executed history into the
    horizon: the number of consecutive hours it has been running (if
    ``z_init`` is 1) or stopped (if ``z_init`` is 0) up to the hour before
    ``t = 0``. With them, minimum up and down times hold across re-plans: a unit
    started less than ``min_up_h`` hours ago is held on for the remainder, and
    one stopped less than ``min_down_h`` hours ago is held off. Leaving them
    ``None`` reproduces the earlier behaviour (no carried history).

    ``fixed_z`` (array of 0/1, length H, or None) fixes the commitment schedule;
    only the continuous dispatch is then optimised. Used by the ablations.

    ``quiet`` (bool array, length H, or None): hours in which the unit must be
    off (quiet-hours scenario). None leaves the model unchanged.
    """
    pv = np.asarray(pv, float)
    load = np.asarray(load, float)
    H = len(pv)
    N = 9 * H
    ic, id_, ig, iu, is_, isoc, iz, iy, iw = (np.arange(H) + k * H for k in range(9))
    G = cost.genset

    if terminal_soc_value is None:
        # Value stored energy at the genset's marginal (not average) fuel cost,
        # net of the throughput cost still to be paid to get it out.
        terminal_soc_value = G.marginal_cost_ghs_per_kwh

    # ---------------- objective ----------------
    obj = np.zeros(N)
    obj[ig] = G.b_l_per_kwh * G.fuel_price_ghs_per_l
    obj[iz] = G.no_load_l_per_h * G.fuel_price_ghs_per_l
    obj[iy] = G.start_cost_ghs
    obj[iu] = cost.w_unmet
    obj[is_] = cost.w_curtail
    obj[ic] = cost.battery_ghs_per_kwh_throughput / 2.0
    obj[id_] = cost.battery_ghs_per_kwh_throughput / 2.0
    # Same non-degenerate terminal credit as lp_dispatch: cycle cost stays on
    # c_t and d_t. Subtracting only half a throughput coefficient from the
    # terminal price cancelled that cost and left a tied face. Subtracting the
    # full coefficient already in the objective breaks the tie toward use.
    # Start cost often selects a vertex anyway; this keeps the two planners
    # consistent when the start term does not.
    thru = cost.battery_ghs_per_kwh_throughput
    obj[isoc[-1]] = -(terminal_soc_value - thru) * DISCHARGE_EFF

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0

    def add(idx_arrays, coefs, lo, hi, nrows):
        nonlocal r
        base = r + np.arange(nrows)
        for idx, cf in zip(idx_arrays, coefs):
            idx = np.asarray(idx)
            keep = idx >= 0
            rows.append(base[keep])
            cols.append(idx[keep])
            vals.append(np.broadcast_to(np.asarray(cf, float), idx.shape)[keep])
        lb.append(np.broadcast_to(np.asarray(lo, float), (nrows,)))
        ub.append(np.broadcast_to(np.asarray(hi, float), (nrows,)))
        r += nrows

    t = np.arange(H)
    prev = t - 1                       # -1 marks "no previous hour"

    # energy balance
    add([ic, id_, ig, iu, is_], [-1, 1, 1, 1, -1], load - pv, load - pv, H)

    # SOC recursion
    rhs = np.zeros(H)
    rhs[0] = float(soc_init)
    add([isoc, ic, id_, np.where(prev >= 0, isoc[0] + prev, -1)],
        [1.0, -CHARGE_EFF, 1.0 / DISCHARGE_EFF, -1.0], rhs, rhs, H)

    # genset output bounded by commitment
    add([ig, iz], [1.0, -G.rated_kw], -np.inf, 0.0, H)          # g <= P z
    if cost.commitment and G.min_load_ratio > 0:
        add([ig, iz], [-1.0, G.min_load_kw], -np.inf, 0.0, H)   # g >= Lmin z

    if cost.commitment:
        # commitment logic: z_t - z_{t-1} - y_t + w_t = 0
        rhs = np.zeros(H)
        rhs[0] = float(z_init)
        add([iz, iy, iw, np.where(prev >= 0, iz[0] + prev, -1)],
            [1.0, -1.0, 1.0, -1.0], rhs, rhs, H)

        # minimum up time:  sum_{k=t-MU+1..t} y_k <= z_t
        if G.min_up_h > 1:
            idxs, coefs = [iz], [-1.0]
            for lag in range(G.min_up_h):
                idxs.append(np.where(t - lag >= 0, iy[0] + t - lag, -1))
                coefs.append(1.0)
            add(idxs, coefs, -np.inf, 0.0, H)

        # minimum down time: sum_{k=t-MD+1..t} w_k <= 1 - z_t
        if G.min_down_h > 1:
            idxs, coefs = [iz], [1.0]
            for lag in range(G.min_down_h):
                idxs.append(np.where(t - lag >= 0, iw[0] + t - lag, -1))
                coefs.append(1.0)
            add(idxs, coefs, -np.inf, 1.0, H)

    A = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(r, N)).tocsr()
    cons = LinearConstraint(A, np.concatenate(lb), np.concatenate(ub))

    # ---------------- bounds ----------------
    xl = np.zeros(N)
    xu = np.empty(N)
    # Under the corrected cost model the genset is allowed to charge the pack:
    # running a unit that must run anyway near rated output and banking the
    # surplus is the entire mechanism that makes foresight valuable, and real
    # hybrid controllers do exactly this. Under the flat model it is never worth
    # doing, and restricting charging to PV surplus keeps the comparison against
    # the published baseline policy exact.
    # With commitment, 'surplus only' still admits the surplus forced by the
    # minimum stable load, else a held-on unit has nowhere to put it.
    # Only the part of minimum load the net deficit cannot take is forced.
    forced = np.maximum(G.min_load_kw - np.maximum(load - pv, 0.0), 0.0) \
        if cost.commitment else 0.0
    xu[ic] = np.minimum(MAX_CHARGE_KW, np.maximum(pv - load, 0.0) + forced) \
        if charge_from_surplus_only else MAX_CHARGE_KW
    xu[id_] = MAX_DISCHARGE_KW
    xu[ig] = G.rated_kw
    xu[iu] = np.inf
    # Curtailment is PV that was generated and not used. Bounding spill by pv
    # stops the solver from satisfying the minimum-load constraint by dumping
    # genset output into the spill variable at the curtailment price.
    xu[is_] = pv
    xl[isoc], xu[isoc] = SOC_MIN, SOC_MAX
    xu[iz] = xu[iy] = xu[iw] = 1.0
    if soc_final_min is not None:
        xl[isoc[-1]] = max(SOC_MIN, float(soc_final_min))
    if cost.commitment:
        # Executed history carried across re-plans (minimum up / down times).
        if z_init and hours_on is not None:
            hold = max(int(G.min_up_h) - int(hours_on), 0)
            xl[iz[:min(hold, H)]] = 1.0
        if (not z_init) and hours_off is not None:
            hold = max(int(G.min_down_h) - int(hours_off), 0)
            xu[iz[:min(hold, H)]] = 0.0
        if quiet is not None:
            q = np.asarray(quiet, bool)[:H]
            xu[iz[:len(q)][q]] = 0.0
            xl[iz[:len(q)][q]] = 0.0
        if fixed_z is not None:
            fz = np.asarray(fixed_z, float)[:H]
            xl[iz[:len(fz)]] = fz
            xu[iz[:len(fz)]] = fz

    integrality = np.zeros(N)
    if cost.commitment:
        integrality[iz] = 1
        integrality[iy] = 1
        integrality[iw] = 1

    res = milp(c=obj, constraints=cons, bounds=Bounds(xl, xu),
               integrality=integrality,
               options=dict(time_limit=time_limit, mip_rel_gap=mip_gap))
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")

    x = res.x
    z = np.round(x[iz]).astype(bool) if cost.commitment else (x[ig] > 1e-9)
    return MILPSolution(charge=x[ic], discharge=x[id_], diesel=x[ig],
                        unmet=x[iu], spill=x[is_], soc=x[isoc],
                        running=z, starts=np.round(x[iy]).astype(int),
                        objective=float(res.fun), status=res.status,
                        gap=getattr(res, "mip_gap", None))


def solve_year_blocked(pv, load, soc_init, cost: CostModel = CORRECTED, *,
                       block_h=744, overlap_h=48, time_limit=120.0,
                       mip_gap=1e-4, verbose=True, quiet=None):
    """Perfect-foresight annual optimum under unit commitment.

    A single 8,760-hour MILP has 26,280 binaries and does not close in
    reasonable time. Solving in monthly blocks with a look-ahead overlap and
    linking SOC and commitment state gives a tight upper bound on the true
    optimum: it is optimal within each block given the state it inherits, and
    the overlap keeps the block boundary from distorting the end of each month.

    Reported as an upper bound on the achievable optimum, never as an exact
    lower bound on cost. The flat-model oracle remains an exact LP.
    """
    pv = np.asarray(pv, float)
    load = np.asarray(load, float)
    n = len(pv)
    out = {k: np.zeros(n) for k in
           ("charge", "discharge", "diesel", "unmet", "spill", "soc")}
    running = np.zeros(n, bool)

    soc = float(soc_init)
    z = 0
    t0 = 0
    while t0 < n:
        t_end = min(t0 + block_h, n)
        t_look = min(t_end + overlap_h, n)
        qkw = {} if quiet is None else {"quiet": np.asarray(quiet, bool)[t0:t_look]}
        sol = solve(pv[t0:t_look], load[t0:t_look], soc, cost,
                    z_init=z, time_limit=time_limit, mip_gap=mip_gap, **qkw)
        k = t_end - t0
        sl = slice(t0, t_end)
        out["charge"][sl] = sol.charge[:k]
        out["discharge"][sl] = sol.discharge[:k]
        out["diesel"][sl] = sol.diesel[:k]
        out["unmet"][sl] = sol.unmet[:k]
        out["spill"][sl] = sol.spill[:k]
        out["soc"][sl] = sol.soc[:k]
        running[sl] = sol.running[:k]
        soc = float(sol.soc[k - 1])
        z = int(sol.running[k - 1])
        if verbose:
            print(f"    block {t0:5d}-{t_end:5d}  soc_end {soc:6.1f}  "
                  f"diesel {sol.diesel[:k].sum():8,.0f} kWh  "
                  f"run {running[sl].sum():4d} h", flush=True)
        t0 = t_end

    from .dispatch_core import SimResult
    return SimResult(soc=out["soc"], charge=out["charge"],
                     discharge=out["discharge"], diesel=out["diesel"],
                     unmet=out["unmet"], curtail=out["spill"],
                     pv=pv, load=load), running
