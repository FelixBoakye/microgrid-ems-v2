"""Receding-horizon controller harness for the attribution ladder.

One loop drives every configuration. What changes between ladder rows is only
which planner produces the plan, which executor applies it, and which defects
are switched on.

Terminology, kept honest per the brief: a controller that plans over `horizon`
hours and applies `commit` of them is a receding-horizon controller only when
commit == 1. With commit == 6 it is *block commitment* running six hours
open-loop, and is labelled as such.
"""
from __future__ import annotations

import time

import numpy as np

from . import dispatch_core as dc
from . import ga as ga2
from . import legacy_executor as legacy
from . import lp_dispatch
from .config import (DIESEL_MARGINAL_COST, DIESEL_MAX_KW, HORIZON, SOC_MIN)
from .dispatch_core import SimResult

_NEED_FORECAST = (
    "no direct forecast was given. Pass trajectory_at, or pass "
    "perfect_foresight=True when the plan is supposed to see the realised "
    "future. A sliced 1-step forecast array is not accepted."
)


# ----------------------------------------------------------------- planners
def plan_ga(pv_fc, load_fc, soc, *, seed=0, clip_diesel=False,
            charge_from_surplus=False, terminal_soc_value=0.0):
    c, d, g, _ = ga2.optimize(pv_fc, load_fc, soc, seed=seed,
                              clip_diesel=clip_diesel,
                              charge_from_surplus=charge_from_surplus,
                              terminal_soc_value=terminal_soc_value)
    return c, d, g, None


def plan_lp(pv_fc, load_fc, soc, *, terminal_soc_value=0.0, **_):
    sol = lp_dispatch.solve(pv_fc, load_fc, soc,
                            terminal_soc_value=terminal_soc_value)
    return sol.charge, sol.discharge, sol.diesel, sol.soc


def make_original_ga_planner(orig_ga):
    """Planner that calls the frozen original 09_ga_optimizer.py verbatim."""
    def _plan(pv_fc, load_fc, soc, *, seed=0, **_):
        c, d, g, _f = orig_ga.optimize(pv_fc, load_fc, soc, seed=seed)
        return c, d, g, None
    return _plan


# ---------------------------------------------------------------- main loop
def run_receding(pv_actual, load_actual, pv_forecast, load_forecast, *,
                 planner, soc_init, horizon=HORIZON, commit=6,
                 executor="core", rt_balance=False, battery_mode="setpoint",
                 use_soc_floor=False, soc_floor_slack=0.0,
                 conserve_energy=True, planner_kw=None,
                 seed_mode="t_start", diesel_max=DIESEL_MAX_KW,
                 trajectory_at=None, perfect_foresight=False):
    """Plan on forecasts, execute against actuals, advance, repeat.

    ``trajectory_at(t0)`` returns the forecast of hours ``t0`` onward issued
    from measurements through ``t0 - 1``. That is the direct-forecast path,
    including the 6 h block-commit rows. ``perfect_foresight=True`` is the
    only way to plan on the realised future, and it is reserved for the rows
    that are labelled as perfect foresight.

    executor="core"    dispatch_core.step (correct physics)
    executor="legacy"  the published executor, defects selectable

    use_soc_floor      pass the planner's SOC trajectory to the executor as a
                       reserve envelope instead of following power setpoints;
                       this is the hierarchical controller the brief specifies
    """
    planner_kw = dict(planner_kw or {})
    pv_actual = np.asarray(pv_actual, float)
    load_actual = np.asarray(load_actual, float)
    pv_forecast = np.asarray(pv_forecast, float)
    load_forecast = np.asarray(load_forecast, float)
    n = len(pv_actual)
    if trajectory_at is None and not perfect_foresight:
        raise RuntimeError("run_receding: " + _NEED_FORECAST)

    out = {k: np.zeros(n) for k in
           ("soc", "charge", "discharge", "diesel", "unmet", "curtail")}
    soc = float(soc_init)

    for t0 in range(0, n, commit):
        t_end = min(t0 + horizon, n)
        if trajectory_at is None:
            pv_fc = pv_forecast[t0:t_end]
            ld_fc = load_forecast[t0:t_end]
        else:
            pv_fc, ld_fc = trajectory_at(t0)
            pv_fc = np.asarray(pv_fc, float)[:t_end - t0]
            ld_fc = np.asarray(ld_fc, float)[:t_end - t0]
            if len(pv_fc) != len(ld_fc) or len(pv_fc) == 0:
                raise ValueError(
                    f"trajectory at {t0} has length {len(pv_fc)}, {len(ld_fc)}")
        if len(pv_fc) < horizon:                      # pad the tail, as the original does
            pad = horizon - len(pv_fc)
            pv_fc = np.concatenate([pv_fc, np.full(pad, pv_fc[-1])])
            ld_fc = np.concatenate([ld_fc, np.full(pad, ld_fc[-1])])

        seed = t0 if seed_mode == "t_start" else planner_kw.get("seed", 0)
        c_p, d_p, g_p, soc_traj = planner(pv_fc, ld_fc, soc, seed=seed,
                                          **planner_kw)

        k = min(commit, n - t0)
        sl = slice(t0, t0 + k)
        pv_a, ld_a = pv_actual[sl], load_actual[sl]

        if executor == "legacy":
            r = legacy.run(pv_a, ld_a, c_p[:k], d_p[:k], g_p[:k], soc,
                           conserve_energy=conserve_energy,
                           rt_balance=rt_balance, diesel_max=diesel_max)
            for key in out:
                out[key][sl] = r[key]
            soc = float(r["soc"][-1])
        else:
            floor = None
            if use_soc_floor:
                # Feasibility-derived reserve: only energy that PV and the genset
                # together cannot supply after the commit window has to be held
                # back. See dispatch_core.feasibility_reserve for why the earlier
                # low-water-mark rule was wrong.
                floor = dc.feasibility_reserve(pv_fc, ld_fc,
                                               diesel_max=diesel_max, from_hour=k)
                floor = max(SOC_MIN, floor - soc_floor_slack)
            res = dc.simulate(pv_a, ld_a, soc_init=soc,
                              c_cmd=c_p[:k], d_cmd=d_p[:k], g_cmd=g_p[:k],
                              rt_balance=rt_balance, battery_mode=battery_mode,
                              soc_floor=floor, diesel_max=diesel_max)
            out["soc"][sl] = res.soc
            out["charge"][sl] = res.charge
            out["discharge"][sl] = res.discharge
            out["diesel"][sl] = res.diesel
            out["unmet"][sl] = res.unmet
            out["curtail"][sl] = res.curtail
            soc = float(res.soc[-1])

    return SimResult(soc=out["soc"], charge=out["charge"],
                     discharge=out["discharge"], diesel=out["diesel"],
                     unmet=out["unmet"], curtail=out["curtail"],
                     pv=pv_actual, load=load_actual)


# ------------------------------------------------- rolling unit commitment
def run_rolling_milp(pv_actual, load_actual, pv_forecast, load_forecast, *,
                     soc_init, horizon=24, commit=1, cost=None,
                     hierarchical=True, time_limit=10.0, mip_gap=1e-3,
                     progress_every=0, trajectory_at=None,
                     perfect_foresight=False, carry_state=True,
                     min_load_backstop=True, fixed_commitment=None,
                     plan_cost=None, allow_genset_charging=True, log=None,
                     quiet_hours=None):
    """Re-plan a unit-commitment MILP on forecasts, execute against actuals.

    This is the controller a site could actually run, as opposed to a single
    long-horizon solve over the whole year with perfect information. The
    planning layer emits a genset commitment and an output target; the execution
    layer sees the real hour and closes the loop.

    The loop index ``t0`` is the hour about to be executed. The plan is built
    before that hour's PV and load are applied, so measurements through hour
    ``t0 - 1`` are known and hour ``t0`` is not. ``trajectory_at(t0)`` returns
    the forecast of hours ``t0`` onward under that rule (see
    ``direct_forecast``). Omitting it raises, unless ``perfect_foresight`` is
    set. A year of 1-step-ahead forecasts is not a substitute: entry ``h`` was
    built with actuals through ``h - 1``.

    In hours the plan commits the genset, the battery is blocked from
    discharging: the unit is there to serve the load and bank the surplus, and
    letting the pack discharge into the same deficit would defeat the point.
    In uncommitted hours the battery runs merit order down to the plan's reserve
    floor, and if it cannot carry the hour the genset starts anyway and serves
    the deficit. That fallback is greedy, and the fuel it burns at low load
    factor is a genuine cost of forecast error rather than something to hide.

    Executed-state handling (defaults; set the flags False to reproduce the
    earlier behaviour):

    * ``carry_state``: the next plan receives the *executed* on/off state and
      how many consecutive hours the unit has been on or off, so minimum up and
      down times hold across re-plans (``milp_dispatch.solve`` hours_on /
      hours_off).
    * ``min_load_backstop``: any executed run-hour (planned or backstop start)
      is commanded at no less than the minimum stable load ``rho_min * P_rated``;
      output above the residual deficit is banked in the battery within its SOC
      and charge limits. A run-hour that still falls below the minimum because
      the battery cannot absorb the surplus (pack full) is instead served by
      the battery with the unit off; any hour still below minimum load is
      counted in ``log``.
    * The returned flags are the executed on/off state (output > 0), so there
      is one start count.

    ``fixed_commitment`` (bool array, length n) fixes the plan's commitment
    schedule (ablations). ``plan_cost`` plans with a different cost model than
    ``cost`` (e.g. start cost 0) while execution is unchanged.
    ``allow_genset_charging=False`` forbids banking generator output.
    ``log`` (dict or None) receives per-re-plan solver status, MIP gap, the
    plan's commitment, and counters.

    ``quiet_hours`` (bool array, length n, or None = scenario off): the plan
    keeps the unit off in those hours, and carried holds never force it on
    there. The executor runs it only as a backstop when the battery cannot
    carry the hour; such hours are counted (``quiet_emergency_hours``).
    """
    from . import milp_dispatch as mp
    from .cost_model import CORRECTED
    cost = cost or CORRECTED

    pv_actual = np.asarray(pv_actual, float)
    load_actual = np.asarray(load_actual, float)
    pv_forecast = np.asarray(pv_forecast, float)
    load_forecast = np.asarray(load_forecast, float)
    n = len(pv_actual)
    if trajectory_at is None and not perfect_foresight:
        raise RuntimeError("run_rolling_milp: " + _NEED_FORECAST)

    out = {k: np.zeros(n) for k in
           ("soc", "charge", "discharge", "diesel", "unmet", "curtail")}
    committed = np.zeros(n, dtype=bool)
    plan_flag = np.zeros(n, dtype=bool)
    status = np.zeros(n, dtype=int)
    gaps = np.full(n, np.nan)
    soc = float(soc_init)
    z = 0
    hours_on, hours_off = 0, 10 ** 6
    pcost = plan_cost or cost
    lmin = cost.genset.min_load_kw if cost.commitment else 0.0
    n_relaxed = 0     # re-plans where a carried hold had to be dropped
    n_short = 0       # run-hours below minimum load the battery could not absorb
    if fixed_commitment is not None:
        fixed_commitment = np.asarray(fixed_commitment, bool)
    if quiet_hours is not None:
        quiet_hours = np.asarray(quiet_hours, bool)
    t_start = time.time()

    for t0 in range(0, n, commit):
        t_end = min(t0 + horizon, n)
        if trajectory_at is None:
            pv_fc = pv_forecast[t0:t_end]
            ld_fc = load_forecast[t0:t_end]
        else:
            pv_fc, ld_fc = trajectory_at(t0)
            pv_fc = np.asarray(pv_fc, float)[:t_end - t0]
            ld_fc = np.asarray(ld_fc, float)[:t_end - t0]
            if len(pv_fc) != len(ld_fc) or len(pv_fc) == 0:
                raise ValueError(
                    f"trajectory at {t0} has length {len(pv_fc)}, {len(ld_fc)}")
        if len(pv_fc) < horizon:
            pad = horizon - len(pv_fc)
            pv_fc = np.concatenate([pv_fc, np.full(pad, pv_fc[-1])])
            ld_fc = np.concatenate([ld_fc, np.full(pad, ld_fc[-1])])
        if not np.isfinite(pv_fc).all() or not np.isfinite(ld_fc).all():
            raise ValueError(f"non-finite forecast at t0={t0}")

        skw = dict(z_init=z)
        if carry_state:
            skw.update(hours_on=hours_on if z else None,
                       hours_off=None if z else hours_off)
        if quiet_hours is not None:
            qz = quiet_hours[t0:t_end]
            skw["quiet"] = np.concatenate([qz, np.zeros(horizon - len(qz), bool)])
            if qz[0] and "hours_on" in skw:
                skw["hours_on"] = None      # quiet hour overrides the min-up hold
        if fixed_commitment is not None:
            fz = fixed_commitment[t0:t_end].astype(float)
            skw["fixed_z"] = np.concatenate([fz, np.full(horizon - len(fz), fz[-1])])
        if not allow_genset_charging:
            skw["charge_from_surplus_only"] = True
        try:
            sol = mp.solve(pv_fc, ld_fc, soc, pcost, time_limit=time_limit,
                           mip_gap=mip_gap, **skw)
        except RuntimeError as exc:
            # A carried hold (or fixed schedule) can be infeasible, e.g. a
            # unit held on at minimum load with a full battery and no load
            # to take it. Drop the hold for this re-plan and count it.
            if "infeasible" not in str(exc).lower():
                raise
            n_relaxed += 1
            for key in ("hours_on", "hours_off", "fixed_z"):
                skw.pop(key, None)
            sol = mp.solve(pv_fc, ld_fc, soc, pcost, time_limit=time_limit,
                           mip_gap=mip_gap, **skw)

        k = min(commit, n - t0)
        sl = slice(t0, t0 + k)
        floor = (dc.feasibility_reserve(pv_fc, ld_fc,
                                        diesel_max=cost.genset.rated_kw,
                                        from_hour=k)
                 if hierarchical else None)

        # Block battery discharge only where the genset can actually carry the
        # hour on its own. Blocking unconditionally causes unmet load whenever
        # the real deficit exceeds the genset rating -- the peakier the demand,
        # the worse it gets. The reactive heuristic already guards this the same
        # way (`block = deficit <= rated`); the MILP executor did not, and on a
        # household-heavy load it dropped 29 kWh/yr that the baseline served.
        deficit = np.maximum(load_actual[sl] - pv_actual[sl], 0.0)
        can_cover = deficit <= cost.genset.rated_kw
        sim_kw = dict(c_cmd=np.full(k, 1e9), d_cmd=np.full(k, 1e9),
                      rt_balance=True,
                      battery_mode="merit" if hierarchical else "setpoint",
                      soc_floor=floor, allow_genset_charging=allow_genset_charging,
                      no_discharge=sol.running[:k] & can_cover)
        g_cmd = np.asarray(sol.diesel[:k], float).copy()
        if min_load_backstop and lmin > 0:
            g_cmd = np.where(sol.running[:k], np.maximum(g_cmd, lmin), g_cmd)
            if carry_state and z and hours_on < cost.genset.min_up_h and not (
                    quiet_hours is not None and quiet_hours[t0]):
                # Executed history says the unit must stay on this hour.
                g_cmd[0] = max(g_cmd[0], lmin)
            if not allow_genset_charging:
                # No deliberate banking: a running unit is commanded at the
                # minimum load only, so the battery takes just the surplus that
                # minimum load forces.
                g_cmd = np.where(g_cmd > 1e-9, lmin, 0.0)
                sim_kw["allow_genset_charging"] = True
        res = dc.simulate(pv_actual[sl], load_actual[sl], soc_init=soc,
                          g_cmd=g_cmd, **sim_kw)
        if min_load_backstop and lmin > 0:
            g_out = res.diesel
            low = (g_out > 1e-9) & (g_out < lmin - 1e-6)
            if low.any():
                # A run-hour below minimum load: typically a backstop start the
                # plan did not schedule. Command the minimum and bank the rest
                # (charging from the generator is always allowed for the
                # surplus forced by minimum load).
                g_cmd = np.where(low, np.maximum(g_cmd, lmin), g_cmd)
                res = dc.simulate(pv_actual[sl], load_actual[sl], soc_init=soc,
                                  g_cmd=g_cmd, **{**sim_kw,
                                                  "allow_genset_charging": True})
                g_out = res.diesel
                low = (g_out > 1e-9) & (g_out < lmin - 1e-6)
                if low.any():
                    # The battery took the load and has too little room for
                    # the rest: let the unit serve the deficit instead (no
                    # discharge in that hour), so its output reaches the minimum.
                    # Only where the unit alone can cover the actual deficit;
                    # above its rating the battery must keep helping.
                    # There the battery supplies only the excess over the rating.
                    nd = np.asarray(sim_kw["no_discharge"]) | (low & can_cover)
                    over = low & ~can_cover
                    d_cmd = np.where(over, deficit - cost.genset.rated_kw, 1e9)
                    res = dc.simulate(pv_actual[sl], load_actual[sl], soc_init=soc,
                                      g_cmd=g_cmd, **{**sim_kw, "no_discharge": nd,
                                                      "d_cmd": d_cmd,
                                                      "battery_mode": "setpoint",
                                                      "allow_genset_charging": True})
                    g_out = res.diesel
                    low = (g_out > 1e-9) & (g_out < lmin - 1e-6)
                if low.any():
                    # The battery cannot take the surplus (it is full), so it
                    # can carry the hour itself: stop the unit and let the
                    # battery discharge by merit order instead.
                    g_cmd = np.where(low, 0.0, g_cmd)
                    nd = np.asarray(sim_kw["no_discharge"]) & ~low
                    res = dc.simulate(pv_actual[sl], load_actual[sl], soc_init=soc,
                                      g_cmd=g_cmd, **{**sim_kw, "no_discharge": nd,
                                                      "allow_genset_charging": True})
                    g_out = res.diesel
                    n_short += int(((g_out > 1e-9) & (g_out < lmin - 1e-6)).sum())

        for key, arr in (("soc", res.soc), ("charge", res.charge),
                         ("discharge", res.discharge), ("diesel", res.diesel),
                         ("unmet", res.unmet), ("curtail", res.curtail)):
            out[key][sl] = arr
        on_exec = np.asarray(res.diesel) > 1e-9
        committed[sl] = on_exec
        plan_flag[sl] = sol.running[:k]
        status[sl] = int(getattr(sol, "status", -1) if sol.status is not None else -1)
        gaps[sl] = np.nan if sol.gap is None else float(sol.gap)
        soc = float(res.soc[-1])
        if carry_state:
            for on in on_exec:
                if on:
                    hours_on = hours_on + 1 if z else 1
                    z = 1
                else:
                    hours_off = hours_off + 1 if not z else 1
                    z = 0
        else:
            z = int(sol.running[k - 1])

        if progress_every and (t0 // commit) % progress_every == 0:
            el = time.time() - t_start
            print(f"    t={t0:5d}/{n}  soc {soc:6.1f}  "
                  f"diesel so far {out['diesel'][:t0 + k].sum():9,.0f} kWh  "
                  f"[{el / 60:.1f} min]", flush=True)

    if log is not None:
        if quiet_hours is not None:
            qe = committed & quiet_hours
            log.update(quiet_emergency_hours=int(qe.sum()),
                       quiet_emergency_kwh=float(out["diesel"][qe].sum()),
                       quiet_unmet_kwh=float(out["unmet"][quiet_hours].sum()))
        log.update(plan_running=plan_flag, solver_status=status, mip_gap=gaps,
                   below_min_load_unabsorbed=n_short, holds_relaxed=n_relaxed,
                   backstop_hours=int((committed & ~plan_flag).sum()))
    return SimResult(soc=out["soc"], charge=out["charge"],
                     discharge=out["discharge"], diesel=out["diesel"],
                     unmet=out["unmet"], curtail=out["curtail"],
                     pv=pv_actual, load=load_actual), committed
