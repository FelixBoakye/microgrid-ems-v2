"""A competent reactive heuristic — the mandatory comparator for Phase 5.

The merit-order baseline trims the genset to the residual deficit, which under
the corrected cost model is the worst possible thing to do: a 40 kW unit idling
at 14% load burns 0.81 L/kWh against 0.33 L/kWh at rated output. Comparing a
MILP against *that* would flatter the MILP for a reason that has nothing to do
with foresight.

This rule fixes the obvious failure without any forecast at all:

    when the genset has to start, run it at no less than `threshold` of rated
    output, bank whatever the load does not take into the battery, and shut down
    once the pack is full

It is purely reactive — it sees only the current hour — so any margin the MILP
retains over it is attributable to foresight rather than to loading policy.

If this heuristic captures most of the available gain, that is the finding, and
it should be reported as such: it would mean the practical recommendation for a
rural operator is a controller change costing nothing, not a forecasting stack.
"""
from __future__ import annotations

import numpy as np

from .config import (CHARGE_EFF, DISCHARGE_EFF, MAX_CHARGE_KW,
                     MAX_DISCHARGE_KW, SOC_MAX, SOC_MIN)
from .cost_model import CORRECTED, GensetModel
from .dispatch_core import SimResult, step, uc_parity_default


def commitment_heuristic(pv, load, *, soc_init, threshold=0.70, bank_to_soc=1.0,
                         genset: GensetModel = None, respect_min_up=True,
                         hold_setpoint=True, quiet_hours=None, info=None,
                         uc_parity=None):
    """Reactive high-loading rule. Returns (SimResult, running_flags).

    `threshold` is the minimum load factor the genset is held at while running,
    as a fraction of rated output.

    `bank_to_soc` is the SOC ceiling, as a fraction of usable capacity, that the
    genset is allowed to charge up to before shutting down. It matters enormously
    and it is the parameter the literal "bank until full" rule is missing:
    filling the pack from the genset overnight leaves no headroom for the
    morning's PV, which is then curtailed. Measured on the test year, banking to
    full doubles PV curtailment from 30,458 to 60,330 kWh/yr and pushes 80% of
    the genset's output through the battery.

    A reactive controller cannot know how much sun tomorrow brings, but it can
    stop short of full and leave room for it. That is the whole of the knob.

    ``hold_setpoint`` (default True): every run-hour is held at no less than
    the minimum stable load. Mid-run the unit holds ``threshold``; in the last
    hour before the banking ceiling it holds max(min load, deficit + room left
    below the ceiling), so banking stops at the ceiling. If even minimum load
    would overshoot the ceiling, the run ends an hour early when the minimum up
    time allows and the battery can carry the hour; otherwise only the surplus
    forced by minimum load goes above the ceiling. ``False`` reproduces the
    earlier rule, which trimmed the final hour to the remaining headroom and
    could run it below minimum load.

    ``quiet_hours`` (bool array or None = scenario off): in those hours a
    running unit stops as soon as its minimum up time allows, and a unit runs
    only if the battery cannot carry the hour (emergency: commanded at the
    deficit, at least minimum load, no banking beyond that). ``info`` (dict)
    receives quiet_run_hours, quiet_emergency_hours, quiet_unmet_kwh.

    ``uc_parity`` (default `dispatch_core.uc_parity_default()`, i.e. on unless
    MG_UC_PARITY=0): every running hour delivers at least
    ``genset.min_load_kw`` (the MILP's minimum stable load); output the battery
    cannot absorb is curtailed. Starts inside the minimum-down window (only
    ever forced by load the battery cannot carry) are counted in
    info["min_down_overrides"].
    """
    genset = genset or CORRECTED.genset
    if uc_parity is None:
        uc_parity = uc_parity_default()
    floor_kw = min(genset.min_load_kw, genset.rated_kw)

    def _floor(g, k):
        # Raise a running hour to minimum stable load; the extra is curtailed.
        if uc_parity and 1e-9 < g < floor_kw:
            return floor_kw, k + (floor_kw - g)
        return g, k
    pv = np.asarray(pv, float)
    load = np.asarray(load, float)
    n = len(pv)
    rated = genset.rated_kw
    target_kw = threshold * rated

    out = {k: np.zeros(n) for k in
           ("soc", "charge", "discharge", "diesel", "unmet", "curtail")}
    running = np.zeros(n, dtype=bool)

    # bank_to_soc may be a scalar or a per-hour schedule.
    psi = np.asarray(bank_to_soc, dtype=float)
    psi = np.full(n, float(psi)) if psi.ndim == 0 else psi
    ceilings = SOC_MIN + psi * (SOC_MAX - SOC_MIN)

    soc = float(soc_init)
    on = False
    hours_up = 0
    down, overrides = 10**9, 0
    quiet = None if quiet_hours is None else np.asarray(quiet_hours, bool)
    q_emerg = 0

    for t in range(n):
        ceiling = ceilings[t]
        deficit = max(load[t] - pv[t], 0.0)
        # Headroom the *genset* is allowed to fill, which stops at the ceiling.
        headroom = max((ceiling - soc) / CHARGE_EFF, 0.0)
        # What merit order alone would leave unserved this hour.
        d_avail = min(MAX_DISCHARGE_KW, max((soc - SOC_MIN) * DISCHARGE_EFF, 0.0),
                      deficit)
        resid_merit = max(deficit - d_avail, 0.0)

        qh = quiet is not None and quiet[t]
        if qh and on and ((not respect_min_up) or hours_up >= genset.min_up_h):
            on, down = False, 0
        if qh and not on:
            if resid_merit > 1e-9:
                # Emergency: battery cannot carry the hour.
                q_emerg += 1
                if down < genset.min_down_h:
                    overrides += 1
                soc, c, d, g, u, k = step(
                    pv[t], load[t], soc, MAX_CHARGE_KW, MAX_DISCHARGE_KW,
                    max(resid_merit, genset.min_load_kw), rt_balance=True,
                    battery_mode="merit", soc_floor=SOC_MIN,
                    allow_genset_charging=True)
                g, k = _floor(g, k)
                running[t] = g > 1e-9
            else:
                soc, c, d, g, u, k = step(
                    pv[t], load[t], soc, MAX_CHARGE_KW, MAX_DISCHARGE_KW, 0.0,
                    rt_balance=False, battery_mode="merit", soc_floor=SOC_MIN)
            hours_up = 0
            down = 0 if running[t] else down + 1
            out["soc"][t], out["charge"][t], out["discharge"][t] = soc, c, d
            out["diesel"][t], out["unmet"][t], out["curtail"][t] = g, u, k
            continue
        if not on:
            # Start only when the battery genuinely cannot carry the hour.
            if resid_merit > 1e-9:
                if down < genset.min_down_h:
                    overrides += 1
                on, hours_up = True, 0
        else:
            # Stop once the pack is full: there is nothing left to bank, and
            # from here the battery can carry the load on its own.
            can_stop = (not respect_min_up) or hours_up >= genset.min_up_h
            # Also stop one hour early if even minimum load would push the
            # pack past the ceiling and the battery can carry the hour.
            no_room = hold_setpoint and deficit + headroom < genset.min_load_kw - 1e-9 \
                and resid_merit <= 1e-9
            if (headroom <= 1e-6 or no_room) and can_stop:
                on, down = False, 0

        if on:
            # The genset covers the load and banks the rest, so the battery must
            # not discharge into the same deficit -- unless the load exceeds what
            # the unit can supply, in which case the battery has to help.
            block = deficit <= rated
            # Never generate more than the load takes plus the room left below
            # the banking ceiling: surplus with nowhere to go is wasted fuel.
            if hold_setpoint:
                # Hold theta, but bank only up to the ceiling; never below
                # minimum load (the only permitted overshoot of the ceiling is
                # the surplus forced by minimum load, when the run cannot end).
                g_cmd = min(rated, max(deficit, target_kw),
                            max(deficit + headroom, genset.min_load_kw))
            else:
                g_cmd = min(rated, max(deficit, target_kw), deficit + headroom)
            soc, c, d, g, u, k = step(
                pv[t], load[t], soc, MAX_CHARGE_KW, 0.0 if block else MAX_DISCHARGE_KW,
                g_cmd, rt_balance=True,
                battery_mode="setpoint" if block else "merit",
                soc_floor=SOC_MIN, allow_genset_charging=True, no_discharge=block)
            g, k = _floor(g, k)
            hours_up += 1
            running[t] = g > 1e-9
        else:
            soc, c, d, g, u, k = step(
                pv[t], load[t], soc, MAX_CHARGE_KW, MAX_DISCHARGE_KW, 0.0,
                rt_balance=False, battery_mode="merit", soc_floor=SOC_MIN)
            hours_up = 0
            down += 1

        out["soc"][t], out["charge"][t], out["discharge"][t] = soc, c, d
        out["diesel"][t], out["unmet"][t], out["curtail"][t] = g, u, k

    if info is not None:
        info["min_down_overrides"] = overrides
    if info is not None and quiet is not None:
        info.update(quiet_run_hours=int((running & quiet).sum()),
                    quiet_emergency_hours=q_emerg,
                    quiet_unmet_kwh=float(out["unmet"][quiet].sum()))
    return SimResult(pv=pv, load=load, **out), running


def psi_from_forecast(pv_fc, load_fc, hours, *, decide_hour=18, lookahead=24,
                      psi_min=0.30, psi_max=1.00):
    """Set the banking ceiling each evening from tomorrow's forecast PV surplus.

    The mechanism under test. The fixed-ψ rule banks diesel to the same level
    every night, so on a night before a sunny day it buys with fuel what the sun
    would have given free, and on a night before a cloudy day it under-banks.
    Knowing tomorrow's PV lets a still-purely-reactive controller pick the level.

    The rule: leave exactly enough headroom for the surplus the forecast says
    tomorrow will deliver, and bank the genset up to whatever is left.

        E = Σ min(max(pv_fc − load_fc, 0), P_charge_max)   over the next `lookahead` h
        ψ = clip(1 − E·η_c / usable_capacity, psi_min, psi_max)

    Decided once per day at `decide_hour` and held, which is what "each night"
    means operationally — one decision, before the evening block starts.
    """
    pv_fc = np.asarray(pv_fc, float)
    load_fc = np.asarray(load_fc, float)
    hours = np.asarray(hours, int)
    n = len(pv_fc)
    usable = SOC_MAX - SOC_MIN

    psi = np.full(n, psi_max)
    current = psi_max
    for t in range(n):
        if hours[t] == decide_hour:
            hi = min(t + 1 + lookahead, n)
            surplus = np.minimum(
                np.maximum(pv_fc[t + 1:hi] - load_fc[t + 1:hi], 0.0), MAX_CHARGE_KW)
            headroom_frac = min(surplus.sum() * CHARGE_EFF / usable, 1.0)
            current = float(np.clip(1.0 - headroom_frac, psi_min, psi_max))
        psi[t] = current
    return psi


def sweep_threshold(pv, load, *, soc_init, thresholds=(0.50, 0.60, 0.70, 0.80),
                    bank_ceilings=(1.0,), cost=CORRECTED):
    """Report every (threshold, ceiling) pair and identify the best."""
    import pandas as pd
    rows = []
    for th in thresholds:
        for bank in bank_ceilings:
            res, run = commitment_heuristic(pv, load, soc_init=soc_init,
                                            threshold=th, bank_to_soc=bank,
                                            genset=cost.genset)
            m = res.metrics()
            e = cost.evaluate(res)
            rows.append(dict(threshold_pct=th * 100, bank_to_soc_pct=bank * 100,
                             **m, **{k: v for k, v in e.items()
                                     if k != "cost_model"}))
    df = pd.DataFrame(rows)
    df["is_best"] = df.total_cost_GHS == df.total_cost_GHS.min()
    return df
