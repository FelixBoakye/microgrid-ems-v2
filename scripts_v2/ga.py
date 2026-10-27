"""Vectorised GA planner.

Two roles:

1. As a *fair benchmark*. With all fix flags off this reproduces the semantics
   of the original 09_ga_optimizer.py exactly — including its defects — but
   evaluates fitness across the population axis instead of looping over
   individuals in Python. The original is 5.08 s per 24 h window; this is
   ~0.157 s. Keeping it fast means the paper's GA-vs-LP comparison is an
   optimality-gap result rather than a straw man.

2. As the vehicle for the attribution ladder. Each defect is a separate flag,
   so rows 3-6 can be produced by turning exactly one fix on at a time.

Flags
-----
clip_diesel          fix D2: never plan more diesel than the residual gap
charge_from_surplus  fix D1: charging can only draw on real surplus
terminal_soc_value   value energy left in the battery at the horizon edge,
                     in GHS/kWh (0.0 reproduces the original: stored energy
                     is free)

The RNG draw *order* differs from the original (batched vs per-child), so
results are statistically equivalent rather than bit-identical. Phase 3
acceptance checks agreement within 5% across random windows.
"""
from __future__ import annotations

import numpy as np

from .config import (CHARGE_EFF, DIESEL_MARGINAL_COST, DIESEL_MAX_KW,
                     DISCHARGE_EFF, HORIZON, MAX_CHARGE_KW, MAX_DISCHARGE_KW,
                     SOC_MAX, SOC_MIN, W_CURTAIL, W_CYCLE, W_DIESEL, W_UNMET)

# ---- GA HYPERPARAMETERS (verbatim from the original) ----
POP_SIZE = 100
N_GENERATIONS = 150
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ELITE_FRACTION = 0.1
GREEDY_SEED_FRACTION = 0.70
MUTATION_SIGMA = 0.1
SEED_NOISE_SIGMA = 0.05


def greedy_heuristic(pv, load, soc_init, horizon=HORIZON):
    """Merit order PV -> battery -> diesel. Identical to the original."""
    charge = np.zeros(horizon)
    discharge = np.zeros(horizon)
    diesel = np.zeros(horizon)
    soc = float(soc_init)
    for t in range(horizon):
        net = pv[t] - load[t]
        if net >= 0:
            c = min(net, MAX_CHARGE_KW, (SOC_MAX - soc) / CHARGE_EFF)
            c = max(c, 0.0)
            charge[t] = c
            soc += c * CHARGE_EFF
        else:
            deficit = -net
            d = min(deficit, MAX_DISCHARGE_KW,
                    max((soc - SOC_MIN) * DISCHARGE_EFF, 0.0))
            discharge[t] = d
            soc -= d / DISCHARGE_EFF
            remaining = deficit - d
            if remaining > 0:
                diesel[t] = min(remaining, DIESEL_MAX_KW)
    return charge, discharge, diesel


def fitness_pop(pop, pv, load, soc_init, *, clip_diesel=False,
                charge_from_surplus=False, terminal_soc_value=0.0,
                horizon=HORIZON):
    """Vectorised fitness across the population axis. Lower is better."""
    P = pop.shape[0]
    ch = (pop[:, :horizon] * MAX_CHARGE_KW).copy()
    dc = (pop[:, horizon:2 * horizon] * MAX_DISCHARGE_KW).copy()
    di = (pop[:, 2 * horizon:] * DIESEL_MAX_KW).copy()
    net = pv - load

    # ---- repair operator ----
    for t in range(horizon):
        if net[t] >= 0:
            di[:, t] = 0.0
        else:
            gap = np.maximum(-net[t] - dc[:, t], 0.0)
            covered = gap <= 0
            di[covered, t] = np.maximum(di[covered, t] - dc[covered, t], 0.0)
            if clip_diesel:
                # FIX D2: diesel[t] = min(diesel[t], max(-net - discharge[t], 0))
                di[:, t] = np.minimum(di[:, t], gap)

    # ---- dispatch simulation ----
    soc = np.full(P, float(soc_init))
    unmet = np.zeros(P)
    dcost = np.zeros(P)
    curt = np.zeros(P)
    tch = np.zeros(P)
    tdc = np.zeros(P)

    for t in range(horizon):
        c = np.clip(ch[:, t], 0, MAX_CHARGE_KW)
        d = np.clip(dc[:, t], 0, MAX_DISCHARGE_KW)
        g = np.clip(di[:, t], 0, DIESEL_MAX_KW)
        if charge_from_surplus:
            # FIX D1: charging can only draw on surplus that physically exists
            c = np.minimum(c, max(net[t], 0.0))
        c = np.minimum(c, np.maximum((SOC_MAX - soc) / CHARGE_EFF, 0.0))
        d = np.minimum(d, np.maximum((soc - SOC_MIN) * DISCHARGE_EFF, 0.0))
        soc = np.clip(soc + c * CHARGE_EFF - d / DISCHARGE_EFF, SOC_MIN, SOC_MAX)

        net_supply = pv[t] + d + g - load[t]
        unmet += np.where(net_supply < 0, -net_supply, 0.0)
        curt += np.where(net_supply >= 0, np.maximum(net_supply - c, 0.0), 0.0)
        dcost += g * DIESEL_MARGINAL_COST
        tch += c
        tdc += d

    thru = (tch + tdc) / 2.0
    fit = (W_DIESEL * dcost + W_UNMET * unmet + W_CYCLE * thru + W_CURTAIL * curt)
    if terminal_soc_value:
        fit = fit - terminal_soc_value * (soc - float(soc_init)) * DISCHARGE_EFF
    return fit, ch, dc, di


def optimize(pv, load, soc_init, seed=42, *, horizon=HORIZON, **kw):
    """Run the GA for one window. Returns (charge, discharge, diesel, fitness)."""
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    rng = np.random.default_rng(seed)
    G = horizon * 3

    pop = rng.uniform(0, 1, (POP_SIZE, G))
    gc, gd, gg = greedy_heuristic(pv, load, soc_init, horizon)
    seed_chrom = np.concatenate([gc / MAX_CHARGE_KW,
                                 gd / MAX_DISCHARGE_KW,
                                 gg / DIESEL_MAX_KW])
    n_greedy = int(POP_SIZE * GREEDY_SEED_FRACTION)
    pop[:n_greedy] = np.clip(
        seed_chrom + rng.normal(0, SEED_NOISE_SIGMA, (n_greedy, G)), 0, 1)
    pop[0] = np.clip(seed_chrom, 0, 1)

    n_elite = max(1, int(POP_SIZE * ELITE_FRACTION))
    n_child = POP_SIZE - n_elite

    for _ in range(N_GENERATIONS):
        fit, *_ = fitness_pop(pop, pv, load, soc_init, horizon=horizon, **kw)
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]

        t1 = rng.integers(0, POP_SIZE, (n_child, 3))
        t2 = rng.integers(0, POP_SIZE, (n_child, 3))
        p1 = pop[t1[np.arange(n_child), np.argmin(fit[t1], axis=1)]]
        p2 = pop[t2[np.arange(n_child), np.argmin(fit[t2], axis=1)]]

        do_cx = rng.random(n_child) < CROSSOVER_RATE
        pt = rng.integers(1, G, n_child)
        left = np.arange(G)[None, :] < pt[:, None]
        kids = np.where(left | ~do_cx[:, None], p1, p2)

        mut = rng.random((n_child, G)) < MUTATION_RATE
        kids = np.clip(kids + mut * rng.normal(0, MUTATION_SIGMA, (n_child, G)), 0, 1)
        # Pre-allocated concatenation: the original rebuilt the array per child.
        pop = np.vstack([pop[:n_elite], kids])

    fit, ch, dc, di = fitness_pop(pop, pv, load, soc_init, horizon=horizon, **kw)
    b = int(np.argmin(fit))
    return ch[b], dc[b], di[b], float(fit[b])
