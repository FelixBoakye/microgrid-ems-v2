"""Phase 3 acceptance — LP planner, perfect-foresight oracle, vectorised GA."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts_v2 import dispatch_core as dc
from scripts_v2 import ga as ga2
from scripts_v2 import lp_dispatch, oracle
from scripts_v2.config import (DIESEL_MARGINAL_COST, HORIZON, SOC_INIT,
                               SOC_MAX, SOC_MIN, W_CYCLE)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def test_year():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(DATA / "synthetic_load.csv", parse_dates=["datetime"])
    df = pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime").sort_values("datetime")
    df = df[df.datetime.dt.year == 2025].reset_index(drop=True)
    return df.PV_power_kW.values, df.total_load_kW.values


# ------------------------------------------------------------------ LP
def test_lp_equals_greedy_where_greedy_is_provably_optimal():
    """A pure-deficit horizon with no PV and no storable surplus: merit order
    (battery to the floor, then genset) is provably optimal because there is
    nothing to trade off across time. LP must match it to 1e-6."""
    pv = np.zeros(HORIZON)
    load = np.full(HORIZON, 20.0)
    soc0 = 100.0

    sol = lp_dispatch.solve(pv, load, soc0)
    greedy = dc.baseline(pv, load, soc_init=soc0)

    assert sol.diesel.sum() == pytest.approx(greedy.diesel.sum(), abs=1e-6)
    assert sol.discharge.sum() == pytest.approx(greedy.discharge.sum(), abs=1e-6)
    assert sol.unmet.sum() == pytest.approx(greedy.unmet.sum(), abs=1e-6)

    lp_cost = (sol.diesel.sum() * DIESEL_MARGINAL_COST
               + W_CYCLE * (sol.charge.sum() + sol.discharge.sum()) / 2.0)
    assert lp_cost == pytest.approx(dc.objective(greedy), abs=1e-6)


def test_lp_terminal_credit_does_not_hoard_through_evening_deficit():
    """Daytime PV surplus, then a 12 h evening deficit, terminal SOC valued at
    diesel marginal. The planner must discharge through the evening rather than
    sit at SOC_MAX and cover the night with the genset.

    Crediting (c_f - W_CYCLE/2)*eta_d is a tied face; HiGHS picked the
    hoarding vertex. Crediting c_f*eta_d (cycle cost only on c,d) strictly
    prefers hoarding. The live credit subtracts the full W_CYCLE coefficient
    so use is strictly cheaper than hoard.
    """
    pv = np.array([40.0] * 12 + [0.0] * 12)
    load = np.array([10.0] * 12 + [20.0] * 12)
    sol = lp_dispatch.solve(pv, load, 65.0,
                            terminal_soc_value=DIESEL_MARGINAL_COST)

    assert sol.soc[-1] == pytest.approx(SOC_MIN, abs=1e-4)
    assert sol.soc[-1] < SOC_MAX - 1.0
    assert sol.discharge[12:].sum() > 1.0
    # Evening load is 20 kW * 12 h = 240 kWh. Serving it entirely from the
    # genset while holding a full pack is the degenerate vertex.
    assert sol.diesel[12:].sum() < 240.0 - 1.0


def test_lp_terminal_credit_discharges_a_pure_deficit_to_the_floor():
    """No PV, constant load, pack starts above the floor. With terminal SOC
    valued at diesel marginal, leftover energy must not be hoarded: the
    planner discharges to SOC_MIN and the genset covers only the residual.
    """
    pv = np.zeros(HORIZON)
    load = np.full(HORIZON, 20.0)
    soc0 = 100.0
    sol = lp_dispatch.solve(pv, load, soc0,
                            terminal_soc_value=DIESEL_MARGINAL_COST)
    greedy = dc.baseline(pv, load, soc_init=soc0)

    assert sol.soc[-1] == pytest.approx(SOC_MIN, abs=1e-4)
    assert sol.soc[-1] == pytest.approx(greedy.soc[-1], abs=1e-4)
    assert sol.diesel.sum() == pytest.approx(greedy.diesel.sum(), abs=1e-4)
    assert sol.discharge.sum() == pytest.approx(greedy.discharge.sum(), abs=1e-4)


def test_lp_solution_is_physically_feasible(test_year):
    """Replaying the LP's battery plan through `step` must reproduce it, which
    proves the LP is not exploiting a modelling gap the simulator would reject."""
    pv, load = test_year
    pv, load = pv[:168], load[:168]
    sol = lp_dispatch.solve(pv, load, 65.0)
    res = dc.simulate(pv, load, soc_init=65.0, c_cmd=sol.charge,
                      d_cmd=sol.discharge, g_cmd=sol.diesel,
                      rt_balance=False, battery_mode="setpoint")
    assert res.diesel.sum() == pytest.approx(sol.diesel.sum(), rel=1e-6)
    assert res.unmet.sum() == pytest.approx(sol.unmet.sum(), abs=1e-6)


def test_lp_window_is_fast(test_year):
    pv, load = test_year
    t0 = time.time()
    for i in range(20):
        lp_dispatch.solve(pv[i * 24:(i + 1) * 24], load[i * 24:(i + 1) * 24], 65.0)
    ms = (time.time() - t0) / 20 * 1000
    print(f"\n  LP 24 h window: {ms:.2f} ms")
    assert ms < 200.0, "LP window solve should be milliseconds, not seconds"


# -------------------------------------------------------------- ORACLE
def test_oracle_solves_full_year_quickly(test_year):
    pv, load = test_year
    t0 = time.time()
    sol, res = oracle.solve_year(pv, load, soc_init=SOC_MIN)
    dt = time.time() - t0
    print(f"\n  oracle 8760 h: {dt:.2f} s")
    assert len(res.load) == 8760
    assert dt < 30.0


def test_oracle_exactly_equals_greedy_baseline(test_year):
    """THE central gate. Under the flat-fuel cost model as written, perfect
    foresight over the whole year buys nothing: the greedy rule is already
    globally optimal.

    Note the totals are bit-identical while the hour-by-hour allocation is not.
    The objective has no time preference, so the LP lands on a different vertex
    of the same optimal face. Cost equality is the claim that matters.
    """
    pv, load = test_year
    base = dc.baseline(pv, load, soc_init=SOC_MIN, uc_parity=False)  # flat-model tie
    _, ores = oracle.solve_year(pv, load, soc_init=SOC_MIN)

    assert dc.objective(ores) == pytest.approx(dc.objective(base), abs=1e-6)
    assert ores.diesel.sum() == pytest.approx(base.diesel.sum(), abs=1e-6)
    assert ores.curtail.sum() == pytest.approx(base.curtail.sum(), abs=1e-6)
    assert ores.discharge.sum() == pytest.approx(base.discharge.sum(), abs=1e-6)
    assert ores.unmet.sum() == pytest.approx(0.0, abs=1e-9)


def test_oracle_is_a_lower_bound_on_every_controller(test_year):
    """If any controller beats the oracle, information is leaking."""
    pv, load = test_year
    _, ores = oracle.solve_year(pv, load, soc_init=SOC_MIN)

    controllers = {
        "greedy baseline": dc.baseline(pv, load, soc_init=SOC_MIN),
        "idle battery": dc.simulate(pv, load, soc_init=SOC_MIN,
                                    d_cmd=np.zeros(len(pv)),
                                    rt_balance=True, battery_mode="setpoint"),
        "reserve floor 60 kWh": dc.simulate(pv, load, soc_init=SOC_MIN,
                                            rt_balance=True,
                                            battery_mode="merit", soc_floor=60.0),
    }
    for label, res in controllers.items():
        o, c = oracle.verify_lower_bound(ores, res, label=label)
        assert o <= c + 1e-6


# ------------------------------------------------------------------ GA
def test_vectorised_ga_fitness_is_semantically_identical(ga, test_year):
    """The strong form of the fairness claim, and it is deterministic: evaluated
    on the *same* population, the vectorised fitness must equal the original's
    chromosome-by-chromosome fitness. Any difference here would mean the
    benchmark had been changed, not just sped up."""
    pv, load = test_year
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(12):
        s = int(rng.integers(0, len(pv) - HORIZON))
        p, l = pv[s:s + HORIZON], load[s:s + HORIZON]
        soc0 = float(rng.uniform(20, 120))
        pop = rng.uniform(0, 1, (60, HORIZON * 3))

        f_vec, *_ = ga2.fitness_pop(pop, p, l, soc0)
        f_orig = np.array([ga.fitness(pop[i], p, l, soc0) for i in range(60)])
        worst = max(worst, float(np.abs(f_vec - f_orig).max()))

    print(f"\n  worst deviation over 720 chromosomes: {worst:.3e}")
    assert worst < 1e-9, "vectorised fitness is not the original fitness"


def test_vectorised_ga_matches_original_within_5pct(ga, test_year):
    """The brief's benchmark criterion, applied to the search as a whole.

    `optimize()` is stochastic and the vectorised form draws its random numbers
    in a different order (batched rather than per-child), so single windows
    scatter — on low-cost daylight windows where the objective is ~10 GHS a
    1.6 GHS difference is 16%. The quantity with a defined value is the mean
    over seeds, so that is what is compared. Per-window figures are printed.
    """
    pv, load = test_year
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(pv) - HORIZON, 10)
    seeds = range(5)

    tot_o = tot_v = 0.0
    t_orig = t_vec = 0.0
    for s in starts:
        p, l = pv[s:s + HORIZON], load[s:s + HORIZON]

        t0 = time.time()
        f_o = np.mean([ga.optimize(p, l, 65.0, seed=sd)[3] for sd in seeds])
        t_orig += time.time() - t0

        t0 = time.time()
        f_v = np.mean([ga2.optimize(p, l, 65.0, seed=sd)[3] for sd in seeds])
        t_vec += time.time() - t0

        tot_o += f_o
        tot_v += f_v
        print(f"  t={s:5d}  original {f_o:9.2f}  vectorised {f_v:9.2f}  "
              f"{(f_v / f_o - 1) * 100:+6.2f}%")

    gap = abs(tot_v / tot_o - 1.0)
    n = len(starts) * len(seeds)
    print(f"\n  aggregate gap {gap * 100:.2f}%  |  "
          f"original {t_orig / n:.3f} s/window, vectorised {t_vec / n:.3f} s/window "
          f"({t_orig / t_vec:.0f}x)")
    assert gap < 0.05, f"vectorised GA is {gap * 100:.1f}% off the original"


def test_ga_repair_fix_caps_diesel_at_residual():
    """With clip_diesel on, the D2 defect is gone from the v2 GA."""
    pv = np.zeros(HORIZON)
    load = np.full(HORIZON, 10.0)
    pop = np.zeros((1, HORIZON * 3))
    pop[0, HORIZON:2 * HORIZON] = 2.0 / 65.0     # discharge 2 kW
    pop[0, 2 * HORIZON:] = 1.0                   # diesel at full 40 kW

    _, _, _, di_broken = ga2.fitness_pop(pop, pv, load, 65.0, clip_diesel=False)
    _, _, _, di_fixed = ga2.fitness_pop(pop, pv, load, 65.0, clip_diesel=True)

    assert di_broken[0, 0] == pytest.approx(40.0)
    assert di_fixed[0, 0] == pytest.approx(8.0)   # the true residual gap


def test_ga_charge_from_surplus_fix_blocks_invented_energy():
    """With charge_from_surplus on, the D1 defect is gone from the v2 GA."""
    pv = np.zeros(HORIZON)
    load = np.zeros(HORIZON)
    pop = np.zeros((1, HORIZON * 3))
    pop[0, :HORIZON] = 1.0                       # charge 65 kW every hour

    f_broken, *_ = ga2.fitness_pop(pop, pv, load, 13.0, charge_from_surplus=False)
    f_fixed, *_ = ga2.fitness_pop(pop, pv, load, 13.0, charge_from_surplus=True)

    # Broken: 117 kWh enters the battery and is charged only as throughput.
    assert f_broken[0] > 0
    # Fixed: nothing can be charged, so throughput and cost are zero.
    assert f_fixed[0] == pytest.approx(0.0)
