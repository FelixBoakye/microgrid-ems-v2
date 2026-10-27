"""Cross-checks between stored result files.

The attribution ladder and the re-planning sweep run the same controller
configurations through the same code. They must therefore agree on the
configurations they share. They once did not — the stored sweep reported +4.88%
and +5.76% where the ladder reported +9.81% and +10.58% for the identical
6 h and 1 h setpoint controllers — because the sweep had been produced with
--quick, i.e. 672 hours of January rather than the full 8,760-hour test year.

Neither number was wrong for what it measured; the sweep was simply answering a
different question and had been written to the canonical filename. These tests
make that class of mistake fail loudly instead of sitting in a CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

LADDER = RESULTS / "attribution_ladder.csv"
SWEEP = RESULTS / "replan_sweep.csv"

TEST_YEAR_HOURS = 8760

pytestmark = pytest.mark.skipif(
    not (LADDER.exists() and SWEEP.exists()),
    reason="result files not generated yet")


@pytest.fixture(scope="module")
def ladder():
    return pd.read_csv(LADDER)


@pytest.fixture(scope="module")
def sweep():
    return pd.read_csv(SWEEP)


def test_canonical_results_cover_the_whole_test_year(ladder, sweep):
    """A canonical result file must never hold a --quick slice."""
    assert (ladder.hours == TEST_YEAR_HOURS).all(), \
        f"attribution_ladder.csv holds {sorted(ladder.hours.unique())} hours"
    assert (sweep.hours == TEST_YEAR_HOURS).all(), \
        f"replan_sweep.csv holds {sorted(sweep.hours.unique())} hours"


def test_ladder_and_sweep_agree_on_shared_configurations(ladder, sweep):
    """Same controller, same data, same code -> same number."""
    setpoint = sweep[sweep.configuration == "LP setpoint"].set_index("interval_h")
    hier = sweep[sweep.configuration == "LP hierarchical"].set_index("interval_h")

    shared = [
        ("7", setpoint, 6, "LP, 6 h block, setpoint execution"),
        ("8", setpoint, 1, "LP, 1 h receding, setpoint execution"),
        ("10", hier, 1, "LP, 1 h, hierarchical execution"),
    ]
    for row, table, interval, label in shared:
        L = ladder[ladder.row.astype(str) == row].iloc[0]
        S = table.loc[interval]
        assert L.diesel_kWh_per_year == pytest.approx(
            S.diesel_kWh_per_year, abs=1e-6), f"{label}: diesel disagrees"
        assert L.premium_vs_oracle_pct == pytest.approx(
            S.premium_vs_oracle_pct, abs=1e-9), f"{label}: premium disagrees"


def test_both_files_share_one_oracle_floor(ladder, sweep):
    L = ladder[ladder.row.astype(str) == "0"].iloc[0]
    S = sweep[sweep.interval_h == 0].iloc[0]
    assert L.diesel_kWh_per_year == pytest.approx(S.diesel_kWh_per_year, abs=1e-6)
    assert L.objective == pytest.approx(S.objective, abs=1e-6)


def test_no_controller_beats_the_oracle(ladder):
    """The oracle is a lower bound. Anything below it means a leak."""
    assert (ladder.premium_vs_oracle_pct >= -1e-6).all(), \
        ladder.loc[ladder.premium_vs_oracle_pct < -1e-6,
                   ["row", "configuration", "premium_vs_oracle_pct"]]


def test_hierarchical_execution_ties_the_oracle(ladder, sweep):
    """The load-bearing Phase 4 result: with the execution layer closing the
    loop on the battery, commitment length stops mattering."""
    hier = sweep[sweep.configuration == "LP hierarchical"]
    assert len(hier) >= 6
    assert (hier.premium_vs_oracle_pct.abs() < 1e-9).all(), \
        "hierarchical execution no longer ties the oracle at every interval"
    for row in ("10", "10p"):
        L = ladder[ladder.row.astype(str) == row].iloc[0]
        assert abs(L.premium_vs_oracle_pct) < 1e-9


# --------------------------------------------------------------- load variant
LOADV1_ORACLE_DIESEL_KWH = 21480.058593454844


def test_canonical_results_hold_load_v1_not_another_variant():
    """Canonical result files must describe the published load_v1 series.

    A `MG_LOAD=v2` run once wrote its replan sweep to results/replan_sweep.csv,
    silently replacing the load_v1 numbers (21,480.06 kWh) with load_v2 ones
    (23,034.65). Writes are now routed through paths.results_subdir, and this
    fingerprint fails loudly if anything overwrites a canonical file with a
    different demand series.
    """
    sweep = pd.read_csv(SWEEP)
    floor = sweep[sweep.interval_h == 0].iloc[0]
    assert floor.diesel_kWh_per_year == pytest.approx(
        LOADV1_ORACLE_DIESEL_KWH, abs=1e-6), (
        "results/replan_sweep.csv is not the load_v1 series")

    ladder = pd.read_csv(LADDER)
    orc = ladder[ladder.row.astype(str) == "0"].iloc[0]
    assert orc.diesel_kWh_per_year == pytest.approx(
        LOADV1_ORACLE_DIESEL_KWH, abs=1e-6), (
        "results/attribution_ladder.csv is not the load_v1 series")
