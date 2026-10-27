"""The corrected cost model — Phase 5.

The flat 0.32 L/kWh in the published model is the genset's *full-load* fuel
figure. Real gensets follow an affine curve (HOMER's standard form):

    F(P) = A * P_rated + B * P          [L/h]

with a no-load term A that is burned whenever the unit is running at all. For
this 40 kW unit that is 3.2 L/h at idle and 13.04 L/h at rated output, giving
0.326 L/kWh at full load — which is where the published 0.32 came from. At the
baseline's measured 14% mean load factor the true figure is 0.82 L/kWh, 2.5x
worse.

Under this model the greedy rule stops being optimal, because the cheapest way
to serve a night deficit is to start the genset once, run it near rated output,
serve the load and bank the surplus in the battery, then shut down — instead of
idling a 40 kW unit for nine hours at 14% load. No reactive controller can do
that. It requires foresight.

Battery throughput is also priced properly. W_CYCLE = 0.05 GHS/kWh is understated
by roughly 20x: a 130 kWh LFP pack delivering ~4,000 cycles at 90% DoD gives
~468,000 kWh of lifetime throughput, so any installed cost near 468,000 GHS
implies of order 1 GHS/kWh. That does not overturn the merit order — the battery
at ~1 GHS/kWh still beats diesel at ~5.28 — so it does not by itself make
foresight valuable, but it materially changes the economics and the lifetime-cost
claims.

BATTERY_COST_PER_KWH_THROUGHPUT below is an ESTIMATE derived from the assumptions
recorded in `battery_throughput_cost()`, not a sourced figure. The installed cost
and warranty cycle life should be replaced with vendor numbers before publication;
`results/sizing_sweep.csv` reports sensitivity to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import (BATTERY_CAPACITY_KWH, DIESEL_MAX_KW, DIESEL_PRICE_GHS,
                     DOD_MAX, W_CURTAIL, W_UNMET)


def battery_throughput_cost(installed_cost_ghs_per_kwh=3600.0,
                            cycle_life=4000.0, dod=DOD_MAX,
                            capacity_kwh=BATTERY_CAPACITY_KWH):
    """GHS per kWh of throughput, amortised over warranty life.

    Default assumptions (all replaceable, none sourced):
      installed cost   3,600 GHS/kWh of nameplate  (~USD 240/kWh at 15 GHS/USD)
      cycle life       4,000 cycles at 90% DoD
    giving 130 * 4000 * 0.9 = 468,000 kWh lifetime throughput against a
    468,000 GHS pack, i.e. 1.00 GHS/kWh.
    """
    capital = installed_cost_ghs_per_kwh * capacity_kwh
    lifetime_throughput = capacity_kwh * cycle_life * dod
    return capital / lifetime_throughput


BATTERY_COST_PER_KWH_THROUGHPUT = battery_throughput_cost()   # 1.00 GHS/kWh


# ---------------------------------------------------------------- fuel curves
# Fitted by least squares to the published 25/50/75/100%-load fuel consumption
# points on three 40 kW-class sets. Provenance, exact quoted figures, per-model
# R² and residuals are in results/parameter_sources.md.
#
# "assumed_legacy" is the original unsourced HOMER-generic pair. It is retained
# so every pre-adoption result stays reproducible.
FUEL_CURVES = {
    #  name                      A       B      R²        source
    "assumed_legacy":        (0.0800, 0.2460, None,   "HOMER generic, UNSOURCED"),
    "cummins_c40d6":         (0.0449, 0.3006, 0.99988, "Cummins C40D6 data sheet"),
    "generac_sd040":         (0.0300, 0.3020, 0.99982, "Generac SD040 spec sheet"),
    "kohler_40reozk":        (0.0613, 0.2990, 0.97840, "Kohler 40REOZK G5437"),
    "sourced_mean":          (0.0454, 0.3005, None,   "mean of the three sourced sets"),
    "sourced_mean_ex_kohler": (0.03745, 0.3013, None,
                              "mean of Cummins and Generac; excludes the Kohler "
                              "set, whose published 50% point departs from the "
                              "affine trend the other two follow (R² 0.978)"),
}

ADOPTED_CURVE = "sourced_mean"


@dataclass(frozen=True)
class GensetModel:
    """Affine fuel curve plus unit-commitment constraints."""
    rated_kw: float = DIESEL_MAX_KW
    a_l_per_kw_rated_h: float = FUEL_CURVES[ADOPTED_CURVE][0]   # no-load
    b_l_per_kwh: float = FUEL_CURVES[ADOPTED_CURVE][1]          # marginal
    fuel_price_ghs_per_l: float = DIESEL_PRICE_GHS
    start_cost_ghs: float = 150.0         # wear, lube, starter battery
    min_load_ratio: float = 0.30
    min_up_h: int = 2
    min_down_h: int = 2

    def __post_init__(self):
        # Catches a GensetModel() built anywhere at the 150 GHS dataclass default.
        _strict_check(self.start_cost_ghs)

    @property
    def no_load_l_per_h(self):
        return self.a_l_per_kw_rated_h * self.rated_kw          # 3.2 L/h

    @property
    def full_load_l_per_h(self):
        return self.no_load_l_per_h + self.b_l_per_kwh * self.rated_kw   # 13.04

    @property
    def no_load_cost_ghs_per_h(self):
        return self.no_load_l_per_h * self.fuel_price_ghs_per_l

    @property
    def marginal_cost_ghs_per_kwh(self):
        return self.b_l_per_kwh * self.fuel_price_ghs_per_l      # 4.06 GHS/kWh

    @property
    def min_load_kw(self):
        return self.min_load_ratio * self.rated_kw

    def litres(self, power_kw, running=None):
        """Fuel burn in L/h at output `power_kw`."""
        if running is None:
            running = power_kw > 1e-9
        return (self.no_load_l_per_h * float(bool(running))
                + self.b_l_per_kwh * power_kw)

    def specific_l_per_kwh(self, power_kw):
        """Realised L/kWh at this output. 0.326 at rated, 0.82 at 14% load."""
        if power_kw <= 1e-9:
            return float("nan")
        return self.litres(power_kw) / power_kw


@dataclass(frozen=True)
class CostModel:
    """Which cost model an experiment is run under."""
    name: str
    genset: GensetModel = field(default_factory=GensetModel)
    battery_ghs_per_kwh_throughput: float = BATTERY_COST_PER_KWH_THROUGHPUT
    w_unmet: float = W_UNMET
    w_curtail: float = W_CURTAIL
    commitment: bool = True

    def fuel_cost(self, diesel_kw, running_hours=None):
        """Annual fuel cost of a dispatch series."""
        import numpy as np
        d = np.asarray(diesel_kw, float)
        run = (d > 1e-9) if running_hours is None else np.asarray(running_hours, bool)
        litres = (self.genset.no_load_l_per_h * run.sum()
                  + self.genset.b_l_per_kwh * d.sum())
        return litres * self.genset.fuel_price_ghs_per_l, litres

    def evaluate(self, res):
        """Total cost of a SimResult under this cost model."""
        import numpy as np
        run = res.diesel > 1e-9
        cost, litres = self.fuel_cost(res.diesel, run)
        starts = int(np.sum(run[1:] & ~run[:-1]) + (1 if run[0] else 0))
        thru = (res.charge.sum() + res.discharge.sum()) / 2.0
        total = (cost
                 + starts * self.genset.start_cost_ghs
                 + thru * self.battery_ghs_per_kwh_throughput
                 + res.unmet.sum() * self.w_unmet
                 + res.curtail.sum() * self.w_curtail)
        lf = res.diesel[run] / self.genset.rated_kw if run.any() else np.array([0.0])
        return {
            "cost_model": self.name,
            "total_cost_GHS": float(total),
            "fuel_cost_GHS": float(cost),
            "fuel_litres": float(litres),
            "start_cost_GHS": float(starts * self.genset.start_cost_ghs),
            "battery_cost_GHS": float(thru * self.battery_ghs_per_kwh_throughput),
            "unmet_penalty_GHS": float(res.unmet.sum() * self.w_unmet),
            "genset_run_hours": int(run.sum()),
            "genset_starts": starts,
            "mean_run_hours_per_start": float(run.sum() / starts) if starts else 0.0,
            "mean_load_factor_pct": float(lf.mean() * 100),
            "hours_below_10pct_load_pct": float((lf < 0.10).mean() * 100),
            "specific_fuel_L_per_kWh": float(litres / res.diesel.sum())
            if res.diesel.sum() > 0 else float("nan"),
        }


#: Code default kept at 150 GHS so V4.0 results stay reproducible. A run sets
#: MG_START_COST_GHS (e.g. 33.75) and every cost model built without an explicit
#: start cost, including CORRECTED below, uses it. MG_START_COST_STRICT=1 makes
#: the 150 GHS default an error, so no stage can fall back to it silently.
LEGACY_START_COST_GHS = 150.0


def nominal_start_cost() -> float:
    import os
    raw = os.environ.get("MG_START_COST_GHS", "").strip()
    if not raw:
        if os.environ.get("MG_START_COST_STRICT", "").strip() == "1":
            raise RuntimeError("MG_START_COST_STRICT=1 but MG_START_COST_GHS is unset")
        return LEGACY_START_COST_GHS
    return float(raw)


def _strict_check(start_cost_ghs: float) -> None:
    import os
    if (os.environ.get("MG_START_COST_STRICT", "").strip() == "1"
            and abs(float(start_cost_ghs) - LEGACY_START_COST_GHS) < 1e-12):
        raise RuntimeError("a cost model was built at the 150 GHS code default "
                           "inside a strict start-cost run")


def corrected(curve=ADOPTED_CURVE, **kw) -> "CostModel":
    """A corrected cost model using a named fuel curve.

    The flat model is unaffected by the choice: it is the *published* cost model
    and its 0.32 L/kWh is a different (flat, full-load) parameterisation
    entirely. Changing the sourced curve therefore cannot move any flat-model
    result, and the three structural invariants are untouched by construction.
    """
    a, b, _r2, _src = FUEL_CURVES[curve]
    # Explicit keyword overrides win over the curve's own A and B, so a
    # one-at-a-time sensitivity sweep can vary either coefficient in isolation.
    gkw = {k: v for k, v in kw.items() if k in GensetModel.__dataclass_fields__}
    gkw.setdefault("a_l_per_kw_rated_h", a)
    gkw.setdefault("b_l_per_kwh", b)
    gkw.setdefault("start_cost_ghs", nominal_start_cost())
    _strict_check(gkw["start_cost_ghs"])
    g = GensetModel(**gkw)
    rest = {k: v for k, v in kw.items()
            if k in CostModel.__dataclass_fields__ and k != "genset"}
    return CostModel(name=f"corrected [{curve}]", genset=g, **rest)


FLAT = CostModel(name="flat (as published)",
                 genset=GensetModel(a_l_per_kw_rated_h=0.0, b_l_per_kwh=0.32,
                                    start_cost_ghs=0.0, min_load_ratio=0.0,
                                    min_up_h=1, min_down_h=1),
                 battery_ghs_per_kwh_throughput=0.05,
                 commitment=False)

#: The adopted model: sourced fuel curve, author-approved 2026-08-13.
CORRECTED = corrected(ADOPTED_CURVE)

#: The pre-adoption model, retained so earlier results stay reproducible.
CORRECTED_ASSUMED = corrected("assumed_legacy")
