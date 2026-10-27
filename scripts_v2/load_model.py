"""Demand model — Phase 6, Step 1.

`load_v1` reproduces the delivered 02_load_profile_generator.py bit-for-bit and
stays selectable, because every published result was produced with it.

`load_v2` fixes the circularity in the original. Its residuals are i.i.d.
multiplicative noise on a deterministic diurnal shape, so the forecastable
component is exactly that shape and the residual is irreducible *by
construction*. For a study whose subject is the operational value of
forecasting, that assumes the answer: forecast error cannot be improved because
nothing in the residual is predictable.

`load_v2` changes three things and keeps everything else:

1. **Autocorrelated residuals.** AR(1) on the log-residual, so that recent load
   carries information about the next hour — which is what makes a forecaster's
   job non-trivial and what real metered demand exhibits.
2. **Weekday/weekend structure** in the productive-user component. Mills, shops
   and welders do not keep Sunday hours.
3. **Temperature dependence** of the household component, driven by the NASA
   POWER T2M series already in the pipeline: fans and cooling above a comfort
   threshold.

Archetypes, diurnal shapes, seasonal scaling and the spike process are retained
unchanged, and the annual energy is calibrated to within 1% of `load_v1` so the
sizing context is unchanged and the comparison is like-for-like.

All three added parameters are ASSUMPTIONS. None is grounded in metered data for
this site, and each is swept in `results/load_validation.md`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---- community composition (unchanged from the original) ----
N_HOUSEHOLDS_BASIC = 149
N_PRODUCTIVE_USE = 26
ANNUAL_KWH_PER_BASIC_HH = 500
ANNUAL_KWH_PER_PRODUCTIVE = 1700
RANDOM_SEED = 42

SPIKE_PROB = 0.03
SPIKE_LO, SPIKE_HI = 1.5, 3.0
NOISE_BASIC_SD = 0.08
NOISE_PRODUCTIVE_SD = 0.12


def basic_household_hourly_shape():
    shape = np.zeros(24)
    shape[0:6] = 0.15
    shape[6:9] = [0.6, 0.9, 0.7]
    shape[9:18] = 0.25
    shape[18:23] = [0.8, 1.0, 1.0, 0.9, 0.6]
    shape[23] = 0.25
    return shape / shape.sum()


def productive_use_hourly_shape():
    shape = np.zeros(24)
    shape[0:6] = 0.05
    shape[6:8] = [0.4, 0.8]
    shape[8:18] = 1.0
    shape[18:21] = [0.7, 0.4, 0.2]
    shape[21:24] = 0.05
    return shape / shape.sum()


SEASONAL = {1: 1.05, 2: 1.05, 3: 1.02, 4: 1.00, 5: 0.98, 6: 0.95,
            7: 0.93, 8: 0.93, 9: 0.96, 10: 1.00, 11: 1.03, 12: 1.06}


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadV2Params:
    """Every field is an ASSUMPTION. None is metered for this site."""

    #: AR(1) coefficient on the log-residual. 0.7 gives a lag-1 residual
    #: autocorrelation of 0.7 and an e-folding memory of ~2.8 h, which is the
    #: order of persistence reported for hourly household demand. Swept.
    phi: float = 0.70

    #: Innovation sd of the log-residual. Chosen so the marginal sd of the
    #: multiplicative residual matches load_v1's, keeping total variability
    #: comparable and isolating the *structure* of the residual as the change.
    sigma_target_basic: float = NOISE_BASIC_SD
    sigma_target_productive: float = NOISE_PRODUCTIVE_SD

    #: Productive-use activity by day of week (Mon=0). Rural mills, welders and
    #: shops run a full week bar Sunday, with a shorter Saturday.
    dow_productive: tuple = (1.00, 1.00, 1.00, 1.00, 1.00, 0.85, 0.45)

    #: Household cooling response: multiplier 1 + beta*(T2M - T_ref)+ per degC
    #: above a comfort threshold. Fans, not compressors, at this consumption level.
    beta_temp: float = 0.020
    t_ref_c: float = 26.0

    #: Calibrate annual energy back onto load_v1 so sizing context is unchanged.
    calibrate_to_v1: bool = True


# --------------------------------------------------------------------------
def generate_v1(datetimes, seed=RANDOM_SEED):
    """Bit-for-bit reproduction of the delivered 02_load_profile_generator.py.

    The legacy `np.random.seed` / `np.random.*` calls and their exact draw order
    are preserved deliberately; changing them would change the published series.
    """
    dt = pd.DatetimeIndex(datetimes)
    n = len(dt)
    np.random.seed(seed)

    daily_basic = ANNUAL_KWH_PER_BASIC_HH * N_HOUSEHOLDS_BASIC / 365.0
    daily_prod = ANNUAL_KWH_PER_PRODUCTIVE * N_PRODUCTIVE_USE / 365.0
    b_shape, p_shape = basic_household_hourly_shape(), productive_use_hourly_shape()

    season = np.array([SEASONAL[m] for m in dt.month])
    basic = daily_basic * b_shape[dt.hour] * season
    prod = daily_prod * p_shape[dt.hour] * season

    basic = basic * np.random.normal(1.0, NOISE_BASIC_SD, n).clip(0.7, 1.3)
    prod = prod * np.random.normal(1.0, NOISE_PRODUCTIVE_SD, n).clip(0.6, 1.4)

    spike_mask = np.random.rand(n) < SPIKE_PROB
    spike_mag = np.random.uniform(SPIKE_LO, SPIKE_HI, n)
    prod = prod.copy()
    prod[spike_mask] *= spike_mag[spike_mask]

    return pd.DataFrame({"datetime": dt, "basic_load_kW": basic,
                         "productive_load_kW": prod,
                         "total_load_kW": basic + prod})


def _ar1(n, phi, sigma_marginal, rng):
    """AR(1) log-residual with a specified *marginal* sd, mean-preserving.

    r_t = phi r_{t-1} + eps_t,  eps ~ N(0, s_eta),  s_eta = s_marg*sqrt(1-phi^2)
    The multiplier exp(r - s_marg^2/2) has unit expectation, so switching the
    residual structure does not change annual energy on its own.
    """
    s_marg = np.log1p(sigma_marginal ** 2) ** 0.5     # lognormal matching
    s_eta = s_marg * np.sqrt(max(1.0 - phi ** 2, 1e-12))
    r = np.empty(n)
    r[0] = rng.normal(0.0, s_marg)                     # start from stationarity
    eps = rng.normal(0.0, s_eta, n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return np.exp(r - 0.5 * s_marg ** 2)


def generate_v2(datetimes, t2m, params: LoadV2Params = None, seed=RANDOM_SEED,
                v1_annual_kwh=None, n_basic=None, n_productive=None):
    """Autocorrelated residuals, weekday/weekend productive use, temperature."""
    p = params or LoadV2Params()
    dt = pd.DatetimeIndex(datetimes)
    n = len(dt)
    t2m = np.asarray(t2m, float)
    rng = np.random.default_rng(seed)

    nb = N_HOUSEHOLDS_BASIC if n_basic is None else n_basic
    npu = N_PRODUCTIVE_USE if n_productive is None else n_productive
    daily_basic = ANNUAL_KWH_PER_BASIC_HH * nb / 365.0
    daily_prod = ANNUAL_KWH_PER_PRODUCTIVE * npu / 365.0
    b_shape, p_shape = basic_household_hourly_shape(), productive_use_hourly_shape()

    season = np.array([SEASONAL[m] for m in dt.month])
    basic = daily_basic * b_shape[dt.hour] * season
    prod = daily_prod * p_shape[dt.hour] * season

    # (2) weekday/weekend structure, productive component only
    dow = np.asarray(p.dow_productive, float)[dt.dayofweek]
    prod = prod * dow

    # (3) temperature response, household component only
    cooling = 1.0 + p.beta_temp * np.maximum(t2m - p.t_ref_c, 0.0)
    basic = basic * cooling

    # (1) autocorrelated residuals in place of i.i.d. noise
    basic = basic * _ar1(n, p.phi, p.sigma_target_basic, rng)
    prod = prod * _ar1(n, p.phi, p.sigma_target_productive, rng)

    # spike process retained
    spike_mask = rng.random(n) < SPIKE_PROB
    spike_mag = rng.uniform(SPIKE_LO, SPIKE_HI, n)
    prod = prod.copy()
    prod[spike_mask] *= spike_mag[spike_mask]

    total = basic + prod
    scale = 1.0
    if p.calibrate_to_v1 and v1_annual_kwh:
        years = n / 8760.0
        scale = v1_annual_kwh / (total.sum() / years)
        basic, prod, total = basic * scale, prod * scale, total * scale

    out = pd.DataFrame({"datetime": dt, "basic_load_kW": basic,
                        "productive_load_kW": prod, "total_load_kW": total})
    out.attrs["calibration_scale"] = scale
    return out


# --------------------------------------------------------------------------
def deterministic_shape(datetimes, t2m=None, params: LoadV2Params = None,
                        version="v2"):
    """The mean profile with no residual — the part a perfect forecaster of the
    deterministic component would get right. Used to extract residuals."""
    p = params or LoadV2Params()
    dt = pd.DatetimeIndex(datetimes)
    daily_basic = ANNUAL_KWH_PER_BASIC_HH * N_HOUSEHOLDS_BASIC / 365.0
    daily_prod = ANNUAL_KWH_PER_PRODUCTIVE * N_PRODUCTIVE_USE / 365.0
    season = np.array([SEASONAL[m] for m in dt.month])
    basic = daily_basic * basic_household_hourly_shape()[dt.hour] * season
    prod = daily_prod * productive_use_hourly_shape()[dt.hour] * season
    if version == "v2":
        prod = prod * np.asarray(p.dow_productive, float)[dt.dayofweek]
        if t2m is not None:
            basic = basic * (1.0 + p.beta_temp
                             * np.maximum(np.asarray(t2m, float) - p.t_ref_c, 0.0))
    # expected spike inflation on the productive component
    prod = prod * (1.0 + SPIKE_PROB * ((SPIKE_LO + SPIKE_HI) / 2.0 - 1.0))
    return basic + prod
