"""Single source of truth for system parameters.

Values are taken verbatim from the original delivered scripts so that v2 results
are comparable to the published ones. Anything that differs from the original is
marked and defaults to the original behaviour.
"""
from __future__ import annotations

import numpy as np

# ---- BATTERY ----
BATTERY_CAPACITY_KWH = 130.0
DOD_MAX = 0.90
SOC_MIN = (1.0 - DOD_MAX) * BATTERY_CAPACITY_KWH   # 13 kWh
SOC_MAX = BATTERY_CAPACITY_KWH                     # 130 kWh
RTE = 0.92
CHARGE_EFF = float(np.sqrt(RTE))
DISCHARGE_EFF = float(np.sqrt(RTE))
MAX_CHARGE_KW = 65.0        # 0.5C
MAX_DISCHARGE_KW = 65.0

# ---- DIESEL ----
DIESEL_MAX_KW = 40.0
DIESEL_FUEL_RATE = 0.32                              # L/kWh (full-load figure)
DIESEL_PRICE_GHS = 16.5                              # GHS/L
DIESEL_MARGINAL_COST = DIESEL_FUEL_RATE * DIESEL_PRICE_GHS   # 5.28 GHS/kWh

# ---- PV ----
PV_RATED_KWP = 90.0

# ---- OBJECTIVE WEIGHTS (as published) ----
W_DIESEL = 1.0
W_UNMET = 5000.0
W_CYCLE = 0.05
W_CURTAIL = 0.02

# ---- HORIZONS ----
HORIZON = 24
REPLAN_INTERVAL = 6

# ---- SIMULATION ----
SOC_INIT = SOC_MAX * 0.5    # 65 kWh, as in the original scripts
TOL = 1e-9
