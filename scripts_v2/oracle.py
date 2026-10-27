"""Perfect-foresight annual optimum.

The same LP as `lp_dispatch`, solved once over all 8,760 hours with ground-truth
PV and load. This is the global lower bound on dispatch cost under the cost
model in force, and it is the number every controller should be scored against
— not the rule-based baseline.

If any controller ever comes in below the oracle, information has leaked.
"""
from __future__ import annotations

import numpy as np

from . import lp_dispatch
from .config import DIESEL_MARGINAL_COST, SOC_INIT
from .dispatch_core import SimResult


def solve_year(pv, load, soc_init=SOC_INIT, *, terminal_soc_value=0.0, **kw):
    """Global optimum over the whole series. Returns (LPSolution, SimResult)."""
    sol = lp_dispatch.solve(pv, load, soc_init,
                            terminal_soc_value=terminal_soc_value, **kw)
    res = SimResult(soc=sol.soc, charge=sol.charge, discharge=sol.discharge,
                    diesel=sol.diesel, unmet=sol.unmet, curtail=sol.spill,
                    pv=np.asarray(pv, float), load=np.asarray(load, float))
    return sol, res


def verify_lower_bound(oracle_res: SimResult, other: SimResult, *,
                       label="controller", tol=1e-6):
    """Assert the oracle is a genuine lower bound on this controller's cost."""
    from .dispatch_core import objective
    o = objective(oracle_res)
    c = objective(other)
    assert o <= c + tol, (
        f"{label} cost {c:.6f} beats the perfect-foresight oracle {o:.6f} — "
        "information is leaking; stop and find it")
    return o, c
