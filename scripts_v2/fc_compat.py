"""Direct-forecast trajectory for a year, on either code line.

The box replication (direct_cache.DirectFC) and the Build code
(direct_forecast.ensure_direct / trajectory_fn) expose the same direct per-lead
predictions through different APIs. Both store pred[k, row] = lead-k forecast
of full-series row ``row``, issued at row - k, and both assemble the plan for
decision hour i as [pred[1, a], pred[2, a+1], ...] with a = absolute row of i,
cut at the end of the given local index range. ``year_trajectory`` returns
``trajectory_at(i) -> (pv, load)`` for local hour i of ``year`` (optionally a
slice of it), whichever API is present.
"""
from __future__ import annotations

import numpy as np


def year_trajectory(year, local=None, seed=42):
    sl = slice(None) if local is None else local
    try:
        from . import direct_cache as DCa          # box replication API
    except ImportError:
        DCa = None
    if DCa is not None:
        from . import paths as P
        from . import walkforward as W
        d = W.full_series()
        dt = d.datetime[d.datetime.dt.year == year].values
        return DCa.for_year(P.variant(), year, datetimes=dt)[sl].trajectory
    from . import direct_forecast as D             # Build API
    frame = D.observed_frame()
    idx = D.year_index(frame, year)
    pv, ld = D.ensure_direct(frame, year, D.cache_path(year, seed), seed=seed)
    return D.trajectory_fn(pv, ld, np.asarray(idx)[sl])
