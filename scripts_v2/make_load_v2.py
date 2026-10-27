"""Generate the load_v2 demand series (and the 1.2%-mix variant).

Split out of an ad-hoc snippet so `reproduce.py` has a real stage to call and the
series is regenerable from raw inputs rather than from a shell one-liner.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import load_model as LM

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    orig = pd.read_csv(DATA / "synthetic_load.csv", parse_dates=["datetime"])
    v1a = orig.total_load_kW.sum() / (len(orig) / 8760)

    v2 = LM.generate_v2(pv.datetime, pv.T2M.values, v1_annual_kwh=v1a)
    v2.to_csv(DATA / "synthetic_load_lv2.csv", index=False)
    print(f"load_v2   {len(v2)} rows, "
          f"{v2.total_load_kW.sum() / (len(v2) / 8760):,.0f} kWh/yr "
          f"({(v2.total_load_kW.sum() / (len(v2) / 8760) / v1a - 1) * 100:+.4f}% vs load_v1)")

    # ESMAP customer mix: 5 commercial of 422 = 1.2% non-household
    mix = LM.generate_v2(pv.datetime, pv.T2M.values, v1_annual_kwh=v1a,
                         n_basic=173, n_productive=2)
    mix.to_csv(DATA / "synthetic_load_mix12.csv", index=False)
    print(f"load_mix12 {len(mix)} rows, "
          f"{mix.total_load_kW.sum() / (len(mix) / 8760):,.0f} kWh/yr, "
          f"2/175 = 1.1% non-household")


if __name__ == "__main__":
    main()
