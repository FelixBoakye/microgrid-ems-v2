# microgrid-ems-v2

Code and data for: *Within-Day Commitment Timing, Not Renewable Anticipation, Explains the Value
of Forecast-Driven Dispatch in Solar-Battery-Diesel Mini-Grids*, F. S. Boakye, submitted
(Energy for Sustainable Development).

## Summary

This repository models an islanded solar-battery-diesel mini-grid serving about 175
customers in the Northern Region of Ghana and compares energy management strategies for it.
XGBoost forecasts of load and PV feed a rolling mixed-integer linear programme (MILP) whose
generator model is commitment-aware (no-load fuel term, minimum loading, minimum up and down
times, start cost). The controller is benchmarked against a merit-order baseline, a reactive
heuristic, and a perfect-foresight bound, using an expanding-window walk-forward evaluation over
four test years. The analysis shows that the value of forecast-driven dispatch comes mainly from
within-day generator commitment timing rather than from anticipating renewable output.

## Key result

Walk-forward evaluation (expanding window, four test years 2022 to 2025; fold mean ± fold
standard deviation), rolling MILP versus the reactive heuristic under the same generator rules
(unit-commitment parity):

| Generator start cost | Saving vs reactive heuristic | Backing file |
|---|---|---|
| 33.75 GHS per start (headline) | **11.36% ± 1.78** | `results/v2/v42b_ucparity_2026-10-23/walkforward/walkforward_summary.csv` |
| 50.625 GHS per start | **10.25% ± 1.60** | `results/v2/v42b_ucparity_2026-10-23/walkforward_cs50p625/walkforward_summary.csv` |

At 33.75 GHS the same runs give **15.61% ± 0.41** versus the merit-order baseline. Loss of
power supply probability (LPSP) is **0.00%** in every fold. Per-fold values are in
`walkforward_folds.csv` next to each summary.

## System

| Item | Value |
|---|---|
| Location | 9.4008 N, 0.8393 W, Northern Region, Ghana |
| Customers | about 175 |
| Operation | Islanded mini-grid |
| PV array | 90 kWp |
| Battery | 130 kWh usable (90% depth of discharge, 92% round-trip efficiency) |
| Diesel generator | 40 kW |
| Annual demand | 119,926 kWh |
| Mean / peak demand | 13.69 kW / 49.24 kW |
| Weather data | NASA POWER hourly, 2018 to 2025 (70,128 records) |
| PV specific yield | 1,486 kWh/kWp/yr |

## Data sources and notes

- **Demand is synthetic (`load_v2`).** The site has no metered load data. The profile (AR(1)
  demand with weekday and temperature structure, 119,926 kWh/yr, seed 42) is anchored to Ghanaian
  and sub-Saharan African statistics cited in the paper. This is a stated limitation.
- **Weather:** NASA POWER hourly reanalysis, 2018 to 2025, included as
  `POWER_Point_Hourly_20180101_20251231_009d40N_000d84W_LST.csv` (2.8 MB). SHA-256 and file
  dates are in `manifests/data_provenance.json`. See `LICENSE-DATA.md` for the acknowledgement.
- **Fuel curves** are fitted from manufacturer specification sheets (Cummins C40D6, Generac
  SD040, Kohler 40REOZK). The sheets are cited in the paper, not redistributed.
- **Generator start cost:** 33.75 GHS per start (USD 3 at 11.25 GHS/USD), with 50.625 GHS
  (USD 4.5) as the upper case; literature range from Liu et al. 2017 (doi:10.1049/iet-gtd.2016.1699)
  and Lee and Chun 2018 (doi:10.3390/en11092468). Battery installed cost and cycle life are
  working assumptions examined by sensitivity analysis in the paper.

## What is included

| Path | Contents |
|---|---|
| `scripts_v2/` | Model, forecasting, MILP / LP dispatch, heuristic, walk-forward, bootstrap, figures |
| `tools/run_v42b_ucparity.py` | Runner that produced the paper results (unit-commitment parity) |
| `tools/run_v42b_extras.py` | Flat-rule tie check, re-plan timing, paper figures |
| `reproduce.py`, `REPRODUCE.md` | Staged pipeline from the raw weather file |
| `tests/` | Test suite (pytest) |
| `data/` | PV output, synthetic demand (`synthetic_load.csv`, `synthetic_load_lv2.csv`), forecasts, baseline simulation |
| `results/v2/v42b_ucparity_2026-10-23/` | **Paper results**: walk-forward (both start costs), bootstrap, ladders, sensitivity, week trace |
| `results/v2/v42b_extras_2026-10-23/` | Flat tie, timing, paper figures (`figures/fig1b`, `fig2` to `fig6`) |
| `results/v2/v42_rerun_2026-10-18c/` | Two input files checked by the runner (`SECTION4.md`, nominal ladder) |
| `manifests/`, `visuals/` | Provenance manifests; interactive dispatch viewer |

Only the two dated folders above back the paper's numbers. Earlier intermediate result
folders are not included. Large intermediate ML datasets (over 50 MB each) are not included;
`python reproduce.py --stage data` regenerates them.

## Reproduction

Python with the pinned packages in `requirements.lock` (35 packages). The MILP is solved with
HiGHS through `scipy.optimize.milp` (SciPy 1.18.0); no commercial solver or licence is needed.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock
python reproduce.py --list          # stages and runtimes
python reproduce.py --stage data    # PV, demand and features (~2 min)
python reproduce.py --verify        # checks and test suite (~2 min)
python tools/run_v42b_ucparity.py   # paper results (long run; resume-safe)
python tools/run_v42b_extras.py     # flat tie, timing, figures
python -m pytest tests
```

Runners write to a new dated folder and never overwrite an existing one. MILP solutions can
differ in the last digits between solver builds.

Reference hardware and wall-clock time for the paper run are recorded in the article and Supplementary material.

## Tables and figures

| Paper item | Backing file |
|---|---|
| Headline walk-forward result | `results/v2/v42b_ucparity_2026-10-23/walkforward/walkforward_summary.csv` |
| 50.625 GHS check | `results/v2/v42b_ucparity_2026-10-23/walkforward_cs50p625/walkforward_summary.csv` |
| Figures 1b, 2 to 6 | `results/v2/v42b_extras_2026-10-23/figures/` |
| Re-plan timing | `results/v2/v42b_extras_2026-10-23/timing.csv` |
| Other tables | see paper and Supplementary |

## License

- Code: MIT, see `LICENSE`.
- Data and generated results: CC BY 4.0, see `LICENSE-DATA.md`. NASA POWER and cited sources keep
  their own terms.

## Citation

See `CITATION.cff`. Software DOI: [10.5281/zenodo.23680302](https://doi.org/10.5281/zenodo.23680302)
(concept DOI [10.5281/zenodo.23678969](https://doi.org/10.5281/zenodo.23678969); cite the version DOI of the release you use).

## Previous repository

`https://github.com/FelixBoakye/microgrid_ems` is a retired earlier pipeline (genetic-algorithm
dispatch). Its results are not those of the paper.
