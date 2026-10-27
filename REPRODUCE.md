# Reproducibility package

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.lock
python reproduce.py --list          # stages and runtimes
python reproduce.py --stage data    # PV + demand + features, ~2 min
python reproduce.py --verify        # CI checks + full test suite, ~2 min
python reproduce.py --all           # everything from the raw CSV, ~6.4 h
```

## Paper results

The numbers reported in the paper come only from these two folders:

- `results/v2/v42b_ucparity_2026-10-23/` - walk-forward (both start costs), bootstrap, ladders, sensitivity, week trace
- `results/v2/v42b_extras_2026-10-23/` - flat tie, timing, paper figures

## What is in the package

| Item | Where |
|---|---|
| Pinned environment | `requirements.lock` (exact versions) |
| Raw input + provenance | `POWER_Point_Hourly_*.csv`, `manifests/data_provenance.json` (sha256, timestamps) |
| Single entry point | `reproduce.py` - raw CSV to every table |
| Method code | `scripts_v2/` |
| Paper-run drivers | `tools/run_v42b_ucparity.py`, `tools/run_v42b_extras.py` |
| No-hardcoded-results check | `tools/check_no_hardcoded_results.py` |
| Test suite | `tests/` (pytest) |

## The CI check

`tools/check_no_hardcoded_results.py` enforces two rules and exits non-zero on
any violation:

1. **No metric literals in source.** A dict in `scripts_v2/` pairing a known
   result key (`LPSP_pct`, `diesel_kWh_per_year`, `total_cost_GHS`, ...) with a
   numeric literal is treated as a hardcoded result.
2. **Every results manifest records a git commit.**

## Data provenance

The NASA POWER download date was not recorded. The manifest carries what is
knowable: the sha256 of the file and its modification timestamps.

## Determinism

Every stochastic component takes an explicit seed. `load_v2` regenerates
byte-stably from seed 42. Solver results use HiGHS with a fixed MIP gap; MILP
solutions can differ in the last digits between solver builds, so invariants
are asserted to 1e-6 rather than exactly.

## Load variants

`MG_LOAD` selects the demand series: unset or `v1`, `v2` (autocorrelated, used
for the paper), `mix12` (1.2% non-household). Results are written under
`results/`, `results/v2/`, `results/mix12/` respectively, so a variant run cannot
overwrite another variant's results.
## Run order

1. Create a venv and `pip install -r requirements.lock`.
2. `python reproduce.py --stage data` then `python reproduce.py --verify`.
3. Paper walk-forward: `python tools/run_v42b_ucparity.py` (long; resume-safe).
4. Extras/figures: `python tools/run_v42b_extras.py`.
5. Optional: `python -m pytest tests`.
